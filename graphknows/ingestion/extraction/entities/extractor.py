"""Dynamic spaCy-hinted GLiNER2 extractor for the memory ingestion pipeline.

spaCy runs once per chunk and mines the ENTITY schema GLiNER2 receives for that
same chunk (entity labels come from spaCy NER). RELATION labels are an open,
per-chunk mined vocabulary: dependency-path triggers between entity pairs,
canonicalized to a verb lemma via WordNet, with FrameNet frame names taking
precedence when a sentence evokes a frame covering the pair (frame-first
labeling). A triplet is only kept when an independent signal (dep path and/or
frame) corroborates the GLiNER2 extraction for that entity pair.
"""

from __future__ import annotations

import contextlib
import logging
import os
import re
import sys
import time
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

from graphknows.ingestion.extraction.entities.gliner_model import load_gliner2_model
from graphknows.ingestion.extraction.relations.canon import canonical_verb
from graphknows.ingestion.extraction.relations.llm_free import clean_relation_object

log = logging.getLogger(__name__)

_BASE_THRESHOLD: float = 0.35
# Chunks per GLiNER2 forward pass in extract_batch. Larger = better throughput,
# more peak RAM. Tunable for the ingest-bound path.
_BATCH_SIZE: int = int(os.environ.get("GRAPHKNOWS_GLINER_BATCH_SIZE", "16"))
_SPACY_MODEL = "en_core_web_lg"
_MIN_ENTITY_LABEL_COUNT = 1
_MIN_RELATION_TRIGGER_COUNT = 1
_MAX_ENTITY_LABELS = 24
_MAX_RELATION_LABELS = 64
_MAX_RELATION_FRAMES = 4
_TRIGGER_POS = frozenset({"VERB", "NOUN"})
# Semantically empty verbs — never useful as open-vocabulary edge types.
# Mirrors the frames channel's LIGHT_VERBS (kept local: that constant is tuned
# for frame-target selection, this one for relation labels).
_LIGHT_VERBS = frozenset(
    {"be", "have", "do", "get", "make", "take", "go", "say", "let", "come", "want"}
)


@dataclass
class ExtractionResult:
    """Structured result from GLiNER2 extraction."""

    entities: list[dict[str, Any]] = field(default_factory=list)
    relations: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class _EntityHint:
    text: str
    label: str
    root: Any
    sent: Any


@dataclass(frozen=True)
class _DynamicSchema:
    entity_spec: dict[str, dict[str, Any]]
    relation_spec: dict[str, str]
    label_to_type: dict[str, str]
    # label -> {(sorted entity-pair casefold texts) -> corroborating signals
    # ("dep" and/or "frame")}. Consumed by the _map_raw agreement gate.
    relation_support: dict[str, dict[tuple[str, str], set[str]]] = field(default_factory=dict)


def _relationship_terms() -> frozenset[str]:
    """Kinship / social-role words from the taxonomy — never valid fact objects.

    Empty on config failure (object cleaning simply skips the role check).
    """
    try:
        from graphknows.ingestion.models.taxonomy import get_default_taxonomy

        group = get_default_taxonomy().entity_groups.get("RELATIONSHIP")
        return frozenset(a.lower() for a in (group.aliases if group else []))
    except Exception as exc:  # pragma: no cover - config guard
        log.warning("relationship-role vocabulary unavailable: %s", exc)
        return frozenset()


def _schema_label(value: str) -> str:
    """Return a GLiNER-friendly label derived only from observed text metadata."""
    label = re.sub(r"[^a-z0-9]+", "_", value.casefold()).strip("_")
    return label or "entity"


def _dependency_path(left: Any, right: Any) -> list[Any]:
    """Return the shortest undirected dependency path between two spaCy tokens."""
    left_path = [left, *left.ancestors]
    right_path = [right, *right.ancestors]
    right_positions = {token.i: idx for idx, token in enumerate(right_path)}
    for left_idx, token in enumerate(left_path):
        right_idx = right_positions.get(token.i)
        if right_idx is None:
            continue
        return left_path[:left_idx] + right_path[right_idx::-1]
    return []


def _iter_entity_pairs(entities: list[_EntityHint]) -> Iterable[tuple[_EntityHint, _EntityHint]]:
    for idx, left in enumerate(entities):
        for right in entities[idx + 1 :]:
            if left.text.casefold() != right.text.casefold():
                yield left, right


def _is_ancestor(ancestor: Any, token: Any) -> bool:
    return any(item.i == ancestor.i for item in token.ancestors)


def _relation_trigger_candidates(path: list[Any]) -> list[Any]:
    """Return broad lexical relation triggers from a dependency path.

    Prefer content-bearing predicates. Bare prepositions/adjectives create labels
    that are too narrow for GLiNER2, and high-level verbs that dominate another
    candidate on the same path tend to express reporting/framing rather than the
    entity-to-entity relation itself.
    """
    candidates = [
        token
        for token in path
        if token.pos_ in _TRIGGER_POS and (token.lemma_ or token.text).strip()
    ]
    return [
        token
        for token in candidates
        if not any(other.i != token.i and _is_ancestor(token, other) for other in candidates)
    ]


def _token_text(token: Any) -> str:
    return str(getattr(token, "text", "") or "").strip()


def _path_text(path: list[Any]) -> str:
    parts = [_token_text(token) for token in path]
    return " -> ".join(part for part in parts if part)


def _relation_frame(left: _EntityHint, trigger: Any, right: _EntityHint, path: list[Any]) -> str:
    trigger_lemma = (trigger.lemma_ or trigger.text).strip()
    frame = f"{left.text} -> {trigger_lemma} -> {right.text}"
    path_hint = _path_text(path)
    if path_hint:
        frame = f"{frame} (dependency path: {path_hint})"
    return frame


def _relation_description(lemma: str, pos: str, frames: list[str]) -> str:
    description = (
        f"A relation between two concise entity or noun-phrase arguments expressed by "
        f"the observed {pos.lower()} trigger '{lemma}' or a close paraphrase. "
        "Prefer direct subject, object, complement, or prepositional-object arguments; "
        "avoid selecting whole clauses as relation heads or tails."
    )
    if frames:
        joined = "; ".join(frames[:_MAX_RELATION_FRAMES])
        description = f"{description} Observed dependency frames in this chunk: {joined}."
    return description


def _frame_relation_description(label: str, frame: str, trigger_lemma: str) -> str:
    """Role-aware GLiNER2 description for a FrameNet-derived relation label."""
    from graphknows.channels._framenet import _core_fes, _frame_def

    cores = _core_fes(frame)
    if len(cores) >= 2:
        roles = f"subject is the {cores[0]}, object is the {cores[1]}"
    elif cores:
        roles = f"subject is the {cores[0]}"
    else:
        roles = "subject and object fill the frame's roles"
    definition = _frame_def(frame)[:160].strip()
    body = f"{roles}. {definition}" if definition else roles
    return f"{label}: {body} (FrameNet {frame}; trigger: '{trigger_lemma}')"


def _argument_fillers(token: Any) -> list[str]:
    """Argument texts of a predicate token from the dependency parse.

    Subject/object/complement children plus clause PPs (reusing the frames
    channel's compound-aware ``_filler`` and ``_clause_pps``), then one hop
    into each argument's own PP so "a nurse at the county hospital" yields
    both 'nurse' and 'county hospital'.
    """
    from graphknows.channels._framenet import _clause_pps, _filler

    queue = [
        child
        for child in token.children
        if child.dep_ in ("nsubj", "nsubjpass", "dobj", "attr", "acomp")
    ]
    queue.extend(pobj for _prep, pobj in _clause_pps(token))
    out: list[str] = []
    seen: set[str] = set()
    for tok in queue:  # queue grows while iterating (one PP hop per argument)
        filler = _filler(tok, "", [])
        if filler and filler.casefold() not in seen:
            seen.add(filler.casefold())
            out.append(filler)
        for child in tok.children:
            if child.dep_ == "prep":
                queue.extend(p for p in child.children if p.dep_ == "pobj")
    return out


def _pair_signals(
    support: dict[tuple[str, str], set[str]], head: str, tail: str
) -> set[str]:
    """Corroborating signals mined for the (head, tail) entity pair, if any.

    Matching is lenient (:func:`_text_match`) and direction-agnostic — the
    mined pair is doc-ordered while GLiNER2 chooses its own head/tail, and the
    tail may have been shortened by object cleaning.
    """
    h, t = head.casefold(), tail.casefold()
    for (a, b), signals in support.items():
        if (_text_match(h, a) and _text_match(t, b)) or (
            _text_match(h, b) and _text_match(t, a)
        ):
            return signals
    return set()


def _text_match(a: str, b: str) -> bool:
    """Lenient casefolded argument match: equal, or a >=4-char containment.

    The length floor stops short-string false positives ('us' in 'justin').
    """
    if a == b:
        return True
    short, long_ = (a, b) if len(a) <= len(b) else (b, a)
    return len(short) >= 4 and short in long_


# Mining-side frame acceptance: top-1 candidate with a lenient similarity
# floor and NO margin gate. The frames channel's absolute floors (0.35/0.42)
# were tuned on the local bge embedder and never pass with lower-cosine remote
# embedders, while top-1 RANKING stays correct. Labeling can afford leniency:
# triplet precision is enforced downstream by the dep/frame agreement gate.
_MINE_SIM_FLOOR = 0.15


def _mine_sentence_frames(sents: list[Any]) -> list[dict[int, tuple[str, float, Any]]]:
    """Top-ranked FrameNet frames per sentence: ``{token.i: (frame, sim, token)}``.

    Reuses the frames channel's LU lookup and frame definitions
    (:mod:`graphknows.channels._framenet`), but batches ALL sentence and
    frame-definition embeddings for the chunk into ONE embedder call — the
    channel's per-sentence ``_disambiguate`` would be too slow at ingest.
    """
    from graphknows.channels._framenet import _embed_norm, _frame_def, _lus, _sentence_targets

    lui = _lus()
    per_sent = [_sentence_targets(sent, lui) for sent in sents]
    all_defs = sorted({f for targets in per_sent for _, _, frames in targets for f in frames})
    if not all_defs:
        return [{} for _ in sents]
    stexts = [sent.text.strip() for sent in sents]
    embs = _embed_norm(stexts + [_frame_def(f) for f in all_defs])
    def_vec = {f: embs[len(sents) + i] for i, f in enumerate(all_defs)}
    out: list[dict[int, tuple[str, float, Any]]] = []
    for si, targets in enumerate(per_sent):
        accepted: dict[int, tuple[str, float, Any]] = {}
        for tok, _pos, frames in targets:
            sim_of = {f: float(def_vec[f] @ embs[si]) for f in frames}
            top = max(frames, key=lambda f: sim_of[f])
            if sim_of[top] >= _MINE_SIM_FLOOR:
                accepted[tok.i] = (top, sim_of[top], tok)
        out.append(accepted)
    return out


class _SpacySchemaMiner:
    """Mine GLiNER2 schema labels from spaCy NER and dependency structure."""

    def __init__(
        self,
        model_name: str = _SPACY_MODEL,
        nlp_loader: Callable[[str], Any] | None = None,
        frame_fn: Callable[[list[Any]], list[dict[int, tuple[str, float, Any]]]] | None = None,
    ) -> None:
        self._model_name = model_name
        self._nlp_loader = nlp_loader
        self._nlp: Any | None = None
        self._frame_fn = frame_fn or _mine_sentence_frames

    def _ensure_loaded(self) -> None:
        if self._nlp is not None:
            return
        loader = self._nlp_loader
        if loader is None:
            import spacy

            loader = spacy.load
        try:
            self._nlp = loader(self._model_name)
        except Exception as exc:
            raise RuntimeError(
                f"spaCy model '{self._model_name}' is required for GLiNER2 schemas. "
                f"Install it with:  python -m spacy download {self._model_name}"
            ) from exc

    @property
    def nlp(self) -> Any:
        """The loaded spaCy pipeline (loads on first access)."""
        self._ensure_loaded()
        return self._nlp

    def build(self, text: str) -> _DynamicSchema:
        self._ensure_loaded()
        assert self._nlp is not None
        return self._build_from_doc(self._nlp(text))

    def build_batch(self, texts: list[str]) -> list[_DynamicSchema]:
        """Mine schemas for many chunks via one batched ``nlp.pipe`` pass."""
        self._ensure_loaded()
        assert self._nlp is not None
        return [self._build_from_doc(doc) for doc in self._nlp.pipe(texts)]

    def _build_from_doc(self, doc: Any) -> _DynamicSchema:
        hints = [
            _EntityHint(
                text=ent.text.strip(),
                label=ent.label_.strip(),
                root=ent.root,
                sent=ent.sent,
            )
            for ent in doc.ents
            if ent.text.strip() and ent.label_.strip()
        ]

        entity_counts = Counter(hint.label for hint in hints)
        label_to_type: dict[str, str] = {}
        entity_spec: dict[str, dict[str, Any]] = {}
        for label, count in entity_counts.most_common(_MAX_ENTITY_LABELS):
            if count < _MIN_ENTITY_LABEL_COUNT:
                continue
            schema_name = _schema_label(label)
            label_to_type[schema_name] = label
            entity_spec[schema_name] = {
                "description": (f"Named entity with the label {label}, observed in this chunk."),
                "threshold": _BASE_THRESHOLD,
            }

        by_sentence: dict[int, list[_EntityHint]] = {}
        sent_by_start: dict[int, Any] = {}
        for hint in hints:
            by_sentence.setdefault(hint.sent.start, []).append(hint)
            sent_by_start[hint.sent.start] = hint.sent

        # Frame evocation on every sentence with an entity: FE fillers come from
        # the dependency parse, so a frame can corroborate a pair even when only
        # one argument is a named entity ("Melanie works as a nurse").
        starts = list(by_sentence)
        frame_maps = self._frame_fn([sent_by_start[s] for s in starts]) if starts else []
        frames_by_start = dict(zip(starts, frame_maps, strict=True))

        label_counts: Counter[str] = Counter()
        lemma_info: dict[str, tuple[str, str]] = {}  # label -> (canonical lemma, pos)
        lemma_frames: dict[str, list[str]] = {}  # label -> dep-frame hint strings
        frame_info: dict[str, tuple[str, str]] = {}  # label -> (frame name, trigger)
        relation_support: dict[str, dict[tuple[str, str], set[str]]] = {}

        def _support_pair(label: str, a: str, b: str, signal: str) -> None:
            pair: tuple[str, str] = tuple(sorted((a.casefold(), b.casefold())))  # type: ignore[assignment]
            relation_support.setdefault(label, {}).setdefault(pair, set()).add(signal)

        def _support(label: str, left: _EntityHint, right: _EntityHint, signal: str) -> None:
            _support_pair(label, left.text, right.text, signal)

        for start, sent_entities in by_sentence.items():
            accepted = frames_by_start.get(start, {})
            entity_roots = {hint.root.i for hint in sent_entities}

            def _lemma_of(token: Any) -> str:
                """Canonical trigger lemma, or '' when the token is not a predicate."""
                if token.i in entity_roots:
                    return ""  # an entity span is an argument, never a predicate
                raw = (token.lemma_ or token.text).strip()
                lemma = canonical_verb(raw, token.pos_)
                if lemma in _LIGHT_VERBS:
                    return ""
                # NOUN triggers must WordNet-derive a DISTINCT verb (adoption ->
                # adopt, camping -> camp); homographic nouns (cabin, winter)
                # stay arguments, not predicates.
                if token.pos_ != "VERB" and lemma == raw.casefold():
                    return ""
                return lemma

            for left, right in _iter_entity_pairs(sent_entities):
                path = _dependency_path(left.root, right.root)
                # Frame-first, per trigger: an accepted frame on a verb/gerund
                # target of the path labels the pair with the frame name
                # (checked on the FULL path — the domination filter below
                # prefers specific nouns and would hide the frame-evoking verb).
                framed_ids: set[int] = set()
                for token in path:
                    frame_hit = accepted.get(token.i)
                    if frame_hit is None or token.i in entity_roots:
                        continue
                    if token.pos_ != "VERB" and not token.text.lower().endswith("ing"):
                        continue  # frame labels come from predicates, not argument nouns
                    frame_name = frame_hit[0]
                    label = _schema_label(frame_name).upper()
                    raw_lemma = (token.lemma_ or token.text).strip().casefold()
                    frame_info.setdefault(label, (frame_name, raw_lemma))
                    label_counts[label] += 1
                    _support(label, left, right, "dep")
                    _support(label, left, right, "frame")
                    framed_ids.add(token.i)
                # Un-framed triggers still get their (canonical) lemma label —
                # frame-first is per trigger, not per pair.
                rest = [token for token in path if token.i not in framed_ids]
                triggers = [
                    (token, lemma)
                    for token in _relation_trigger_candidates(rest)
                    if (lemma := _lemma_of(token))
                ]
                if not triggers:
                    # Domination filter can leave only argument nouns; fall back
                    # to the path's verbs so the predicate is not lost
                    # ("Caroline adopted a dog named Biscuit" -> adopt).
                    triggers = [
                        (token, lemma)
                        for token in rest
                        if token.pos_ == "VERB" and (lemma := _lemma_of(token))
                    ]
                for token, lemma in triggers:
                    label = _schema_label(lemma)
                    lemma_info.setdefault(label, (lemma, token.pos_))
                    label_counts[label] += 1
                    _support(label, left, right, "dep")
                    # The trigger's parse arguments corroborate too — GLiNER2
                    # often picks a common-noun object ("adopted a dog") that
                    # is not a named entity.
                    fillers = _argument_fillers(token)
                    for i, a in enumerate(fillers):
                        for b in fillers[i + 1 :]:
                            _support_pair(label, a, b, "dep")
                    frame = _relation_frame(left, token, right, path)
                    frames = lemma_frames.setdefault(label, [])
                    if frame not in frames and len(frames) < _MAX_RELATION_FRAMES:
                        frames.append(frame)

            # Frame-only corroboration: an accepted frame whose target has >=2
            # parse arguments supports the frame label for those filler pairs —
            # this is what reaches relations whose object is a common noun
            # ("Melanie works as a nurse"). Anchored: at least one filler must
            # match a named entity in the sentence.
            for frame_name, _sim, tok in accepted.values():
                fillers = _argument_fillers(tok)
                if len(fillers) < 2:
                    continue
                anchored = any(
                    _text_match(f.casefold(), hint.text.casefold())
                    for f in fillers
                    for hint in sent_entities
                )
                if not anchored:
                    continue
                label = _schema_label(frame_name).upper()
                frame_info.setdefault(
                    label, (frame_name, (tok.lemma_ or tok.text).strip().casefold())
                )
                label_counts[label] += 1
                for i, a in enumerate(fillers):
                    for b in fillers[i + 1 :]:
                        _support_pair(label, a, b, "frame")

        relation_spec: dict[str, str] = {}
        for label, count in label_counts.most_common(_MAX_RELATION_LABELS):
            if count < _MIN_RELATION_TRIGGER_COUNT:
                continue
            if label in frame_info:
                frame_name, trigger = frame_info[label]
                relation_spec[label] = _frame_relation_description(label, frame_name, trigger)
            else:
                lemma, pos = lemma_info[label]
                relation_spec[label] = _relation_description(
                    lemma=lemma,
                    pos=pos,
                    frames=lemma_frames.get(label, []),
                )

        return _DynamicSchema(
            entity_spec=entity_spec,
            relation_spec=relation_spec,
            label_to_type=label_to_type,
            relation_support=relation_support,
        )


class GLiNER2EntityExtractor:
    """GLiNER2 extractor with dynamic spaCy-derived schemas.

    The GLiNER2 model is loaded lazily once.  The schema is rebuilt for every
    chunk from spaCy signals, so entity and relation labels adapt to the input
    text without taxonomy files or domain-specific constants.
    """

    def __init__(
        self,
        model_loader: Callable[[], Any] = load_gliner2_model,
        schema_miner: _SpacySchemaMiner | None = None,
        spacy_model: str = "",
        relation_schema: dict[str, str] | None = None,
    ) -> None:
        self._gliner: Any | None = None
        self._model_loader = model_loader
        if not spacy_model:
            from graphknows.settings import GraphKnowsSettings

            spacy_model = GraphKnowsSettings().spacy_model
        self._schema_miner = schema_miner or _SpacySchemaMiner(model_name=spacy_model)
        # Open-vocabulary relations: the per-chunk mined relation_spec (WordNet
        # canonical lemmas + FrameNet frame labels) IS the schema. An explicit
        # *relation_schema* (escape hatch for taxonomy-guided extraction) is
        # honored when passed, and bypasses the mined-corroboration gate.
        self._relation_schema: dict[str, str] = dict(relation_schema) if relation_schema else {}
        self._relationship_terms: frozenset[str] = _relationship_terms()

    def _ensure_model_loaded(self) -> None:
        """Load GLiNER2 model on first extraction call."""
        if self._gliner is not None:
            return

        log.info("GLiNER2 model not loaded; loading now (first call, expect 15-30s)...")
        start = time.monotonic()
        with contextlib.redirect_stdout(sys.stderr):
            model = self._model_loader()

        if model is None:
            raise RuntimeError(
                "GLiNER2 model failed to load; check RAM (~2 GB required) and "
                "HuggingFace access. No silent fallback."
            )

        self._gliner = model
        log.info("GLiNER2 model loaded in %.1fs", time.monotonic() - start)

    def _build_gliner_schema(self, dynamic_schema: _DynamicSchema) -> Any:
        assert self._gliner is not None
        schema = self._gliner.create_schema()
        if dynamic_schema.entity_spec:
            schema.entities(dynamic_schema.entity_spec)
        relation_spec = self._relation_schema or dynamic_schema.relation_spec
        if relation_spec:
            schema.relations(relation_spec)
        return schema

    def extract(self, text: str) -> ExtractionResult:
        """Extract entities and relations using a spaCy-mined GLiNER2 schema."""
        self._ensure_model_loaded()
        assert self._gliner is not None

        dynamic_schema = self._schema_miner.build(text)
        if not dynamic_schema.entity_spec:
            return ExtractionResult()

        gliner_schema = self._build_gliner_schema(dynamic_schema)
        raw = self._gliner.extract(
            text,
            gliner_schema,
            _BASE_THRESHOLD,
            True,  # format_results
            True,  # include_confidence
            True,  # include_spans
        )
        return self._map_raw(raw, dynamic_schema)

    def extract_batch(self, texts: list[str]) -> list[ExtractionResult]:
        """Extract for many chunks in one batched GLiNER2 forward pass.

        Same per-text result as :meth:`extract`, but the model runs the chunks as
        padded batches (``batch_extract``) instead of one call each — the win is
        amortized tokenization + fewer ONNX ``session.run`` calls, which is where
        the CPU-bound ingest spends its time. Chunks whose spaCy schema mines no
        entities skip the model entirely (empty result), same as :meth:`extract`.
        """
        if not texts:
            return []
        self._ensure_model_loaded()
        assert self._gliner is not None

        # Mine all chunk schemas in one batched spaCy pass, then batch the GLiNER2
        # forward pass over the chunks that yielded entity labels.
        dynamic_schemas = self._schema_miner.build_batch(texts)
        results: list[ExtractionResult | None] = [None] * len(texts)
        batch_texts: list[str] = []
        batch_schemas: list[Any] = []
        batch_dynamic: list[Any] = []
        batch_idx: list[int] = []
        for i, (text, dynamic_schema) in enumerate(zip(texts, dynamic_schemas, strict=True)):
            if not dynamic_schema.entity_spec:
                results[i] = ExtractionResult()
                continue
            batch_texts.append(text)
            batch_schemas.append(self._build_gliner_schema(dynamic_schema))
            batch_dynamic.append(dynamic_schema)
            batch_idx.append(i)

        if batch_texts:
            raws = self._gliner.batch_extract(
                batch_texts,
                batch_schemas,
                _BATCH_SIZE,
                _BASE_THRESHOLD,
                0,  # num_workers (0 = current process; ONNX is already threaded)
                True,  # format_results
                True,  # include_confidence
                True,  # include_spans
            )
            for i, raw, dynamic_schema in zip(batch_idx, raws, batch_dynamic, strict=True):
                results[i] = self._map_raw(raw, dynamic_schema)

        final = [r if r is not None else ExtractionResult() for r in results]
        # Guardrail: the label space is open — log its size per batch so a
        # runaway vocabulary is visible in ingest logs.
        label_counts = Counter(rel["relation"] for res in final for rel in res.relations)
        if label_counts:
            log.info(
                "open-vocab relation labels: %d distinct across %d chunks; top10=%s",
                len(label_counts),
                len(final),
                label_counts.most_common(10),
            )
        return final

    def _map_raw(self, raw: dict[str, Any], dynamic_schema: _DynamicSchema) -> ExtractionResult:
        """Map one GLiNER2 raw result → ExtractionResult (entities + relations)."""
        entities: list[dict[str, Any]] = []
        for raw_label, spans in (raw.get("entities") or {}).items():
            schema_label = _schema_label(raw_label)
            canonical = dynamic_schema.label_to_type.get(schema_label, raw_label.upper())
            for span in spans or []:
                conf = float(span.get("confidence", 0.0))
                if conf < _BASE_THRESHOLD:
                    continue
                name = span.get("text", "").strip()
                if not name:
                    continue
                entities.append({"name": name, "type": canonical, "score": conf})

        relations: list[dict[str, Any]] = []
        for rel_type, triplets in (raw.get("relation_extraction") or {}).items():
            for triplet in triplets or []:
                head = triplet.get("head", {})
                tail = triplet.get("tail", {})
                src = head.get("text", "").strip()
                tgt = tail.get("text", "").strip()
                if not src or not tgt or src.lower() == tgt.lower():
                    continue  # drop empty and self-relations (X -rel-> X is noise)
                # Canonicalize the object to a content noun; drop clause /
                # question-echo / wrong-type objects at extraction.
                tgt = clean_relation_object(tgt, self._schema_miner.nlp, self._relationship_terms)
                if not tgt or src.lower() == tgt.lower():
                    continue
                score = min(
                    float(head.get("confidence", 0.0)),
                    float(tail.get("confidence", 0.0)),
                )
                if score < _BASE_THRESHOLD:
                    continue
                # Agreement gate (open-vocab path only): keep the triplet only
                # when a mined dep-path trigger and/or an accepted frame
                # corroborates THIS label for THIS entity pair. Both signals
                # agreeing earns a confidence boost (multiple independent
                # extractors landed on the same fact).
                if self._relation_schema:
                    source = "gliner"  # taxonomy escape hatch: no mined support
                else:
                    signals = _pair_signals(
                        dynamic_schema.relation_support.get(rel_type, {}), src, tgt
                    )
                    if not signals:
                        continue
                    if "dep" in signals and "frame" in signals:
                        score = min(1.0, score * 1.15)
                    source = "gliner+" + "+".join(
                        s for s in ("dep", "frame") if s in signals
                    )
                src_type = next((e["type"] for e in entities if e["name"] == src), "ENTITY")
                tgt_type = next((e["type"] for e in entities if e["name"] == tgt), "ENTITY")
                relations.append(
                    {
                        "head": src,
                        "head_type": src_type,
                        "relation": rel_type,
                        "tail": tgt,
                        "tail_type": tgt_type,
                        "score": score,
                        "source": source,
                    }
                )

        return ExtractionResult(entities=entities, relations=relations)

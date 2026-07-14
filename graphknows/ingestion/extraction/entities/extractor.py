"""Dynamic spaCy-hinted GLiNER2 extractor for the memory ingestion pipeline.

spaCy runs once per chunk. It mines the ENTITY schema GLiNER2 receives (entity
labels come from spaCy NER) and — the primary source of relations — the
subject-centric SVO triples of every clause (see
:mod:`graphknows.ingestion.extraction.relations.svo`): the open-vocabulary
predicate is what the text says (WORK_AS, LOSE_AT, ADOPT), the subject resolves
to the utterance speaker for first/second-person clauses, and date/phatic/
dangling noise is dropped at the source.

GLiNER2 is a *validator*, not a relation source. The RELATION schema handed to
GLiNER2 is built FROM the mined SVO predicate labels, so an SVO triple is
promoted to ``svo+gliner`` (with GLiNER2's confidence) when GLiNER2 independently
emits that predicate with a matching object. An SVO triple GLiNER2 does not
corroborate is kept only when both arguments are named (a high-confidence
structural fact); otherwise it is dropped as idiom / parse noise.
"""

from __future__ import annotations

import contextlib
import logging
import os
import re
import sys
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from graphknows.ingestion.extraction.entities.gliner_model import load_gliner2_model
from graphknows.ingestion.extraction.relations.svo import extract_relations, parse_speaker

log = logging.getLogger(__name__)

_BASE_THRESHOLD: float = 0.35
# Chunks per GLiNER2 forward pass in extract_batch. Larger = better throughput,
# more peak RAM. Tunable for the ingest-bound path.
_BATCH_SIZE: int = int(os.environ.get("GRAPHKNOWS_GLINER_BATCH_SIZE", "16"))
# GLiNER2 attends better to short spans: the ENTITY + relation pass runs on
# ~128-token / 32-overlap sub-windows of each 384-token retrieval chunk rather
# than the whole blob — more entities, and (unlike per-sentence) windows still
# hold the 2+ entities a relation needs. Windows attribute back to the parent
# chunk_id; the spaCy/SVO pass still sees the full chunk. Tokens are approximated
# by whitespace words (no tokenizer dependency); both knobs are tunable.
_EXTRACT_WINDOW_TOKENS: int = 128
_EXTRACT_WINDOW_OVERLAP: int = 32
_SPACY_MODEL = "en_core_web_lg"
_MIN_ENTITY_LABEL_COUNT = 1
_MAX_ENTITY_LABELS = 24
# SVO mines one predicate per clause verb, so a multi-turn dialogue chunk yields
# many distinct labels — raised from the entity-pair era's 64 so GLiNER2's
# validation schema covers them (only the labels in the schema can be
# corroborated to svo+gliner; uncorroborated triples still survive structurally).
_MAX_RELATION_LABELS = 96


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
    # Relation triples mined for the chunk (the candidate facts): verb-SVO plus
    # rich dependency-grammar triples (copular/possessive/appositive/noun-prep),
    # each dict as returned by ``extract_relations``; consumed by the _map_raw
    # corroboration/structural gate (rich triples ride the structural keep).
    svo_triples: list[dict[str, Any]] = field(default_factory=list)


def _schema_label(value: str) -> str:
    """Return a GLiNER-friendly label derived only from observed text metadata."""
    label = re.sub(r"[^a-z0-9]+", "_", value.casefold()).strip("_")
    return label or "entity"


def _text_match(a: str, b: str) -> bool:
    """Lenient casefolded argument match: equal, or a >=4-char containment.

    The length floor stops short-string false positives ('us' in 'justin').
    """
    if a == b:
        return True
    short, long_ = (a, b) if len(a) <= len(b) else (b, a)
    return len(short) >= 4 and short in long_


def _svo_relation_description(verb: str, prep: str) -> str:
    """GLiNER2 validator description for a mined SVO predicate label."""
    phrase = f"{verb} {prep}".strip()
    return (
        f"A factual relation where the subject '{phrase}' the object. The object "
        "is a concise entity or noun phrase (a person, place, organization, role, "
        "or thing) — never a clause, question, or date."
    )


def _line_speakers(text: str) -> list[tuple[int, str]]:
    """(char offset of each line start, its LoCoMo speaker) for a chunk's turns.

    A chunk is newline-delimited "ts | Speaker: text" turn-lines; a line with no
    speaker prefix (e.g. a section heading) maps to "".
    """
    offsets: list[tuple[int, str]] = []
    pos = 0
    for line in text.split("\n"):
        offsets.append((pos, parse_speaker(line)))
        pos += len(line) + 1  # + the '\n' consumed by split
    return offsets


def _speaker_at(line_speakers: list[tuple[int, str]], char: int) -> str:
    """Speaker of the turn-line containing character offset *char* ('' if none)."""
    speaker = ""
    for start, sp in line_speakers:
        if start > char:
            break
        speaker = sp
    return speaker


def _gliner_corroboration(
    triplets: list[dict[str, Any]] | None, svo_head: str, svo_tail: str
) -> float | None:
    """GLiNER2 confidence when a triplet under this label agrees with the SVO
    triple on its object OR its (resolved) subject, else None.

    Object agreement is the primary signal. Subject agreement is the fallback:
    GLiNER2 often names the same object by a coreferent surface the SVO head-noun
    does not share ("a cat named Luna" -> GLiNER2 'Luna', SVO 'cat'), so a match
    on the resolved speaker keeps the fact with the SVO parse's cleaner object.
    Matching is lenient (:func:`_text_match`); a sub-threshold GLiNER2 confidence
    is not a validation.
    """
    head, tail = svo_head.casefold(), svo_tail.casefold()
    for triplet in triplets or []:
        g_head = (triplet.get("head", {}).get("text") or "").strip().casefold()
        g_tail = (triplet.get("tail", {}).get("text") or "").strip().casefold()
        if (g_tail and _text_match(tail, g_tail)) or (g_head and _text_match(head, g_head)):
            conf = min(
                float(triplet.get("head", {}).get("confidence", 0.0)),
                float(triplet.get("tail", {}).get("confidence", 0.0)),
            )
            if conf >= _BASE_THRESHOLD:
                return conf
    return None


def _sub_windows(text: str) -> list[str]:
    """Slide a ~128-word / 32-overlap window over a chunk for the GLiNER2 pass.

    Windows snap to whitespace so no word is split; a chunk that fits in one
    window is returned unchanged (so short turns cost exactly one forward pass).
    """
    words = text.split()
    step = _EXTRACT_WINDOW_TOKENS - _EXTRACT_WINDOW_OVERLAP
    if len(words) <= _EXTRACT_WINDOW_TOKENS or step <= 0:
        return [text]
    windows: list[str] = []
    for start in range(0, len(words), step):
        windows.append(" ".join(words[start : start + _EXTRACT_WINDOW_TOKENS]))
        if start + _EXTRACT_WINDOW_TOKENS >= len(words):
            break
    return windows


def _merge_raw(acc: dict[str, Any] | None, raw: dict[str, Any]) -> dict[str, Any]:
    """Fold one sub-window's GLiNER2 raw result into a per-chunk accumulator.

    Entity spans dedup by casefolded text (keep the max confidence) so the same
    name seen in overlapping windows becomes one entity; relation triplets (used
    only for SVO corroboration lookup) are concatenated per label.
    """
    if acc is None:
        acc = {"entities": {}, "relation_extraction": {}}
    ents: dict[str, list[dict[str, Any]]] = acc["entities"]
    for label, spans in (raw.get("entities") or {}).items():
        bucket = ents.setdefault(label, [])
        for span in spans or []:
            name = (span.get("text") or "").strip().casefold()
            existing = next(
                (s for s in bucket if (s.get("text") or "").strip().casefold() == name), None
            )
            if existing is None:
                bucket.append(dict(span))
            elif float(span.get("confidence", 0.0)) > float(existing.get("confidence", 0.0)):
                existing["confidence"] = span.get("confidence")
    rels: dict[str, list[Any]] = acc["relation_extraction"]
    for label, triplets in (raw.get("relation_extraction") or {}).items():
        rels.setdefault(label, []).extend(triplets or [])
    return acc


class _SpacySchemaMiner:
    """Mine GLiNER2 schema labels from spaCy NER and SVO relation structure."""

    def __init__(
        self,
        model_name: str = _SPACY_MODEL,
        nlp_loader: Callable[[str], Any] | None = None,
    ) -> None:
        self._model_name = model_name
        self._nlp_loader = nlp_loader
        self._nlp: Any | None = None

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
        # --- ENTITY schema (spaCy NER labels) ---
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

        # --- SVO relations (primary) ---
        # Resolve each sentence's speaker from the turn-line it starts in, so a
        # first/second-person clause subject ("I work as a nurse") anchors to the
        # utterance speaker. The chunk's distinct speakers drive 2nd-person
        # resolution ("you" -> the other of a two-party dialogue).
        line_speakers = _line_speakers(doc.text)
        speakers = sorted({s for _, s in line_speakers if s})
        svo_triples: list[dict[str, Any]] = []
        label_verb: dict[str, tuple[str, str]] = {}
        label_counts: Counter[str] = Counter()
        for sent in doc.sents:
            speaker = _speaker_at(line_speakers, sent.start_char)
            for triple in extract_relations(sent, speaker, speakers):
                svo_triples.append(triple)
                # Only verb-SVO predicates join the GLiNER2 validation schema;
                # rich structural triples (IS_A/HAS/prep) ride the structural
                # keep in _map_raw and carry no verb/prep to describe.
                if triple.get("pattern", "svo") != "svo":
                    continue
                label = triple["relation"]
                label_counts[label] += 1
                label_verb.setdefault(label, (triple["verb"], triple["prep"]))

        # The GLiNER2 RELATION schema IS the mined SVO predicate vocabulary —
        # GLiNER2 looks only for the predicates the text actually stated.
        relation_spec: dict[str, str] = {}
        for label, _count in label_counts.most_common(_MAX_RELATION_LABELS):
            verb, prep = label_verb[label]
            relation_spec[label] = _svo_relation_description(verb, prep)

        return _DynamicSchema(
            entity_spec=entity_spec,
            relation_spec=relation_spec,
            label_to_type=label_to_type,
            svo_triples=svo_triples,
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
        # Open-vocabulary relations: the per-chunk mined SVO predicate labels ARE
        # the schema. An explicit *relation_schema* (escape hatch for
        # taxonomy-guided extraction) is honored when passed.
        self._relation_schema: dict[str, str] = dict(relation_schema) if relation_schema else {}

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

        # Run the GLiNER2 pass on sub-windows of the chunk and merge each window's
        # entities/relations back onto the parent chunk (schema built once).
        gliner_schema = self._build_gliner_schema(dynamic_schema)
        merged: dict[str, Any] | None = None
        for window in _sub_windows(text):
            raw = self._gliner.extract(
                window,
                gliner_schema,
                _BASE_THRESHOLD,
                True,  # format_results
                True,  # include_confidence
                True,  # include_spans
            )
            merged = _merge_raw(merged, raw)
        return self._map_raw(merged or {}, dynamic_schema)

    def extract_batch(self, texts: list[str]) -> list[ExtractionResult]:
        """Extract for many chunks in one batched GLiNER2 forward pass.

        Same per-text result as :meth:`extract`, but the model runs the chunks'
        sub-windows as padded batches (``batch_extract``) instead of one call
        each — the win is amortized tokenization + fewer ONNX ``session.run``
        calls, which is where the CPU-bound ingest spends its time. Chunks whose
        spaCy schema mines no entities skip the model entirely (empty result),
        same as :meth:`extract`.
        """
        if not texts:
            return []
        self._ensure_model_loaded()
        assert self._gliner is not None

        # Mine all chunk schemas in one batched spaCy pass. Each chunk that
        # yielded entity labels is expanded into ~128-token sub-windows; ALL
        # sub-windows across ALL chunks go into ONE batched GLiNER2 forward pass
        # (schema built once per chunk, reused across its windows), then each
        # window's raw result is merged back onto its parent chunk before mapping.
        dynamic_schemas = self._schema_miner.build_batch(texts)
        results: list[ExtractionResult | None] = [None] * len(texts)
        sub_texts: list[str] = []
        sub_schemas: list[Any] = []
        sub_parent: list[int] = []
        parent_dynamic: dict[int, Any] = {}
        for i, (text, dynamic_schema) in enumerate(zip(texts, dynamic_schemas, strict=True)):
            if not dynamic_schema.entity_spec:
                results[i] = ExtractionResult()
                continue
            gliner_schema = self._build_gliner_schema(dynamic_schema)
            parent_dynamic[i] = dynamic_schema
            for window in _sub_windows(text):
                sub_texts.append(window)
                sub_schemas.append(gliner_schema)
                sub_parent.append(i)

        if sub_texts:
            raws = self._gliner.batch_extract(
                sub_texts,
                sub_schemas,
                _BATCH_SIZE,
                _BASE_THRESHOLD,
                0,  # num_workers (0 = current process; ONNX is already threaded)
                True,  # format_results
                True,  # include_confidence
                True,  # include_spans
            )
            merged: dict[int, dict[str, Any]] = {}
            for parent_i, raw in zip(sub_parent, raws, strict=True):
                merged[parent_i] = _merge_raw(merged.get(parent_i), raw)
            for i, raw in merged.items():
                results[i] = self._map_raw(raw, parent_dynamic[i])

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

        # SVO-primary relations. The spaCy SVO triples mined for this chunk are the
        # candidate facts; GLiNER2 (its relation schema built FROM the SVO
        # predicate labels) is a validator, not an independent source:
        #   * svo+gliner — GLiNER2 emitted a relation under the same predicate
        #     label agreeing on the object (or, as a fallback, the resolved
        #     subject). Kept with GLiNER2's confidence. This rescues clean facts
        #     with a common-noun object ("WORK_AS -> nurse", "ADOPT -> cat") that
        #     structural gating drops.
        #   * svo — no GLiNER2 corroboration; kept at a fixed 0.6 prior ONLY when
        #     BOTH arguments are named (speaker name / spaCy-typed entity), a
        #     high-confidence structural fact. An uncorroborated triple with a
        #     bare common-noun argument is dropped (idiom / parse noise).
        #   * rich structural triples (pattern != 'svo': copular IS_A / possessive
        #     HAS / appositive / noun-prep) are high-precision by construction, so
        #     they are kept at the same 0.6 structural prior when BOTH args are
        #     named OR the pattern is copular/appositive/possessive with a named
        #     head — GLiNER2 agreement is not required (its schema is verb-only).
        _NAMED_HEAD_PATTERNS = ("copular", "appositive", "possessive")
        gliner_rels = raw.get("relation_extraction") or {}
        relations: list[dict[str, Any]] = []
        seen: set[tuple[str, str, str]] = set()
        for triple in dynamic_schema.svo_triples:
            head, label, tail = triple["head"], triple["relation"], triple["tail"]
            key = (head.casefold(), label, tail.casefold())
            if key in seen:
                continue
            pattern = triple.get("pattern", "svo")
            conf = _gliner_corroboration(gliner_rels.get(label), head, tail)
            if conf is not None:
                source, score = "svo+gliner", conf
            elif triple["head_named"] and triple["tail_named"]:
                source, score = "svo", 0.6
            elif pattern in _NAMED_HEAD_PATTERNS and triple["head_named"]:
                source, score = "svo", 0.6
            else:
                continue
            seen.add(key)
            head_type = next(
                (e["type"] for e in entities if e["name"].casefold() == head.casefold()),
                "ENTITY",
            )
            tail_type = next(
                (e["type"] for e in entities if e["name"].casefold() == tail.casefold()),
                "ENTITY",
            )
            relations.append(
                {
                    "head": head,
                    "head_type": head_type,
                    "relation": label,
                    "tail": tail,
                    "tail_type": tail_type,
                    "score": score,
                    "source": source,
                }
            )

        return ExtractionResult(entities=entities, relations=relations)

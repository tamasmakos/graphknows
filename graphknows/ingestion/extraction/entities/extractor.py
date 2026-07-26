"""Dynamic spaCy-hinted relex extractor for the memory ingestion pipeline.

spaCy runs once per chunk and mines the two label vocabularies the extraction
model needs: the ENTITY labels (from spaCy NER) and the open-vocabulary predicate
labels of every clause (see :mod:`graphknows.ingestion.extraction.relations.svo`)
— the predicate is what the text actually says (WORK_AS, LOSE_AT, ADOPT), the
subject resolves to the utterance speaker for first/second-person clauses, and
date/phatic/dangling noise is dropped at the source. Supplying labels the text
itself produced is what makes a label-conditioned extractor usable on open
dialogue at all, so that half of the old design is unchanged.

What changed is the model. GLiNER2 could only *corroborate* a mined SVO triple:
it emitted a predicate, and agreement promoted the triple to ``svo+gliner``.
relex (knowledgator/gliner-relex-large-v1.0) extracts entities and relations
JOINTLY and returns typed, span-grounded head/tail pairs — so its output IS the
fact, not a vote on someone else's. Two relation sources survive:

* ``relex`` — the model emitted (head, predicate, tail) under a mined predicate
  label, kept at its own score.
* ``svo`` — a mined triple relex did not emit, kept at a fixed 0.6 structural
  prior when BOTH arguments are named, or the pattern is copular/appositive/
  possessive with a named head. This retains the durable identity/possession
  facts ("I'm a nurse", "Caroline's dog") that a verb-relation model has no
  predicate label to express.

One forward pass per WHOLE chunk: relex has no fixed max length, so the old
128-token sub-windowing (~4 calls/chunk, needed because GLiNER2 attended poorly
to long spans) is gone.
"""

from __future__ import annotations

import contextlib
import logging
import os
import re
import sys
import time
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from graphknows.ingestion.extraction.entities.gliner_model import load_gliner2_model
from graphknows.ingestion.extraction.relations.svo import (
    extract_relations,
    noun_supersense,
    parse_speaker,
)

log = logging.getLogger(__name__)

_BASE_THRESHOLD: float = float(os.environ.get("GRAPHKNOWS_RELEX_THRESHOLD", "0.35"))
# relex classifies a candidate entity PAIR as bearing a relation; its own card
# recommends a markedly higher floor here than for entity spans (0.7-0.9 vs
# 0.3-0.5), because a low relation threshold pairs up every co-occurring entity.
_RELATION_THRESHOLD: float = float(os.environ.get("GRAPHKNOWS_RELEX_RELATION_THRESHOLD", "0.7"))
# Floor on relex's entity-PAIR candidate score. Without it every co-occurring
# pair of entities is a relation candidate and gets scored against every supplied
# predicate, which manufactures confident nonsense between unrelated spans.
_ADJACENCY_THRESHOLD: float = float(os.environ.get("GRAPHKNOWS_RELEX_ADJACENCY", "0.6"))

# relex is ZERO-SHOT: it can only assign a type it was handed, so feeding it only
# the labels spaCy happened to observe in one chunk collapses the type space —
# a chunk whose NER found just PERSON types "LGBTQ support group" and "lake
# sunrise" as person. This is spaCy's own NER inventory rendered in the natural
# language relex expects, UNIONed with the chunk's observed labels, plus the
# conversational types spaCy has no tag for at all (occupation/animal/activity/
# emotion) — the ones dialogue is actually made of. Not a domain taxonomy: no
# entity is required to exist, and the miner still decides what it saw.
_BASE_ENTITY_LABELS: tuple[str, ...] = (
    "person",
    "organization",
    "location",
    "date",
    "time",
    "event",
    "work_of_art",
    "product",
    "nationality_or_group",
    "facility",
    "language",
    "law",
    "money",
    "quantity",
    "occupation",
    "animal",
    "activity",
    "emotion",
    "food",
    "artifact",
)

# relex has no speaker resolution, so it happily returns "I"/"you"/"we" as PERSON
# spans and would create an ENTITY node called "I" that every speaker's facts
# collapse onto. The SVO path resolves these to the utterance speaker
# (svo._resolve); a relex relation anchored on a bare pronoun carries no
# recoverable referent, so it is dropped and the SVO triple for the same clause
# is the one that survives.
_PRONOUNS: frozenset[str] = frozenset(
    {
        "i",
        "me",
        "my",
        "mine",
        "myself",
        "you",
        "your",
        "yours",
        "yourself",
        "we",
        "us",
        "our",
        "ours",
        "ourselves",
        "he",
        "him",
        "his",
        "she",
        "her",
        "hers",
        "they",
        "them",
        "their",
        "theirs",
        "it",
        "its",
        "this",
        "that",
        "these",
        "those",
    }
)
_SPACY_MODEL = "en_core_web_lg"
_MIN_ENTITY_LABEL_COUNT = 1
_MAX_ENTITY_LABELS = 24
# SVO mines one predicate per clause verb, so a multi-turn dialogue chunk yields
# many distinct labels. relex scores every supplied relation label against every
# candidate entity pair, so this cap is now a COST knob as well as a recall one.
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
    # Casefolded speaker names of the chunk's turn-lines. Model spans sometimes
    # run ACROSS the "Speaker:" colon ("Caroline: Thanks, Melanie" as one
    # PERSON), fragmenting the real speaker node — the mint points reduce such
    # surfaces to the speaker. Empty for non-dialogue text, making that a no-op.
    speakers: frozenset[str] = frozenset()


def _schema_label(value: str) -> str:
    """Return a GLiNER-friendly label derived only from observed text metadata."""
    label = re.sub(r"[^a-z0-9]+", "_", value.casefold()).strip("_")
    return label or "entity"


def _clean_surface(text: str) -> str:
    """Strip enclosing quotes/brackets and trailing punctuation from a span.

    relex returns the span exactly as the text wrote it, so a quoted title comes
    back as ``"Becoming Nicole"`` WITH its quote marks. That is wrong as an entity
    surface, and it is actively corrupting: the store's merge key is derived from
    this string, ArcadeDB strips the quotes when persisting ``name_norm`` but not
    ``name``, so the two fields diverge, a later MERGE cannot find the record it
    just wrote, and the write fails on the ENTITY[name_norm] UNIQUE index — taking
    the caller's whole document down with it.
    """
    cleaned = " ".join((text or "").split())
    prev = None
    while cleaned and cleaned != prev:
        prev = cleaned
        cleaned = cleaned.strip("\"'“”‘’`()[]{}")  # noqa: RUF001 - typographic quotes are the point: they are being stripped
        cleaned = cleaned.strip(" .,;:!?")
    return cleaned


def _arg_type(arg: str, entities: list[dict[str, Any]]) -> str:
    """Type for a relation argument: spaCy/GLiNER2 NER label, else WordNet supersense.

    NER only labels NAMED entities, so before this fell back to a bare "ENTITY"
    the great majority of dialogue relation arguments (common nouns — "a nurse",
    "cat", "job") reached the graph untyped, leaving the type space too degenerate
    to condition on. The supersense ("noun.person", "noun.artifact", ...) is
    already computed inside svo's concreteness gates and was being discarded.

    Returns "" — NOT a placeholder — when the argument is neither NER-typed nor in
    WordNet. ENTITY.type is written first-write-wins (``coalesce``), so emitting a
    placeholder here would let a relation argument squat on the vertex and block
    the real NER label a later chunk assigns to the same entity.
    """
    named = next(
        (e["type"] for e in entities if e["name"].casefold() == arg.casefold()),
        "",
    )
    return named or noun_supersense(arg)


def _canonical_type(raw_label: str, label_to_type: dict[str, str]) -> str:
    """Relex label -> the canonical type the graph stores.

    An OBSERVED label maps back to its spaCy tag (person -> PERSON); a
    base-inventory label spaCy has no tag for (occupation, activity) has
    no mapping and stands on its own, upper-cased. Returning "" here would
    silently demote relex's own typing to the WordNet-supersense fallback.
    """
    raw_label = raw_label.strip()
    if not raw_label:
        return ""
    return label_to_type.get(_schema_label(raw_label), raw_label.upper())


def _svo_relation_description(verb: str, prep: str) -> str:
    """GLiNER2 validator description for a mined SVO predicate label."""
    phrase = f"{verb} {prep}".strip()
    return (
        f"A factual relation where the subject '{phrase}' the object. The object "
        "is a concise entity or noun phrase (a person, place, organization, role, "
        "or thing) — never a clause, question, or date."
    )


def _reduce_speaker_span(name: str, speakers: frozenset[str]) -> str:
    """``"Caroline: Thanks, Melanie"`` -> ``"Caroline"`` when Caroline spoke.

    Chunk text embeds literal ``"[ts |] Speaker: utterance"`` turn-lines, and
    model spans sometimes run across that colon, minting the span verbatim as an
    entity — measured on conv-26, 27 such names carried 42.5% of all REL edges,
    fragmenting the real person node. The semantic referent of such a span is
    the speaker (its facts are things the SPEAKER did), and any real entity in
    the utterance remainder is emitted by the model as its own span anyway.

    Keys strictly on *known speakers of this chunk*: a colon alone never fires,
    so titles like "Charlotte's Web: A Review" pass through, and non-dialogue
    text (empty ``speakers``) makes this the identity function.
    """
    if not speakers or ":" not in name:
        return name
    head = name.split(":", 1)[0]
    if " ".join(head.split()).casefold() in speakers:
        return head.strip()
    return name


def _dedupe_entities(
    raw_entities: list[dict[str, Any]],
    label_to_type: dict[str, str],
    speakers: frozenset[str] = frozenset(),
) -> list[dict[str, Any]]:
    """Score-ranked, merge-key-deduped entity dicts from one relex span list.

    Dedupe by the graph's OWN merge key. ENTITY.name_norm is UNIQUE-indexed
    and written with a batched UNWIND+MERGE, so the same name twice in one
    chunk's list is not a harmless repeat — ArcadeDB raises
    DuplicatedKeyException and the whole DOCUMENT's ingest dies, silently
    removing every chunk in it from the graph. relex runs with
    flat_ner=False and therefore returns OVERLAPPING spans: one surface can
    come back several times under different labels ("Becoming Nicole" as
    both work_of_art and person). Highest score wins the type.
    """
    best: dict[str, dict[str, Any]] = {}
    for span in raw_entities or []:
        conf = float(span.get("score", 0.0) or 0.0)
        if conf < _BASE_THRESHOLD:
            continue
        name = _reduce_speaker_span(_clean_surface(str(span.get("text") or "")), speakers)
        label = str(span.get("label") or "").strip()
        if not name or not label or name.casefold() in _PRONOUNS:
            continue
        # relex echoes the schema key it was given (lowercased spaCy label);
        # map back to the canonical NER type the graph stores.
        canonical = _canonical_type(label, label_to_type)
        key = " ".join(name.split()).casefold()  # mirrors graph_store._name_norm
        prev = best.get(key)
        if prev is None or conf > prev["score"]:
            best[key] = {"name": name, "type": canonical, "score": conf}
    return list(best.values())


def _build_relation(
    head: str,
    label: str,
    tail: str,
    score: float,
    source: str,
    entities: list[dict[str, Any]],
    head_type: str = "",
    tail_type: str = "",
    speakers: frozenset[str] = frozenset(),
) -> dict[str, Any] | None:
    """One relation dict, or None when the argument pair is unusable."""
    # Same cleaning as the entity pass: these strings become ENTITY merge
    # keys via write_relation, so a quoted surface here corrupts the graph
    # exactly as it does there — and a speaker-glued surface here is what
    # attached 42.5% of REL edges to malformed hub names.
    head = _reduce_speaker_span(_clean_surface(head), speakers)
    tail = _reduce_speaker_span(_clean_surface(tail), speakers)
    if not head or not label or not tail or head.casefold() == tail.casefold():
        return None
    if head.casefold() in _PRONOUNS or tail.casefold() in _PRONOUNS:
        return None
    return {
        "head": head,
        "head_type": head_type or _arg_type(head, entities),
        "relation": label,
        "tail": tail,
        "tail_type": tail_type or _arg_type(tail, entities),
        "score": score,
        "source": source,
    }


def _relex_relations(
    raw_relations: list[dict[str, Any]],
    entities: list[dict[str, Any]],
    label_to_type: dict[str, str],
    speakers: frozenset[str] = frozenset(),
) -> list[dict[str, Any]]:
    """Relation dicts for the (head, predicate, tail) triples relex emitted."""
    out: list[dict[str, Any]] = []
    for rel in raw_relations or []:
        score = float(rel.get("score", 0.0) or 0.0)
        if score < _RELATION_THRESHOLD:
            continue
        head_d, tail_d = rel.get("head") or {}, rel.get("tail") or {}
        # relex types its own arguments — richer than the NER label / WordNet
        # supersense fallback, so prefer it and only fall back when absent.
        built = _build_relation(
            str(head_d.get("text") or ""),
            str(rel.get("relation") or "").strip(),
            str(tail_d.get("text") or ""),
            score,
            "relex",
            entities,
            head_type=_canonical_type(str(head_d.get("type") or ""), label_to_type),
            tail_type=_canonical_type(str(tail_d.get("type") or ""), label_to_type),
            speakers=speakers,
        )
        if built is not None:
            out.append(built)
    return out


def _svo_structural_relations(
    svo_triples: list[dict[str, Any]],
    entities: list[dict[str, Any]],
    speakers: frozenset[str] = frozenset(),
) -> list[dict[str, Any]]:
    """Relation dicts for mined triples kept at the fixed structural prior.

    Gate: BOTH arguments named, or a copular/appositive/possessive pattern
    with a named head.
    """
    named_head_patterns = ("copular", "appositive", "possessive")
    out: list[dict[str, Any]] = []
    for triple in svo_triples:
        pattern = triple.get("pattern", "svo")
        if (triple["head_named"] and triple["tail_named"]) or (
            pattern in named_head_patterns and triple["head_named"]
        ):
            pass
        else:
            continue
        built = _build_relation(
            triple["head"],
            triple["relation"],
            triple["tail"],
            0.6,
            "svo",
            entities,
            speakers=speakers,
        )
        if built is not None:
            out.append(built)
    return out


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


def _mine_entity_schema(doc: Any) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    """(entity_spec, label_to_type) mined from a doc's spaCy NER labels."""
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
    return entity_spec, label_to_type


def _mine_svo_schema(doc: Any) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """(svo_triples, relation_spec) mined from a doc's clause structure."""
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
    return svo_triples, relation_spec


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
        entity_spec, label_to_type = _mine_entity_schema(doc)
        svo_triples, relation_spec = _mine_svo_schema(doc)
        return _DynamicSchema(
            entity_spec=entity_spec,
            relation_spec=relation_spec,
            label_to_type=label_to_type,
            svo_triples=svo_triples,
            speakers=frozenset(s.casefold() for _, s in _line_speakers(doc.text) if s),
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
        relation_verifier: Any | None = None,
    ) -> None:
        self._gliner: Any | None = None
        self._model_loader = model_loader
        settings = None
        if not spacy_model:
            from graphknows.settings import GraphKnowsSettings

            settings = GraphKnowsSettings()
            spacy_model = settings.spacy_model
        self._schema_miner = schema_miner or _SpacySchemaMiner(model_name=spacy_model)
        # Semantic gate that drops relex false positives before the graph merge.
        # Built from settings by default (enabled, threshold 0.5); tests inject a
        # null gate. See relations/verifier.py.
        if relation_verifier is None:
            from graphknows.ingestion.extraction.relations.verifier import (
                build_relation_verifier,
            )
            from graphknows.settings import GraphKnowsSettings

            relation_verifier = build_relation_verifier(settings or GraphKnowsSettings())
        self._relation_verifier = relation_verifier
        # Open-vocabulary relations: the per-chunk mined SVO predicate labels ARE
        # the schema. An explicit *relation_schema* (escape hatch for
        # taxonomy-guided extraction) is honored when passed.
        self._relation_schema: dict[str, str] = dict(relation_schema) if relation_schema else {}

    @property
    def nlp(self) -> Any:
        """The loaded spaCy pipeline (shared with the schema miner; loads lazily).

        Reused by the frame-SRL ingest pass so a second ``en_core_web_lg`` is
        never loaded.
        """
        return self._schema_miner.nlp

    @property
    def gliner(self) -> Any:
        """The loaded GLiNER2 model (loads lazily on first access).

        Reused by the frame-SRL ingest pass so a second GLiNER2 is never loaded.
        """
        self._ensure_model_loaded()
        return self._gliner

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

    def _relex(
        self,
        text: str,
        dynamic_schema: _DynamicSchema,
        extra_entity_labels: Sequence[str] | None = None,
        extra_relation_labels: dict[str, str] | None = None,
    ) -> tuple[list, list]:
        """One relex forward pass over a WHOLE chunk. Returns (entities, relations).

        No sub-windowing: relex has no fixed max length, and the old 128-token
        windows existed because GLiNER2 attended poorly to long spans — they cost
        ~4 model calls per chunk. One call per chunk both simplifies the merge away
        and offsets relex being the larger model.

        ``extra_entity_labels``/``extra_relation_labels`` are ontology terms
        selected for THIS chunk. They are unioned in, never substituted: the
        mined vocabulary keeps its precedence, so ontology injection can only add
        candidates, never suppress an entity the text actually states.
        """
        assert self._gliner is not None
        relation_spec = dict(self._relation_schema or dynamic_schema.relation_spec)
        if extra_relation_labels:
            for label, description in extra_relation_labels.items():
                relation_spec.setdefault(label, description)
        # Observed labels first (they carry the chunk's own evidence), then the
        # ontology's suggestions, then the base inventory — relex cannot emit a
        # type it was not given.
        labels = list(
            dict.fromkeys(
                [
                    *dynamic_schema.entity_spec.keys(),
                    *(extra_entity_labels or ()),
                    *_BASE_ENTITY_LABELS,
                ]
            )
        )
        # Serialised inside the model wrapper (see gliner_model._SerialInference):
        # this model is shared across concurrently-ingesting conversations AND
        # with the frame-SRL pass, and its inference() is not reentrant.
        out = self._gliner.inference(
            texts=[text],
            labels=labels,
            relations=list(relation_spec.keys()),
            threshold=_BASE_THRESHOLD,
            adjacency_threshold=_ADJACENCY_THRESHOLD,
            relation_threshold=_RELATION_THRESHOLD,
            return_relations=bool(relation_spec),
            flat_ner=False,
        )
        # inference() returns (entities, relations) ONLY when return_relations is
        # true; with it false it returns the entities list alone. A chunk whose
        # SVO pass mined no predicates takes that branch, so unpacking blind
        # raises ValueError and loses the chunk's entities too.
        entities, relations = out if isinstance(out, tuple) else (out, [])
        return (entities[0] if entities else []), (relations[0] if relations else [])

    def extract(
        self,
        text: str,
        extra_entity_labels: Sequence[str] | None = None,
        extra_relation_labels: dict[str, str] | None = None,
    ) -> ExtractionResult:
        """Extract entities and relations using a spaCy-mined relex schema.

        Args:
            text: The chunk to extract from.
            extra_entity_labels: Ontology class labels to offer for this chunk.
            extra_relation_labels: Ontology property label -> definition, offered
                as additional relation predicates for this chunk.
        """
        self._ensure_model_loaded()
        assert self._gliner is not None

        dynamic_schema = self._schema_miner.build(text)
        if not dynamic_schema.entity_spec:
            return ExtractionResult()
        ents, rels = self._relex(text, dynamic_schema, extra_entity_labels, extra_relation_labels)
        return self._map_relex(text, ents, rels, dynamic_schema)

    def extract_batch(
        self,
        texts: list[str],
        extra_entity_labels: Sequence[Sequence[str]] | None = None,
        extra_relation_labels: Sequence[dict[str, str]] | None = None,
    ) -> list[ExtractionResult]:
        """Extract for many chunks. Same per-text result as :meth:`extract`.

        relex takes ONE label set per call, but the entity/relation vocabularies
        are mined PER CHUNK — so chunks cannot share a forward pass without
        pooling their schemas, which would let a chunk match another chunk's
        predicates. One call per chunk keeps the per-chunk schema honest; the
        old 4-windows-per-chunk fan-out is what actually cost the calls, and that
        is gone. Chunks whose spaCy schema mines no entities skip the model
        entirely (empty result), same as :meth:`extract`.

        ``extra_entity_labels``/``extra_relation_labels``, when given, are
        PER-TEXT and must be the same length as ``texts`` — ontology terms are
        selected per chunk, so they cannot be shared across the batch.
        """
        if not texts:
            return []
        self._ensure_model_loaded()
        assert self._gliner is not None

        # Mine all chunk schemas in one batched spaCy pass.
        dynamic_schemas = self._schema_miner.build_batch(texts)
        results: list[ExtractionResult | None] = [None] * len(texts)
        for i, (text, dynamic_schema) in enumerate(zip(texts, dynamic_schemas, strict=True)):
            if not dynamic_schema.entity_spec:
                results[i] = ExtractionResult()
                continue
            try:
                ents, rels = self._relex(
                    text,
                    dynamic_schema,
                    extra_entity_labels[i] if extra_entity_labels else None,
                    extra_relation_labels[i] if extra_relation_labels else None,
                )
            except Exception as exc:
                log.warning("relex failed on chunk %d (%s); skipping", i, exc)
                results[i] = ExtractionResult()
                continue
            results[i] = self._map_relex(text, ents, rels, dynamic_schema)

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

    def _map_relex(
        self,
        text: str,
        raw_entities: list[dict[str, Any]],
        raw_relations: list[dict[str, Any]],
        dynamic_schema: _DynamicSchema,
    ) -> ExtractionResult:
        """Map one relex (entities, relations) pair → ExtractionResult.

        relex extracts entities and relations JOINTLY, returning typed head/tail
        spans — so unlike the GLiNER2 era its relations are facts in their own
        right, not corroboration of an SVO triple. Two relation sources survive:

          * ``relex`` — the model emitted (head, predicate, tail) under a
            SVO-mined predicate label, with both arguments span-grounded and
            typed. Kept at the model's own score.
          * ``svo`` — a mined triple relex did NOT emit, kept at the same fixed
            0.6 structural prior as before and under the same gates: BOTH
            arguments named, or a copular/appositive/possessive pattern with a
            named head. This retains the durable identity/possession facts
            ("I'm a nurse", "Caroline's dog") that a verb-relation model has no
            predicate label for.
        """
        # getattr: tests stub the schema with SimpleNamespace and predate the field.
        speakers: frozenset[str] = getattr(dynamic_schema, "speakers", frozenset())
        entities = _dedupe_entities(raw_entities, dynamic_schema.label_to_type, speakers)

        # relex first, then SVO: a mined triple relex already emitted is dropped
        # here, so the model's own score wins over the structural prior.
        relations: list[dict[str, Any]] = []
        seen: set[tuple[str, str, str]] = set()
        candidates = [
            *_relex_relations(raw_relations, entities, dynamic_schema.label_to_type, speakers),
            *_svo_structural_relations(dynamic_schema.svo_triples, entities, speakers),
        ]
        for rel in candidates:
            key = (rel["head"].casefold(), rel["relation"], rel["tail"].casefold())
            if key in seen:
                continue
            seen.add(key)
            relations.append(rel)

        # Semantic gate: drop (head, relation, tail) triples the verifier judges
        # false positives before they reach the graph merge. Null-object when
        # disabled, so this is unconditional. getattr: tests predating the field.
        verifier = getattr(self, "_relation_verifier", None)
        if verifier is not None:
            relations = verifier.filter(text, relations)

        return ExtractionResult(entities=entities, relations=relations)

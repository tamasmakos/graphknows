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
# conversational types spaCy has no tag for at all (occupation, animal, food).
# Not a domain taxonomy: no entity is required to exist, and the miner still
# decides what it saw.
#
# "activity" and "emotion" were here and are REMOVED. A zero-shot extractor asked
# for an "activity" returns every verb, and for an "emotion" every adjective —
# they name STATES and PROCESSES, which are not things and cannot be graph nodes.
# Measured on LoCoMo conv-30 they produced 906 of 2105 ENTITY nodes (43%), and
# the most-mentioned "entities" in the whole graph were `keep` x56, `really` x30,
# `going` x32, `make` x29. Those reached the generator's fact-sheet as frame role
# fillers. No downstream name filter can repair a label that instructs the model
# to tag the wrong part of speech; the fix has to be not asking for it.
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
    # Casefolded surfaces used ONLY as a verb in this chunk (see
    # _mine_verb_surfaces). Consumed by _dedupe_entities to drop predicate spans
    # the model returned as entities.
    verb_surfaces: frozenset[str] = frozenset()
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


# WordNet noun supersenses -> the NER taxonomy the rest of the graph speaks.
# _arg_type falls back to a supersense when NER does not label a relation
# argument, and ENTITY.type is first-write-wins, so an entity first seen as a
# relation endpoint kept the supersense forever: measured on conv-30, every
# entity whose type sat outside the taxonomy was a relation endpoint
# ('clothes store' noun.artifact, 'stress buster' noun.person, 'target audience'
# noun.group). Two vocabularies in one column make ENTITY.type unqueryable, and
# the supersense form also reaches the ontology class-matcher as an unknown
# label, where it can only ever score by embedding.
#
# A supersense with no defensible taxonomy equivalent maps to "" — untyped, not
# wrong — which coalesce lets a later NER write fill in.
_SUPERSENSE_TO_TYPE = {
    "noun.person": "PERSON",
    "noun.group": "ORGANIZATION",
    "noun.location": "LOCATION",
    "noun.artifact": "ARTIFACT",
    "noun.event": "EVENT",
    "noun.act": "ACTIVITY",
    "noun.possession": "POSSESSION",
    "noun.time": "TIME",
    "noun.animal": "ANIMAL",
    "noun.feeling": "EMOTION",
    "noun.food": "PRODUCT",
    "noun.body": "ARTIFACT",
    "noun.communication": "ARTIFACT",
}


def _taxonomy_type(supersense: str) -> str:
    """A WordNet supersense rendered in the graph's own type vocabulary."""
    return _SUPERSENSE_TO_TYPE.get((supersense or "").strip().lower(), "")


def _arg_type(arg: str, entities: list[dict[str, Any]]) -> str:
    """Type for a relation argument: spaCy/GLiNER2 NER label, else WordNet supersense.

    NER only labels NAMED entities, so before this fell back to a bare "ENTITY"
    the great majority of dialogue relation arguments (common nouns — "a nurse",
    "cat", "job") reached the graph untyped, leaving the type space too degenerate
    to condition on. The supersense ("noun.person", "noun.artifact", ...) is
    already computed inside svo's concreteness gates and was being discarded.

    The supersense is translated into the graph's own taxonomy before it is
    returned (see ``_SUPERSENSE_TO_TYPE``); it must never reach ENTITY.type in
    WordNet's vocabulary.

    Returns "" — NOT a placeholder — when the argument is neither NER-typed nor in
    WordNet. ENTITY.type is written first-write-wins (``coalesce``), so emitting a
    placeholder here would let a relation argument squat on the vertex and block
    the real NER label a later chunk assigns to the same entity.
    """
    named = next(
        (e["type"] for e in entities if e["name"].casefold() == arg.casefold()),
        "",
    )
    return named or _taxonomy_type(noun_supersense(arg))


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


_DETERMINERS = ("a ", "an ", "the ")


def _strip_determiner(name: str) -> str:
    """Drop a leading article: "a banker" -> "banker".

    The determiner is not part of the entity, and it forks the merge key —
    "a banker" and "banker" become two ENTITY nodes for one referent. Measured
    on conv-30: 31 names carry one, and they account for a share of the 472
    nested-span pairs (``('a banker', 'as a banker')``).
    """
    low = name.casefold()
    for det in _DETERMINERS:
        if low.startswith(det) and len(name) > len(det):
            return name[len(det) :].strip()
    return name


def _mine_verb_surfaces(doc: Any) -> frozenset[str]:
    """Casefolded surfaces this chunk uses ONLY as a verb.

    The entity model returns predicate spans as entities: measured on conv-30,
    164 of 912 mentioned entities (18.0%) are verb-headed where they occur, and
    the extractor types most of them PERSON (79) or EVENT (52) -- "speaks",
    "practicing", "explore", "Winning".

    Context is what makes this safe, and a name-only test is not a substitute:
    tagging the same 912 names in ISOLATION calls 67 more of them verbs (7.3%)
    that the parser reads as nouns in their own sentence -- "shot" in "take a
    shot", "work" in "after work", "dances" in "I love all dances". A lexicon or
    a bare-string rule deletes exactly those.

    A surface is only reported when it appears as a verb and NEVER as a
    noun/proper noun anywhere in this chunk, so "I work at the store after work"
    keeps "work".
    """
    verbal: set[str] = set()
    nominal: set[str] = set()
    for tok in doc:
        surface = tok.text.casefold()
        if tok.pos_ in {"VERB", "AUX"}:
            verbal.add(surface)
        elif tok.pos_ in {"NOUN", "PROPN"}:
            nominal.add(surface)
    return frozenset(verbal - nominal)


def _overlaps(a: dict[str, Any], b: dict[str, Any]) -> bool:
    """Whether two resolved spans share any character of the chunk.

    A span with no offsets (``start < 0``) overlaps nothing: relex always emits
    them, but callers that stub the model do not, and a missing offset is not
    evidence of overlap.
    """
    if a["start"] < 0 or b["start"] < 0:
        return False
    return a["start"] < b["end"] and b["start"] < a["end"]


def _resolve_overlaps(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One entity per group of overlapping spans; the discards become aliases.

    relex runs with ``flat_ner=False``, so nested spans are its DESIGN, not a
    defect: "Customers love new offers and promotions." comes back with 'new',
    'new offers' and 'offers' all typed PRODUCT — three ENTITY nodes for one
    thing, and (measured on 60 real chunks) 5.95 entities per chunk against 0.48
    relations. In the live conv-30 graph the same mechanism minted the pairs
    'best'/'best moves', 'brand'/'brand identity', 'biz'/'biz plan' and duplicate
    REL edges whose tails differed only by a suffix.

    Retention rule, from the spec: the LONGER span wins, higher confidence breaks
    a tie. Type is deliberately not consulted — two spans covering the same
    characters are one mention of one thing whatever labels the model hung on
    them ("Becoming Nicole" as both work_of_art and person).
    """
    kept: list[dict[str, Any]] = []
    for cand in sorted(candidates, key=lambda c: (c["start"] - c["end"], -c["score"])):
        winner = next((k for k in kept if _overlaps(k, cand)), None)
        if winner is None:
            kept.append(cand)
        else:
            winner.setdefault("aliases", []).append(cand["name"])
    return kept


def _dedupe_entities(
    raw_entities: list[dict[str, Any]],
    label_to_type: dict[str, str],
    speakers: frozenset[str] = frozenset(),
    verb_surfaces: frozenset[str] = frozenset(),
) -> list[dict[str, Any]]:
    """Score-ranked, overlap- and merge-key-deduped dicts from one relex span list.

    Two passes. First :func:`_resolve_overlaps` collapses spans that cover the
    same characters. Then dedupe by the graph's OWN merge key: ENTITY.name_norm
    is UNIQUE-indexed and written with a batched UNWIND+MERGE, so the same name
    twice in one chunk's list is not a harmless repeat — ArcadeDB raises
    DuplicatedKeyException and the whole DOCUMENT's ingest dies, silently
    removing every chunk in it from the graph. Two occurrences of one name in a
    chunk do NOT overlap, so the key pass is still what catches them. Highest
    score wins the type.

    Surface forms lost to either pass are carried on ``aliases`` (absent when
    empty, so an entity with no discarded variant keeps its three-key shape).
    """
    candidates: list[dict[str, Any]] = []
    for span in raw_entities or []:
        conf = float(span.get("score", 0.0) or 0.0)
        if conf < _BASE_THRESHOLD:
            continue
        raw = str(span.get("text") or "")
        name = _strip_determiner(_reduce_speaker_span(_clean_surface(raw), speakers))
        label = str(span.get("label") or "").strip()
        if not name or not label or name.casefold() in _PRONOUNS:
            continue
        # A span this chunk only ever uses as a verb is a predicate, not a thing.
        if name.casefold() in verb_surfaces:
            continue
        # `or -1` would be wrong here: offset 0 is the first span of every chunk.
        start = int(span["start"]) if span.get("start") is not None else -1
        end = int(span["end"]) if span.get("end") is not None else -1
        # Cleaning TRIMS the surface ("Caroline: Thanks, Melanie" -> "Caroline",
        # "a banker" -> "banker"), so re-anchor the span on the substring that
        # survived. A span still covering discarded text would swallow the real
        # entities inside it — "Melanie" would become an alias of "Caroline".
        if start >= 0 and (offset := raw.find(name)) >= 0:
            start += offset
            end = start + len(name)
        candidates.append(
            # relex echoes the schema key it was given (lowercased spaCy label);
            # map back to the canonical NER type the graph stores.
            {
                "name": name,
                "type": _canonical_type(label, label_to_type),
                "score": conf,
                "start": start,
                "end": end,
            }
        )

    best: dict[str, dict[str, Any]] = {}
    for cand in _resolve_overlaps(candidates):
        key = " ".join(cand["name"].split()).casefold()  # mirrors graph_store._name_norm
        prev = best.get(key)
        if prev is None:
            best[key] = cand
            continue
        loser, winner = (prev, cand) if cand["score"] > prev["score"] else (cand, prev)
        winner.setdefault("aliases", []).extend([loser["name"], *loser.get("aliases", ())])
        best[key] = winner

    out: list[dict[str, Any]] = []
    for entity in best.values():
        aliases = [
            alias
            for alias in dict.fromkeys(entity.pop("aliases", ()))
            if alias.casefold() != entity["name"].casefold()
        ]
        entity.pop("start")
        entity.pop("end")
        if aliases:
            entity["aliases"] = aliases
        out.append(entity)
    return out


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


def _mine_svo_schema(
    doc: Any, speaker: str = "", known_speakers: Sequence[str] = ()
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """(svo_triples, relation_spec) mined from a doc's clause structure.

    ``speaker`` is who uttered this chunk, when the caller knows it. Almost all
    conversational predication is first-person ("I work as a nurse"), and such a
    clause only becomes a relation once its subject resolves to a name — so
    without a speaker the dominant relation shape is silently dropped.

    It falls back to scraping ``"ts | Speaker: text"`` turn-lines out of the text
    for callers that still flatten the speaker into the content. That fallback is
    the ONLY way this used to work, which meant a caller passing clean text plus
    structured metadata — the arrangement that keeps speakers from being minted
    as entities — lost its relations as the price. Measured on LoCoMo conv-30:
    57 relations with the prefix, 15 without.
    """
    line_speakers = _line_speakers(doc.text)
    scraped = sorted({s for _, s in line_speakers if s})
    speakers = sorted(
        {*scraped, *(s for s in known_speakers if s), *([speaker] if speaker else [])}
    )
    svo_triples: list[dict[str, Any]] = []
    label_verb: dict[str, tuple[str, str]] = {}
    label_counts: Counter[str] = Counter()
    for sent in doc.sents:
        # A scraped turn-line wins for its own sentence — a multi-turn chunk has
        # a different speaker per line, which the caller's single value cannot
        # express. The caller's speaker is the fallback, not the override.
        sent_speaker = _speaker_at(line_speakers, sent.start_char) or speaker
        for triple in extract_relations(sent, sent_speaker, speakers):
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

    def build(self, text: str, speaker: str = "") -> _DynamicSchema:
        self._ensure_loaded()
        assert self._nlp is not None
        return self._build_from_doc(self._nlp(text), speaker)

    def build_batch(self, texts: list[str], speakers: Sequence[str] = ()) -> list[_DynamicSchema]:
        """Mine schemas for many chunks via one batched ``nlp.pipe`` pass.

        ``speakers`` is positional per text; a short or empty sequence simply
        leaves the remaining chunks to the turn-line fallback.
        """
        self._ensure_loaded()
        assert self._nlp is not None
        return [
            self._build_from_doc(doc, speakers[i] if i < len(speakers) else "")
            for i, doc in enumerate(self._nlp.pipe(texts))
        ]

    def _build_from_doc(self, doc: Any, speaker: str = "") -> _DynamicSchema:
        entity_spec, label_to_type = _mine_entity_schema(doc)
        svo_triples, relation_spec = _mine_svo_schema(doc, speaker)
        scraped = {s.casefold() for _, s in _line_speakers(doc.text) if s}
        return _DynamicSchema(
            entity_spec=entity_spec,
            relation_spec=relation_spec,
            label_to_type=label_to_type,
            svo_triples=svo_triples,
            speakers=frozenset(scraped | ({speaker.casefold()} if speaker else set())),
            verb_surfaces=_mine_verb_surfaces(doc),
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
        speaker: str = "",
    ) -> ExtractionResult:
        """Extract entities and relations using a spaCy-mined relex schema.

        Args:
            text: The chunk to extract from.
            extra_entity_labels: Ontology class labels to offer for this chunk.
            extra_relation_labels: Ontology property label -> definition, offered
                as additional relation predicates for this chunk.
            speaker: Who uttered this chunk, when the caller knows it. Anchors
                first/second-person clause subjects, which are the dominant
                relation shape in dialogue. Omit it and the speaker is scraped
                from ``"ts | Speaker: text"`` turn-lines in the text instead.
        """
        self._ensure_model_loaded()
        assert self._gliner is not None

        dynamic_schema = self._schema_miner.build(text, speaker)
        if not text.strip():
            return ExtractionResult()
        # No early return on an empty entity_spec. The labels handed to relex are
        # the chunk's OBSERVED spaCy labels unioned with _BASE_ENTITY_LABELS, so
        # an empty observation is not an empty schema — it is exactly the case
        # the base inventory exists for. Skipping here was a cost optimisation
        # that assumed spaCy sees what matters; on dialogue it does not.
        # Measured on LoCoMo conv-30: 170 of 369 turns (46%) observed no label
        # and so were never extracted from, including plain factual statements.
        ents, rels = self._relex(text, dynamic_schema, extra_entity_labels, extra_relation_labels)
        return self._map_relex(text, ents, rels, dynamic_schema)

    def extract_batch(
        self,
        texts: list[str],
        extra_entity_labels: Sequence[Sequence[str]] | None = None,
        extra_relation_labels: Sequence[dict[str, str]] | None = None,
        speakers: Sequence[str] = (),
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

        ``speakers`` is likewise per-text and optional; see :meth:`extract`.
        """
        if not texts:
            return []
        self._ensure_model_loaded()
        assert self._gliner is not None

        # Mine all chunk schemas in one batched spaCy pass.
        dynamic_schemas = self._schema_miner.build_batch(texts, speakers)
        results: list[ExtractionResult | None] = [None] * len(texts)
        for i, (text, dynamic_schema) in enumerate(zip(texts, dynamic_schemas, strict=True)):
            # Same as extract(): an empty spaCy observation is not an empty
            # schema, because _BASE_ENTITY_LABELS is unioned in downstream.
            if not text.strip():
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
        entities = _dedupe_entities(
            raw_entities,
            dynamic_schema.label_to_type,
            speakers,
            getattr(dynamic_schema, "verb_surfaces", frozenset()),
        )

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

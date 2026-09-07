"""Ontology-labelled relex extractor for the memory ingestion pipeline.

Two vocabularies condition the extraction model, and they come from different
places.

ENTITY labels are mined per chunk by spaCy NER, unioned with
:data:`_BASE_ENTITY_LABELS` — relex is zero-shot and can only emit a type it was
handed, so an empty spaCy observation is not an empty schema.

RELATION labels are the ontology's. ``extra_relation_labels`` is
``{property label -> definition}`` for the ontology properties nearest THIS
chunk's embedding, selected by the caller (``stm/ingest.py::_label_hints``), and
it IS the relation spec — relex is asked for CCO's ``is supervised by``/``has
brother``,
nothing else.

That used to be an SVO miner's open-vocabulary predicates, unioned with the
ontology hints. The miner is gone. Measured on a live conv-30 graph it produced
28 of the 43 relation edges and 25 of those 28 were ``<speaker> HAS <common
noun>`` (``Jon HAS back``, ``Jon HAS corner``) — a four-label vocabulary
``{HAS: 25, BE_TO: 1, IS_A: 1, LOSE_AT: 1}`` against relex's 9 distinct
predicates over 15 edges. So there is now exactly ONE relation source:

* ``relex`` — the model emitted (head, ontology property, tail) with both
  arguments span-grounded and typed, kept at its own score.

A chunk for which no ontology property was selected has no relation label space;
relex is asked for entities only and the omission is logged (see :meth:`_relex`),
because an unlogged empty result is indistinguishable from a broken extractor.

One forward pass per WHOLE chunk: relex has no fixed max length, so the old
128-token sub-windowing (~4 calls/chunk, needed because GLiNER2 attended poorly
to long spans) is gone.
"""

from __future__ import annotations

import contextlib
import logging
import re
import sys
import time
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from graphknows.ingestion.extraction.entities.gliner_model import load_gliner2_model
from graphknows.ingestion.extraction.entities.hygiene import is_graph_entity_name
from graphknows.ingestion.extraction.relations._lingfeatures import (
    noun_supersense,
    parse_speaker,
    resolve_deixis,
)
from graphknows.nlp import load_spacy_model
from graphknows.settings import get_settings

log = logging.getLogger(__name__)

_BASE_THRESHOLD: float = get_settings().relex_threshold
# relex classifies a candidate entity PAIR as bearing a relation, and its card
# recommends a markedly higher floor here than for entity spans (0.7-0.9 vs
# 0.3-0.5) because a low relation threshold pairs up every co-occurring entity.
#
# That figure was 0.7 here and it discarded EVERY correct extraction. The card is
# calibrated for the model's own in-distribution label vocabulary; this pipeline
# injects ontology property names instead, which score far lower. Measured:
# "Caroline works for Mercy Hospital" under the injected label ``worksFor`` — a
# textbook sentence, a correct relation — scores 0.437. At 0.3 it survives; at
# 0.1 sub-span noise appears; at 0.0 the output explodes. Precision is defended
# by the endpoint guards in _relation_reject, not by an unreachable floor.
_RELATION_THRESHOLD: float = get_settings().relex_relation_threshold
# Floor on relex's entity-PAIR candidate score. Without it every co-occurring
# pair of entities is a relation candidate and gets scored against every supplied
# predicate, which manufactures confident nonsense between unrelated spans.
_ADJACENCY_THRESHOLD: float = get_settings().relex_adjacency

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
# collapse onto. A relex relation anchored on a bare pronoun carries no
# recoverable referent, so it is dropped.
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


@dataclass
class ExtractionResult:
    """Structured result from GLiNER2 extraction."""

    entities: list[dict[str, Any]] = field(default_factory=list)
    relations: list[dict[str, Any]] = field(default_factory=list)
    # {frame, trigger, trigger_offset, roles: {role: [filler]}} — filled only by
    # the LLM decoder; empty for the local one, so llm_free behaviour is unchanged.
    frames: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class _EntityHint:
    text: str
    label: str
    root: Any
    sent: Any


@dataclass(frozen=True)
class _DynamicSchema:
    """Per-chunk ENTITY vocabulary. Relation labels are NOT mined — see module doc."""

    entity_spec: dict[str, dict[str, Any]]
    label_to_type: dict[str, str]
    # Casefolded surfaces used ONLY as a verb in this chunk (see
    # _mine_verb_surfaces). Consumed by _dedupe_entities to drop predicate spans
    # the model returned as entities.
    verb_surfaces: frozenset[str] = frozenset()
    # Casefolded speaker names of the chunk's turn-lines. Model spans sometimes
    # run ACROSS the "Speaker:" colon ("Caroline: Thanks, Melanie" as one
    # PERSON), fragmenting the real speaker node — the mint points reduce such
    # surfaces to the speaker. Empty for non-dialogue text, making that a no-op.
    speakers: frozenset[str] = frozenset()
    # The chunk's text with 1st/2nd-person pronouns rewritten to the speaker's
    # name, for the RELATION pass only (see _relex). "" when no speaker is known
    # or the text carries no deixis, which is the signal to use the text itself.
    deixis_surface: str = ""


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
    what fills that gap.

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
    """Relex label -> the type of a RELATION ENDPOINT.

    An OBSERVED label maps back to its spaCy tag (person -> PERSON); any other
    label — base inventory, or an ontology class the hint pass offered — stands
    on its own, upper-cased. Returning "" here would silently demote relex's own
    typing to the WordNet-supersense fallback, and the endpoint type is a gate
    input (``ingest._domain_admits``/``_range_admits``) that wants the richest
    label there is.

    ENTITY.type is a different question with a different answer — see
    :func:`_entity_type`.
    """
    raw_label = raw_label.strip()
    if not raw_label:
        return ""
    return label_to_type.get(_schema_label(raw_label), raw_label.upper())


_TAXONOMY_LABELS: frozenset[str] = frozenset(_BASE_ENTITY_LABELS)


def _entity_type(raw_label: str, label_to_type: dict[str, str]) -> str:
    """ENTITY.type for a span — the graph's OWN taxonomy, or "" (untyped).

    relex is handed three label sources (the chunk's spaCy observations, the
    ontology class hints, the base inventory) and echoes back whichever it
    picked. Only the first and third name types the rest of the graph speaks;
    the ontology's class names are steering for the EXTRACTOR, not a vocabulary
    for this column, and storing them was minting a type space nothing could
    query. Censused on the full conv-30 ingest (369 turns): 1256 entities over
    41 distinct types against a 763/10 baseline, led by EMOTION 275, ACTIVITY
    202, RELATIONSHIP 113, POSSESSION 106, GOAL 96 — all class labels of the
    ontology that run configured, all MENTIONS-linked (so minted here, not as
    relation endpoints). The bundled default leaks the same way at a smaller
    scale: BANKACCOUNT, HOWTO, WHOLESALESTORE on dance-studio dialogue.

    The span itself is kept: the ontology hint is what made relex find it, and
    ENTITY.type is written with ``coalesce``, so "" leaves room for a later
    NER-labelled mention to type the same node. A wrong type does not.

    Relation ENDPOINT types deliberately do NOT come through here (see
    ``_canonical_type``): there the type is a gate input, feeding
    ``ingest._range_admits``, which needs the richest label available.

    This gate does not reach the WordNet supersense fallback either — that one
    lives in ``_arg_type``, on the relation-endpoint path only. Measured over
    both conv-30 graphs, endpoint-only ENTITY nodes number ZERO: every endpoint
    is also an extracted span, so the supersense loses the ``coalesce`` race.
    """
    key = _schema_label(raw_label)
    if key in label_to_type:  # a label spaCy actually observed in this chunk
        return label_to_type[key]
    return key.upper() if key in _TAXONOMY_LABELS else ""


def _speaker_token(token: str, speakers: frozenset[str]) -> str:
    """The speaker this token names, stripped of punctuation/genitive — else ""."""
    bare = token.strip(".,;:!?()[]\"'").removesuffix("'s").removesuffix("'")
    return bare if bare.casefold() in speakers else ""


def _speaker_at_both_ends(tokens: list[str], speakers: frozenset[str]) -> str:
    """The speaker a span opens AND closes with, else "".

    "Caroline, so Caroline" — what the deixis rewrite leaves when it resolves a
    pronoun on either side of a discourse marker. The all-tokens rule in
    :func:`_reduce_speaker_span` misses it because "so" is not a speaker.
    Measured on conv-26 this minted a PERSON node carrying 63 mentions, the
    graph's second-largest hub, for a person who already had one.

    Both ends must name the SAME speaker, so "Caroline, and later Melanie" —
    two people, a real distinction — is untouched.
    """
    if len(tokens) <= 2:
        return ""
    first = _speaker_token(tokens[0], speakers)
    last = _speaker_token(tokens[-1], speakers)
    return first if first and first.casefold() == last.casefold() else ""


def _reduce_speaker_span(name: str, speakers: frozenset[str], ent_type: str = "") -> str:
    """``"Caroline: Thanks, Melanie"`` -> ``"Caroline"`` when Caroline spoke.

    Chunk text embeds literal ``"[ts |] Speaker: utterance"`` turn-lines, and
    model spans sometimes run across that colon, minting the span verbatim as an
    entity — measured on conv-26, 27 such names carried 42.5% of all REL edges,
    fragmenting the real person node. The semantic referent of such a span is
    the speaker (its facts are things the SPEAKER did), and any real entity in
    the utterance remainder is emitted by the model as its own span anyway.

    A second shape, same referent and same fix: a span that OPENS with a speaker
    name and continues into a clause. On the raw text that shape is a pronoun
    clause and ``is_graph_entity_name`` drops it (``_is_clause_fragment`` keys on
    a pronoun first token); the deixis-resolved relation surface rewrites the
    pronoun to a NAME and so walks straight past that rule. Measured on the
    40-turn conv-30 probe: "Gina is over the moon because" and "Jon is staying
    positive. Jon" both reached the graph as relation endpoints. A remainder that
    is entirely capitalised is a proper-name sequence ("Gina Torres") and is left
    alone; anything lowercase in it means the span stopped being a name.

    Three further shapes, all minted by the per-turn deixis rewrite, which
    replaces every "I"/"my" with the speaker's name and so puts that name two or
    three times into one line. Measured on conv-30/skos_v1, where the pieces
    below carried 215 of the graph's 283 REL edges while the plain ``Jon`` and
    ``Gina`` nodes carried NONE:

    * every token is a speaker ("Jon Jon", "Gina, Gina", "Jon, Gina") — the span
      is nothing but repeated deixis output, so it is that speaker;
    * the span ENDS in ", <Speaker>" after a clause ("Gina's hard work and
      effort, Gina") — the mirror of the speaker-first rule below, and the shape
      a trailing resolved pronoun leaves;
    * a PERSON-typed genitive of a speaker ("Jon's students", 123 edges — 43% of
      the graph — typed PERSON and holding Jon's own facts: ``owns Jon's
      studio``, ``has occupation``, ``lives in place``). The TYPE is what makes
      this safe: "Jon's studio" is typed ENTITY and keeps its owner, exactly as
      the whole-span genitive rule below intends. Without a type the two shapes
      are indistinguishable, so an untyped span is left alone.

    Keys strictly on *known speakers of this chunk*: a colon alone never fires,
    so titles like "Charlotte's Web: A Review" pass through, and non-dialogue
    text (empty ``speakers``) makes this the identity function.
    """
    if not speakers:
        return name
    if ":" in name:
        head = name.split(":", 1)[0]
        if " ".join(head.split()).casefold() in speakers:
            return head.strip()
        return name
    tokens = name.split()
    # Nothing but repeated deixis output — collapse onto the first speaker named.
    if tokens and all(_speaker_token(t, speakers) for t in tokens):
        return _speaker_token(tokens[0], speakers)
    if bracketed := _speaker_at_both_ends(tokens, speakers):
        return bracketed
    # "<clause>, <Speaker>" — a trailing resolved pronoun, referent is the speaker.
    if (
        len(tokens) > 1
        and tokens[-2].endswith(",")
        and (sp := _speaker_token(tokens[-1], speakers))
    ):
        return sp
    # "<Speaker>'s <noun>" typed PERSON — the person named is the speaker.
    if (
        len(tokens) > 1
        and ent_type.upper() == "PERSON"
        and tokens[0].endswith(("'s", "'"))
        and (sp := _speaker_token(tokens[0], speakers))
    ):
        return sp
    first, *rest = tokens or [""]
    # A bare genitive of a speaker is that speaker. Measured live it forked the
    # node ("Jon's owns biz" beside "Jon owns biz") and hid a self-relation
    # outright — "Jon's -[spouse]-> Jon" walked past the head != tail check.
    # Whole-span only: "Jon's studio" names something else and keeps its owner.
    if not rest and (bare := first.removesuffix("'s").removesuffix("'")).casefold() in speakers:
        return bare
    if rest and first.casefold() in speakers and any(not w[:1].isupper() for w in rest):
        return first
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
        label = str(span.get("label") or "").strip()
        # Typed BEFORE the span is reduced: the PERSON rule in
        # _reduce_speaker_span is what separates "Jon's students" (the speaker)
        # from "Jon's studio" (a place), and the type is the only signal that
        # tells them apart. _entity_type reads the label only, never the name.
        ent_type = _entity_type(label, label_to_type)
        name = _strip_determiner(_reduce_speaker_span(_clean_surface(raw), speakers, ent_type))
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
            # relex echoes the schema key it was given (lowercased spaCy label,
            # base-inventory label, or ontology class name); map back to the
            # canonical NER type the graph stores, or leave it untyped.
            {
                "name": name,
                "type": ent_type,
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


def _relation_reject(head: str, label: str, tail: str, head_type: str, tail_type: str) -> str:
    """Why this triple must not reach the graph, or "" when it may.

    The ontology label set is CLOSED, so relex is FORCED to pick one of the
    properties it was handed — offered five on two friends chatting it emitted
    both ``Gina spouse Jon`` and ``Gina children Jon``, neither true. Nothing here
    can judge a predicate, but two structural checks cost nothing and are exactly
    what the extra recall from deixis resolution makes necessary:

    * a relation whose ends are the same entity says nothing (and the resolved
      surface, where both ends can be the speaker, makes "Gina knows Gina" easy);
    * an endpoint that is not a graph entity name at all — a date, a clause
      fragment, a backchannel. ``is_graph_entity_name`` is the ONE junk predicate
      every minting path shares, so it applies to BOTH ends here too.
    """
    if not head or not label or not tail:
        return "empty argument"
    if head.casefold() == tail.casefold():
        return "self-relation"
    if head.casefold() in _PRONOUNS or tail.casefold() in _PRONOUNS:
        return "pronoun endpoint"
    if not is_graph_entity_name(head, head_type):
        return "head is not an entity name"
    if not is_graph_entity_name(tail, tail_type):
        return "tail is not an entity name"
    return ""


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
) -> tuple[dict[str, Any] | None, str]:
    """(relation dict, "") or (None, reason) when the argument pair is unusable."""
    # Same cleaning as the entity pass: these strings become ENTITY merge
    # keys via write_relation, so a quoted surface here corrupts the graph
    # exactly as it does there — and a speaker-glued surface here is what
    # attached 42.5% of REL edges to malformed hub names.
    # _strip_determiner for the same reason the entity path applies it, plus one:
    # "the fair" and "fair" are one entity, and the article walked the pair past
    # the same-entity check as `the fair -[attendee]-> fair` (measured, live).
    # The caller's type (when it has one) gates the PERSON-genitive rule; an
    # endpoint typed only by _arg_type is typed FROM the entity list, which the
    # entity pass already reduced, so the name arriving here is already folded.
    head = _strip_determiner(_reduce_speaker_span(_clean_surface(head), speakers, head_type))
    tail = _strip_determiner(_reduce_speaker_span(_clean_surface(tail), speakers, tail_type))
    head_type = head_type or _arg_type(head, entities)
    tail_type = tail_type or _arg_type(tail, entities)
    reason = _relation_reject(head, label, tail, head_type, tail_type)
    if reason:
        return None, reason
    return {
        "head": head,
        "head_type": head_type,
        "relation": label,
        "tail": tail,
        "tail_type": tail_type,
        "score": score,
        "source": source,
    }, ""


def _relex_relations(
    raw_relations: list[dict[str, Any]],
    entities: list[dict[str, Any]],
    label_to_type: dict[str, str],
    speakers: frozenset[str] = frozenset(),
) -> list[dict[str, Any]]:
    """Relation dicts for the (head, predicate, tail) triples relex emitted.

    Drops are tallied by reason and logged: an unlogged swallow makes "the model
    found nothing" and "the guard is eating everything" indistinguishable from
    outside, which is how this repo has previously shipped inert layers.
    """
    out: list[dict[str, Any]] = []
    drops: Counter[str] = Counter()
    for rel in raw_relations or []:
        score = float(rel.get("score", 0.0) or 0.0)
        if score < _RELATION_THRESHOLD:
            drops["below relation threshold"] += 1
            continue
        head_d, tail_d = rel.get("head") or {}, rel.get("tail") or {}
        # relex types its own arguments — richer than the NER label / WordNet
        # supersense fallback, so prefer it and only fall back when absent.
        built, reason = _build_relation(
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
        if built is None:
            drops[reason] += 1
        else:
            out.append(built)
    if drops:
        log.info(
            "relation guard dropped %d of %d relex candidates: %s",
            sum(drops.values()),
            len(raw_relations or []),
            dict(drops.most_common()),
        )
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


class _SpacySchemaMiner:
    """Mine the per-chunk ENTITY label vocabulary from spaCy NER.

    Relation labels are NOT mined here — they are the ontology properties the
    caller selected for the chunk. See the module docstring.
    """

    def __init__(self, model_name: str = _SPACY_MODEL) -> None:
        self._model_name = model_name

    @property
    def nlp(self) -> Any:
        """The shared spaCy pipeline, memoized by :func:`graphknows.nlp.load_spacy_model`."""
        return load_spacy_model(self._model_name)

    def build(self, text: str, speaker: str = "", doc: Any | None = None) -> _DynamicSchema:
        """Mine the schema for *text*, parsing it unless the caller already has a ``doc``.

        ``GLiNER2EntityExtractor.extract`` hands over the doc its own
        :meth:`~GLiNER2EntityExtractor.parse` memo just produced, so the window
        is parsed once per turn rather than once here and again there.
        """
        return self._build_from_doc(doc if doc is not None else self.nlp(text), speaker)

    def build_batch(self, texts: list[str], speakers: Sequence[str] = ()) -> list[_DynamicSchema]:
        """Mine schemas for many chunks via one batched ``nlp.pipe`` pass.

        ``speakers`` is positional per text; a short or empty sequence simply
        leaves the remaining chunks to the turn-line fallback.
        """
        return [
            self._build_from_doc(doc, speakers[i] if i < len(speakers) else "")
            for i, doc in enumerate(self.nlp.pipe(texts))
        ]

    def _build_from_doc(self, doc: Any, speaker: str = "") -> _DynamicSchema:
        entity_spec, label_to_type = _mine_entity_schema(doc)
        scraped = {s.casefold() for _, s in _line_speakers(doc.text) if s}
        surface = resolve_deixis(doc, speaker)
        return _DynamicSchema(
            entity_spec=entity_spec,
            label_to_type=label_to_type,
            speakers=frozenset(scraped | ({speaker.casefold()} if speaker else set())),
            verb_surfaces=_mine_verb_surfaces(doc),
            deixis_surface="" if surface == doc.text else surface,
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
            from graphknows.settings import get_settings

            settings = get_settings()
            spacy_model = settings.spacy_model
        self._schema_miner = schema_miner or _SpacySchemaMiner(model_name=spacy_model)
        # One-entry memo behind `parse()`; see its docstring.
        self._doc_cache: tuple[str, Any] | None = None
        # Semantic gate that drops relex false positives before the graph merge.
        # Built from settings by default (enabled, threshold 0.5); tests inject a
        # null gate. See relations/verifier.py.
        if relation_verifier is None:
            from graphknows.ingestion.extraction.relations.verifier import (
                build_relation_verifier,
            )
            from graphknows.settings import get_settings

            relation_verifier = build_relation_verifier(settings or get_settings())
        self._relation_verifier = relation_verifier
        # The relation label space is normally the per-chunk ontology properties
        # the caller passes to extract(). An explicit *relation_schema* (escape
        # hatch: one fixed taxonomy for every chunk) overrides them when passed.
        self._relation_schema: dict[str, str] = dict(relation_schema) if relation_schema else {}

    @property
    def nlp(self) -> Any:
        """The loaded spaCy pipeline (shared with the schema miner; loads lazily).

        ``load_spacy_model`` caches per ``(model, options)``, so every site
        asking for this configuration — including the frame-SRL ingest pass —
        shares this one resident pipeline.
        """
        return self._schema_miner.nlp

    def parse(self, text: str) -> Any:
        """The spaCy ``Doc`` for *text*, parsed once and shared across the segment.

        Three call sites want the same chunk parsed: lexical label selection,
        frame-candidate ranking, and the modality pass. Sharing one model
        instance already avoids loading ``en_core_web_lg`` twice, but each call
        still ran the pipeline again — around 20ms per five-turn window, three
        times over.

        A one-entry memo, because those calls are consecutive within a segment.
        Under chunk concurrency (``INGEST_CHUNK_CONCURRENCY``, default 1) two
        segments can interleave and evict each other, which degrades to
        re-parsing — slower, never wrong.
        """
        cached = self._doc_cache
        if cached is not None and cached[0] == text:
            return cached[1]
        doc = self.nlp(text)
        self._doc_cache = (text, doc)
        return doc

    @property
    def gliner(self) -> Any:
        """The loaded GLiNER2 model (loads lazily on first access).

        Reused by the frame-SRL ingest pass so a second GLiNER2 is never loaded.
        """
        self._ensure_model_loaded()
        return self._gliner

    @property
    def relation_verifier(self) -> Any:
        """The relation verifier gate this extractor was built with.

        Exposed so the ingest write funnel scores the non-relex tiers with the
        ALREADY-LOADED gate: a second :class:`RelationVerifierGate` would mean a
        second DeBERTa checkpoint resident in memory.
        """
        return self._relation_verifier

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

        ``extra_entity_labels``/``extra_relation_labels`` are the ontology terms
        nearest THIS chunk's embedding. The two are used differently, on purpose:

        * entity labels are UNIONed with the chunk's own spaCy observations and
          the base inventory — ontology injection may add candidate types, never
          suppress an entity the text actually states;
        * relation labels ARE the relation spec. There is no other source since
          the SVO miner was deleted, so an empty hint set means relex is asked
          for entities only. That is a real outcome (no ontology property scored
          above the candidate threshold for this chunk), not a silent nothing, so
          it is logged: an unlogged empty result is indistinguishable from a
          broken extractor.

        TWO passes when the chunk carries first/second-person deixis: entities off
        the ORIGINAL text, relations off the speaker-resolved surface. See
        :func:`~..relations._lingfeatures.resolve_deixis` — a pronoun is not an
        entity span, so on the raw text the subject of a self-stated fact simply
        is not there. One pass otherwise, which is most non-dialogue text.
        """
        assert self._gliner is not None
        relation_spec = dict(self._relation_schema or extra_relation_labels or {})
        if not relation_spec:
            # ``None`` and ``{}`` are different facts and must stay so. The ingest
            # handler is two-pass — pass 1 passes None because the property
            # vocabulary is selected FROM the entity types pass 1 finds — so
            # warning on None fires once per chunk by design and drowns the case
            # the warning exists for: the selection RAN and matched nothing.
            if extra_relation_labels is None:
                log.debug(
                    "entity-only pass over this chunk (%d chars, starts %r): "
                    "no relation label space was requested",
                    len(text),
                    text[:60],
                )
            else:
                log.warning(
                    "no ontology property matched this chunk (%d chars, starts %r); "
                    "relex is asked for entities only, so it can emit no relations",
                    len(text),
                    text[:60],
                )
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
        # getattr: tests stub the schema with SimpleNamespace and predate the field.
        surface: str = getattr(dynamic_schema, "deixis_surface", "")
        if relation_spec and not surface:
            log.debug(
                "no speaker resolved for this chunk (%d chars, starts %r); relations are "
                "read off the raw text, so a first-person stated fact has no subject span",
                len(text),
                text[:60],
            )
        if not (surface and relation_spec):
            return self._infer(text, labels, relation_spec)
        # Character offsets, alias candidates and overlap resolution all anchor on
        # the original text, and the stored chunk text stays unmodified — so the
        # surface is used for the relation pass ONLY. The two decouple because a
        # relation's head/tail are matched downstream by NAME, not by offset.
        entities, _ = self._infer(text, labels, {})
        _, relations = self._infer(surface, labels, relation_spec)
        return entities, relations

    def _infer(
        self, text: str, labels: list[str], relation_spec: dict[str, str]
    ) -> tuple[list, list]:
        """One relex forward pass. Relations only when ``relation_spec`` is non-empty."""
        assert self._gliner is not None
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
        # true; with it false it returns the entities list alone. A chunk with no
        # ontology property takes that branch, so unpacking blind raises
        # ValueError and loses the chunk's entities too.
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

        # Schema mining used to re-parse text that self.parse() had already (or
        # was about to) cache for lexical labelling / frame-candidate ranking /
        # modality -- see its docstring. Parsing here instead of inside build()
        # populates that same one-entry memo, so the window is parsed once.
        dynamic_schema = self._schema_miner.build(text, speaker, doc=self.parse(text))
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
        # The label space is the ontology's and therefore closed, but which of
        # its properties actually fire is the thing to watch: a batch that logs
        # one label, or none, is the relation layer going inert.
        label_counts = Counter(rel["relation"] for res in final for rel in res.relations)
        log.info(
            "ontology relation labels used: %d distinct across %d chunks; top10=%s",
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
        spans, so its output IS the fact. It is the only relation source: every
        relation here is one the model emitted under an ontology property label,
        kept at the model's own score.
        """
        # getattr: tests stub the schema with SimpleNamespace and predate the field.
        speakers: frozenset[str] = getattr(dynamic_schema, "speakers", frozenset())
        entities = _dedupe_entities(
            raw_entities,
            dynamic_schema.label_to_type,
            speakers,
            getattr(dynamic_schema, "verb_surfaces", frozenset()),
        )

        # Dedupe by (head, relation, tail): relex can emit the same fact twice
        # from two spans of one chunk, and the graph merge writes one edge.
        relations: list[dict[str, Any]] = []
        seen: set[tuple[str, str, str]] = set()
        candidates = _relex_relations(
            raw_relations, entities, dynamic_schema.label_to_type, speakers
        )
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

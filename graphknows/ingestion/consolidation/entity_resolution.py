"""Layered entity resolution for the consolidation (flush) path.

Run before ``consolidate_session`` to declutter the golden ENTITY graph. The
layers are ordered most-confident-first; each re-reads the (shrinking) entity set
so a later layer sees earlier layers' effects:

1. **Junk/hub filter** — DETACH DELETE generic non-referential nouns that SVO
   object extraction emits as junk hub entities ("others", "many", "kind").
2. **Name-standardization merge** — collapse morphological (plural) variants of
   *non-person* entities onto one node (paths->path, dance competitions->dance
   competition). Persons are skipped here so proper names ending in ``s``
   (Chris, James) are never singularized into a false collision.
3. **Person-nickname merge** — collapse a short name into its person superstring
   (Mel -> Melanie), gated on the anchor being a known person (spaCy PERSON type
   or a session speaker) and the short being a single-token alphabetic prefix.

Two further layers — a speaker-prefix merge for ``"X: utterance"`` spans and a
temporal/junk drop over the whole entity set — were deleted: both existed only
to repair graphs written before the mint-point gates landed, and every write
now passes ``extractor._reduce_speaker_span`` and ``is_graph_entity_name``
before an ENTITY node exists at all.

Embedding / graph-similarity thresholds are deliberately NOT used as the primary
mechanism: on conv-26 a true nickname pair (Mel~Melanie: emb 0.528) scores BELOW
two distinct co-occurring people (Melanie~Caroline: emb 0.497 / jaccard 0.367),
so a similarity threshold corrupts the graph. A conservative same-type
co-occurring emb>=0.92 layer is left as a documented, off-by-default hook (see
``resolve_entities``); it is not implemented for v1.
"""

from __future__ import annotations

import logging
from typing import Any

from graphknows.storage.arcadedb.graph_store import _name_norm

log = logging.getLogger(__name__)

# Layer 1: generic non-referential nouns SVO object extraction surfaces as junk
# hub entities. Matched on name_norm (casefolded, whitespace-collapsed).
_JUNK_ENTITIES: frozenset[str] = frozenset(
    {
        "others",
        "many",
        "kind",
        "thing",
        "things",
        "everyone",
        "someone",
        "anyone",
        "everything",
        "nothing",
        "lot",
        "lots",
        "one",
        "ones",
        "some",
        "all",
        "none",
        "way",
        "ways",
        "bit",
        "part",
    }
)

# Leading determiners / possessives stripped when building a standardization key.
_LEADING_DET: frozenset[str] = frozenset(
    {
        "the",
        "a",
        "an",
        "this",
        "that",
        "these",
        "those",
        "my",
        "your",
        "his",
        "her",
        "its",
        "our",
        "their",
    }
)


def _singularize(word: str) -> str:
    """Best-effort English plural -> singular for the standardization key.

    Rule-based (no spaCy/inflect dependency) — handles the regular plural
    families that actually create duplicate common-noun nodes: paths->path,
    competitions->competition, successes->success, stories->story, glasses->
    glass. Gerunds are deliberately NOT stripped (building != build, meeting !=
    meet, wedding != wed) to avoid over-merging distinct concepts.
    """
    w = word.lower()
    if len(w) <= 3 or w.endswith(("ss", "us", "is")):
        return w
    if w.endswith("ies"):
        return w[:-3] + "y"
    if w.endswith(("ses", "xes", "zes", "ches", "shes")):
        return w[:-2]
    if w.endswith("s"):
        return w[:-1]
    return w


def _standard_key(name: str) -> str:
    """Standardized merge key for entity de-duplication.

    Casefold + whitespace-collapse (``_name_norm``), strip a leading
    determiner/possessive, singularize the head (last) token.
    """
    toks = _name_norm(name).split()
    while len(toks) > 1 and toks[0] in _LEADING_DET:
        toks = toks[1:]
    if not toks:
        return _name_norm(name)
    toks[-1] = _singularize(toks[-1])
    return " ".join(toks)


# spaCy NER types that mark a PROPER-NOUN entity. The morphological
# standardization layer must never singularize/merge these — a team "Warriors"
# (ORG) must not collapse onto the common noun "warrior", nor "Philippines" (GPE)
# onto "philippine". Only untyped / common-noun entities are morphologically merged.
_PROPER_TYPES: frozenset[str] = frozenset(
    {
        "PERSON",
        "ORG",
        "GPE",
        "LOC",
        "FAC",
        "NORP",
        "PRODUCT",
        "EVENT",
        "WORK_OF_ART",
        "LAW",
        "LANGUAGE",
    }
)


def _is_person(entity: dict[str, Any], speakers: set[str]) -> bool:
    """Direct person signal: spaCy PERSON type, or a known session speaker."""
    return (entity.get("type") or "").upper() == "PERSON" or entity["name_norm"] in speakers


def _is_proper(entity: dict[str, Any]) -> bool:
    """True when the entity's spaCy NER type marks it a proper noun."""
    return (entity.get("type") or "").upper() in _PROPER_TYPES


def _canonical(members: list[dict[str, Any]]) -> dict[str, Any]:
    """Pick the canonical entity of a merge group: most mentions, then longest surface.

    Correct for the nickname and speaker-prefix layers, where the longer surface
    IS the fuller name ("mel" -> "Melanie"). Wrong for the standardization layer
    — see :func:`_standard_canonical`.
    """
    return max(members, key=lambda m: (m.get("mention_count", 0), len(m["name"])))


def _standard_canonical(key: str, members: list[dict[str, Any]]) -> dict[str, Any]:
    """Canonical entity of a plural-variant group: the SINGULAR surface.

    ``_canonical``'s longest-surface tiebreak elects the plural every time in a
    {style, styles} group, so a layer whose whole job is to "collapse
    morphological (plural) variants" standardised onto the plural. Measured on
    conv-30: 38 of 2491 MENTIONS edges pointed at a name absent from their own
    chunk text — ``downs`` x8, ``dreams`` x7, ``customers``, ``plans``,
    ``projects`` — because the surviving node was named for a form that chunk
    never used. ``downs`` (EMOTION), minted from "don't let it get you down", is
    the shape of it.

    ``key`` is the group's standard key, which is already the singularized form,
    so the member matching it is the singular. Falls back to ``_canonical`` when
    no member does (every variant is plural).
    """
    exact = [m for m in members if str(m["name_norm"]) == key]
    return _canonical(exact) if exact else _canonical(members)


async def _drop_junk(store: Any) -> int:
    """Layer 1 — delete generic non-referential hub entities."""
    entities = await store.list_entities()
    junk = [e["name_norm"] for e in entities if e["name_norm"] in _JUNK_ENTITIES]
    return await store.drop_entities(junk) if junk else 0


async def _merge_standardized(store: Any) -> int:
    """Layer 2 — collapse morphological (plural) variants of non-person entities."""
    entities = await store.list_entities()
    speakers = set(await store.distinct_speakers())
    groups: dict[str, list[dict[str, Any]]] = {}
    for e in entities:
        if _is_person(e, speakers) or _is_proper(e):
            continue  # persons -> nickname layer; other proper nouns never singularized
        groups.setdefault(_standard_key(e["name"]), []).append(e)

    merged = 0
    for key, members in groups.items():
        if len(members) < 2:
            continue
        winner = _standard_canonical(key, members)
        for m in members:
            if m["name_norm"] == winner["name_norm"]:
                continue
            if await store.merge_entities(winner["name_norm"], m["name_norm"]):
                merged += 1
    return merged


def _parser() -> Any:
    """The shared spaCy pipeline, memoized by :func:`graphknows.nlp.load_spacy_model`."""
    from graphknows.nlp import load_spacy_model
    from graphknows.settings import GraphKnowsSettings

    return load_spacy_model(GraphKnowsSettings().spacy_model)


def _head_key(name: str, nlp: Any) -> str:
    """The mention reduced to its NAMING core — head noun plus compounds.

    Determiners, adjectives and trailing prepositional phrases describe a
    mention; the head and its compound modifiers name it. So::

        "the ideal dance studio by the water" -> "dance studio"
        "Jon's new store"                     -> "store"

    ``_standard_key`` above cannot reach these: it strips a leading determiner
    and singularizes, which leaves "ideal dance studio" and "dance studio" as
    two nodes. Measured on a conv-30 run, one dance studio was split across six
    surfaces, and the duplicates produced self-referential edges that look like
    structure and carry nothing.

    Returns "" when reduction is unsafe — a proper-noun HEAD ("Door Dash") names
    a thing and must survive whole, whereas a proper noun in MODIFIER position
    is a possessor whose claim belongs in a relation, not in the node's name.
    """
    doc = nlp(" ".join(name.split()))
    head = next((t for t in doc if t.dep_ == "ROOT"), None)
    if head is None or head.pos_ != "NOUN":
        return ""
    # Compared by INDEX: spaCy rebuilds Token objects per access, so `is`
    # identity never holds and the reduction silently returns nothing.
    parts = [t for t in doc if t.i == head.i or (t.head.i == head.i and t.dep_ == "compound")]
    reduced = " ".join(t.lemma_.lower() for t in sorted(parts, key=lambda t: t.i))
    return reduced.strip()


async def _head_groups(store: Any, nlp: Any) -> dict[tuple[str, str], list[dict[str, Any]]]:
    """Non-person, non-proper entities grouped by (syntactic head, type).

    Type is part of the key: a Facility "store" and an Organization "store" are
    genuinely different nodes even when spelled alike.
    """
    entities = await store.list_entities()
    speakers = set(await store.distinct_speakers())
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for e in entities:
        if _is_person(e, speakers) or _is_proper(e):
            continue
        if key := _head_key(str(e["name"]), nlp):
            groups.setdefault((key, str(e.get("type") or "")), []).append(e)
    return groups


async def _merge_syntactic_heads(store: Any) -> int:
    """Layer 3 — collapse mentions sharing a syntactic head and a type."""
    try:
        nlp = _parser()
    except Exception as exc:  # a missing model must not take the flush down
        log.warning("Syntactic entity merge skipped: spaCy unavailable (%s)", exc)
        return 0

    groups = await _head_groups(store, nlp)
    merged = 0
    for members in groups.values():
        if len(members) < 2:
            continue
        # The shortest surface wins: it carries the fewest of one mention's
        # incidental modifiers, so it reads as the thing rather than an instance
        # of describing it.
        winner = min(members, key=lambda m: (len(str(m["name"])), str(m["name"])))
        for m in members:
            if m["name_norm"] == winner["name_norm"]:
                continue
            if await store.merge_entities(winner["name_norm"], m["name_norm"]):
                merged += 1
    if merged:
        log.info("syntactic head merge collapsed %d duplicate mentions", merged)
    return merged


# Multi-anchor disambiguation: when a short prefix matches several person
# superstrings, merge into the top only if it dominates the runner-up by this
# factor (else the anchors are genuinely different people -> skip). This lets
# "mel" collapse into "Melanie" (63 mentions) despite a stray "Mell" typo (1),
# while keeping "ann" split across "Anna"/"Anne" (close counts).
_ANCHOR_DOMINANCE = 2


async def _merge_nicknames(store: Any) -> int:
    """Layer 4 — merge a short person name into its person superstring.

    A candidate ``s`` merges into anchor ``l`` when: BOTH are person-like (spaCy
    PERSON type or a known speaker — the strict gate that keeps common nouns like
    "mile"/"art"/"path" out), ``s`` is a strict case-insensitive prefix of ``l``
    (len(s) >= 3, single alphabetic token), a single dominant such anchor exists
    (most mentions, and — when several match — dominating the runner-up by
    ``_ANCHOR_DOMINANCE`` so genuinely distinct names like Anna/Anne are left
    split), and both co-occur in the namespace. Merges Mel -> Melanie while
    leaving Caroline (not a prefix) and non-person prefixes untouched.
    """
    entities = await store.list_entities()
    speakers = set(await store.distinct_speakers())
    persons = [e for e in entities if _is_person(e, speakers)]

    merged = 0
    for e in persons:  # the short side must itself be person-like
        s = e["name_norm"]
        if len(s) < 3 or " " in s or not s.isalpha() or s in _JUNK_ENTITIES:
            continue
        anchors = sorted(
            (
                p
                for p in persons
                if p["name_norm"] != s
                and p["name_norm"].startswith(s)
                # A malformed "X: utterance" name must never anchor a nickname
                # merge — "caroline" is a prefix of "caroline: thanks, melanie",
                # and merging the REAL person into the junk name inverts the
                # repair layer 4 performs.
                and ":" not in p["name_norm"]
            ),
            key=lambda p: (p.get("mention_count", 0), len(p["name_norm"])),
            reverse=True,
        )
        if not anchors:
            continue  # not a nickname
        anchor = anchors[0]
        if len(anchors) > 1:
            runner_up = anchors[1].get("mention_count", 0)
            if anchor.get("mention_count", 0) < _ANCHOR_DOMINANCE * max(runner_up, 1):
                continue  # ambiguous between distinct people
        if not _co_occurs(e, anchor, speakers):
            continue
        if await store.merge_entities(anchor["name_norm"], s):
            merged += 1
    return merged


def _co_occurs(a: dict[str, Any], b: dict[str, Any], speakers: set[str]) -> bool:
    """Both endpoints appear in the namespace (mentioned, or a known speaker)."""

    def present(e: dict[str, Any]) -> bool:
        return e.get("mention_count", 0) > 0 or e["name_norm"] in speakers

    return present(a) and present(b)


async def resolve_entities(store: Any, session_id: str = "") -> dict[str, int]:
    """Run the layered resolver over the namespace ENTITY graph.

    Entities are namespace-global (like ``consolidate_session``'s entity
    promotion), so resolution is namespace-wide; ``session_id`` is accepted for
    call-site symmetry with the rest of the flush path.

    Returns ``{"dropped": n, "merged": n, "considered": n}``.

    A conservative 4th layer (emb>=0.92 AND same-type AND co-occurring) is left
    as a deliberate, off-by-default hook: it is not run here because embedding
    similarity does not cleanly separate nickname merges from distinct
    co-occurring people (see module docstring).
    """
    considered = len(await store.list_entities())
    dropped = await _drop_junk(store)
    merged_morph = await _merge_standardized(store)
    merged_head = await _merge_syntactic_heads(store)
    merged_nick = await _merge_nicknames(store)
    merged = merged_morph + merged_head + merged_nick
    log.info(
        "resolve_entities: dropped=%d merged=%d (morphological=%d syntactic=%d nickname=%d)",
        dropped,
        merged,
        merged_morph,
        merged_head,
        merged_nick,
    )
    return {"dropped": dropped, "merged": merged, "considered": considered}

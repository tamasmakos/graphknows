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
3. **Speaker-prefix merge** — collapse ``"X: utterance"`` spans into the person
   ``X`` (repairs graphs ingested before the mint-point fix). Runs before the
   nickname layer, which would otherwise mistake the malformed superstring for
   a nickname anchor and swallow the real person node.
4. **Person-nickname merge** — collapse a short name into its person superstring
   (Mel -> Melanie), gated on the anchor being a known person (spaCy PERSON type
   or a session speaker) and the short being a single-token alphabetic prefix.
5. **Temporal drop** — delete date/time-shaped entities that entered via the
   (formerly unfiltered) relation-endpoint path.

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


async def _merge_speaker_prefixed(store: Any) -> int:
    """Layer 3 — collapse ``"X: utterance"`` entities into the entity ``X``.

    Model spans that ran across a turn-line's ``Speaker:`` colon used to be
    minted verbatim ("Caroline: Thanks, Melanie" as a PERSON distinct from
    "Caroline") — on conv-26, 27 such names carried 42.5% of all REL edges,
    fragmenting the real person node. New ingests reduce these at the mint
    point (see ``extractor._reduce_speaker_span``); this layer repairs graphs
    ingested before that fix, so an existing namespace heals on its next flush
    without a re-ingest.

    Two cases per colon-head group:

    - ``X`` exists as a person anchor (PERSON entity or session speaker):
      every ``"X: ..."`` member merges into it.
    - ``X`` does not exist at all — the frequent case, because the glued span
      swallowed EVERY mention and no clean node was ever minted (on conv-26,
      "Caroline" existed only as "Caroline: Thanks, Melanie", 63 mentions).
      When the group carries person evidence (a PERSON-typed member), the
      dominant member is RENAMED to ``X`` and the rest merge into it.

    A lone non-person colon name ("Charlotte's Web: A Review") has neither an
    anchor nor person evidence and is never touched.
    """
    entities = await store.list_entities()
    speakers = set(await store.distinct_speakers())
    anchor_norms = {e["name_norm"] for e in entities if _is_person(e, speakers)} | speakers

    groups: dict[str, list[dict[str, Any]]] = {}
    for e in entities:
        norm = e["name_norm"]
        if ":" not in norm:
            continue
        head = norm.split(":", 1)[0].strip()
        if head and head != norm:
            groups.setdefault(head, []).append(e)

    merged = 0
    for head, members in groups.items():
        if head not in anchor_norms:
            if not any((m.get("type") or "").upper() == "PERSON" for m in members):
                continue  # no person evidence — not a speaker-glued group
            # Promote the dominant member to be the person node itself.
            winner = max(members, key=lambda m: m.get("mention_count", 0))
            head_cased = str(winner["name"]).split(":", 1)[0].strip() or head
            if not await store.rename_entity(winner["name_norm"], head_cased):
                continue
            members = [m for m in members if m["name_norm"] != winner["name_norm"]]
        for m in members:
            if await store.merge_entities(head, m["name_norm"]):
                merged += 1
    return merged


async def _drop_temporal(store: Any) -> int:
    """Layer 5 — delete entities the graph-entity gate now rejects.

    Covers what an older ingest let in: temporal/numeric shapes, stopwords,
    code/markup, over-long clauses, and 1-2 char fragments.

    Relation endpoints used to bypass ``graph_entities`` entirely, minting
    "8 May, 2023" / "last Saturday" ENTITY nodes that then received a
    community_id and fragmented the topic layer; the same path (and GLiNER's
    over-generation on technical text) minted code fragments and stopwords. New
    writes are gated (see ``is_graph_entity_name``); this drops the survivors
    from older ingests so existing namespaces heal on their next flush.
    """
    from graphknows.ingestion.stm._entity_writer import is_graph_entity_name

    entities = await store.list_entities()
    junk = [
        e["name_norm"]
        for e in entities
        if not is_graph_entity_name(e.get("name") or e["name_norm"], e.get("type") or "")
    ]
    return await store.drop_entities(junk) if junk else 0


async def resolve_entities(store: Any, session_id: str = "") -> dict[str, int]:
    """Run the layered resolver over the namespace ENTITY graph.

    Entities are namespace-global (like ``consolidate_session``'s entity
    promotion), so resolution is namespace-wide; ``session_id`` is accepted for
    call-site symmetry with the rest of the flush path.

    Returns ``{"dropped": n, "merged": n}``.

    A conservative 4th layer (emb>=0.92 AND same-type AND co-occurring) is left
    as a deliberate, off-by-default hook: it is not run here because embedding
    similarity does not cleanly separate nickname merges from distinct
    co-occurring people (see module docstring).
    """
    dropped = await _drop_junk(store)
    merged_morph = await _merge_standardized(store)
    # Speaker-prefix repair runs BEFORE nicknames: a malformed
    # "caroline: thanks" superstring would otherwise look like a nickname
    # anchor for "caroline" and swallow the real person node.
    merged_speaker = await _merge_speaker_prefixed(store)
    merged_nick = await _merge_nicknames(store)
    dropped_temporal = await _drop_temporal(store)
    dropped += dropped_temporal
    merged = merged_morph + merged_nick + merged_speaker
    log.info(
        "resolve_entities: dropped=%d merged=%d "
        "(morphological=%d nickname=%d speaker_prefixed=%d temporal_dropped=%d)",
        dropped,
        merged,
        merged_morph,
        merged_nick,
        merged_speaker,
        dropped_temporal,
    )
    return {"dropped": dropped, "merged": merged}

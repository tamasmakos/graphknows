"""Unit tests for the layered consolidation-time entity resolver.

The pure key/standardization helpers are tested directly; the layer orchestration
is exercised against an in-memory ``FakeStore`` implementing the four primitives
``resolve_entities`` depends on (list_entities / distinct_speakers /
drop_entities / merge_entities), so the merge *decisions* are verified without a
live ArcadeDB.
"""

from __future__ import annotations

from typing import Any

import pytest

from graphknows.ingestion.consolidation.entity_resolution import (
    _singularize,
    _standard_key,
    resolve_entities,
)

# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "plural,singular",
    [
        ("paths", "path"),
        ("dances", "dance"),
        ("competitions", "competition"),
        ("successes", "success"),
        ("stories", "story"),
        ("glasses", "glass"),
        # Non-plural / gerunds must survive unchanged (no over-stripping).
        ("success", "success"),
        ("business", "business"),
        ("building", "building"),
        ("meeting", "meeting"),
        ("bus", "bus"),
    ],
)
def test_singularize(plural: str, singular: str) -> None:
    assert _singularize(plural) == singular


def test_standard_key_strips_determiner_and_singularizes_head() -> None:
    assert _standard_key("the dance competitions") == "dance competition"
    assert _standard_key("My Paths") == "path"
    assert _standard_key("Dance") == "dance"
    # head-noun singularized, compound modifier kept
    assert _standard_key("dance competition") == "dance competition"


# ---------------------------------------------------------------------------
# In-memory store for orchestration tests
# ---------------------------------------------------------------------------


class FakeStore:
    """Minimal in-memory stand-in for the four GraphStore primitives used."""

    def __init__(self, entities: list[dict[str, Any]], speakers: list[str]) -> None:
        self.entities: dict[str, dict[str, Any]] = {
            e["name_norm"]: {"aliases": [], **e} for e in entities
        }
        self.speakers = list(speakers)
        self.dropped: list[str] = []
        self.merges: list[tuple[str, str]] = []

    async def list_entities(self) -> list[dict[str, Any]]:
        return [dict(e) for e in self.entities.values()]

    async def distinct_speakers(self) -> list[str]:
        return list(self.speakers)

    async def drop_entities(self, name_norms: list[str]) -> int:
        n = 0
        for norm in name_norms:
            if norm in self.entities:
                del self.entities[norm]
                self.dropped.append(norm)
                n += 1
        return n

    async def rename_entity(self, old_norm: str, new_name: str) -> bool:
        new_norm = " ".join(new_name.split()).casefold()
        if old_norm not in self.entities or new_norm in self.entities or old_norm == new_norm:
            return False
        e = self.entities.pop(old_norm)
        e.setdefault("aliases", []).append(e["name"])
        e["name"], e["name_norm"] = new_name, new_norm
        self.entities[new_norm] = e
        return True

    async def merge_entities(self, winner_norm: str, loser_norm: str) -> bool:
        if winner_norm == loser_norm:
            return False
        if winner_norm not in self.entities or loser_norm not in self.entities:
            return False
        win = self.entities[winner_norm]
        lose = self.entities.pop(loser_norm)
        win["mention_count"] = win.get("mention_count", 0) + lose.get("mention_count", 0)
        win.setdefault("aliases", []).append(lose["name"])
        self.merges.append((winner_norm, loser_norm))
        return True


def _mel_caroline_store() -> FakeStore:
    return FakeStore(
        entities=[
            {"name_norm": "melanie", "name": "Melanie", "type": "PERSON", "mention_count": 40},
            {"name_norm": "mel", "name": "Mel", "type": "PERSON", "mention_count": 6},
            {"name_norm": "caroline", "name": "Caroline", "type": "PERSON", "mention_count": 38},
            {"name_norm": "others", "name": "others", "type": "", "mention_count": 9},
            {"name_norm": "kind", "name": "kind", "type": "", "mention_count": 4},
            {"name_norm": "paths", "name": "paths", "type": "", "mention_count": 2},
            {"name_norm": "path", "name": "path", "type": "", "mention_count": 5},
        ],
        speakers=["melanie", "caroline"],
    )


@pytest.mark.asyncio
async def test_resolver_end_to_end_mel_caroline_junk_and_plural() -> None:
    store = _mel_caroline_store()
    result = await resolve_entities(store, "conv-26")

    remaining = set(store.entities)
    # (a) Mel merged away; Melanie survives with Mel as an alias.
    assert "mel" not in remaining
    assert "melanie" in remaining
    assert "Mel" in store.entities["melanie"]["aliases"]
    assert ("melanie", "mel") in store.merges
    # (b) Caroline stays a distinct node (not a prefix of Melanie).
    assert "caroline" in remaining
    assert not any(loser == "caroline" for _, loser in store.merges)
    # (c) junk hub entities dropped.
    assert "others" not in remaining and "kind" not in remaining
    assert result["dropped"] == 2
    # (d) morphological merge: paths -> path (more mentions wins as canonical).
    assert "paths" not in remaining and "path" in remaining
    assert ("path", "paths") in store.merges
    # nickname + morphological both counted.
    assert result["merged"] == 2


@pytest.mark.asyncio
async def test_prefix_common_noun_not_merged_without_person_anchor() -> None:
    """A short common-noun prefix must NOT merge when the longer form is not a person."""
    store = FakeStore(
        entities=[
            # "art" is a >=3 alphabetic prefix of "article", but neither is a person.
            {"name_norm": "art", "name": "art", "type": "", "mention_count": 3},
            {"name_norm": "article", "name": "article", "type": "", "mention_count": 2},
        ],
        speakers=[],
    )
    result = await resolve_entities(store, "s1")
    assert result["merged"] == 0
    assert set(store.entities) == {"art", "article"}


@pytest.mark.asyncio
async def test_ambiguous_prefix_skipped() -> None:
    """A prefix shared by two person anchors is ambiguous and must be skipped."""
    store = FakeStore(
        entities=[
            {"name_norm": "ann", "name": "Ann", "type": "PERSON", "mention_count": 2},
            {"name_norm": "anna", "name": "Anna", "type": "PERSON", "mention_count": 5},
            {"name_norm": "anne", "name": "Anne", "type": "PERSON", "mention_count": 4},
        ],
        speakers=[],
    )
    result = await resolve_entities(store, "s1")
    # "ann" prefixes both "anna" and "anne" -> ambiguous -> no merge.
    assert result["merged"] == 0
    assert {"ann", "anna", "anne"} <= set(store.entities)


@pytest.mark.asyncio
async def test_dominant_anchor_wins_over_noise_variant() -> None:
    """Mel merges into Melanie even with a stray 1-mention 'Mell' typo anchor."""
    store = FakeStore(
        entities=[
            {"name_norm": "melanie", "name": "Melanie", "type": "PERSON", "mention_count": 63},
            {"name_norm": "mel", "name": "Mel", "type": "PERSON", "mention_count": 39},
            {"name_norm": "mell", "name": "Mell", "type": "PERSON", "mention_count": 1},
        ],
        speakers=[],
    )
    result = await resolve_entities(store, "conv-26")
    assert ("melanie", "mel") in store.merges
    assert "mel" not in store.entities
    assert result["merged"] == 1  # 'mell' has no longer superstring -> stays


@pytest.mark.asyncio
async def test_persons_not_singularized_in_standardization_layer() -> None:
    """A person whose name ends in 's' (Chris) is never singularized/merged in layer 2."""
    store = FakeStore(
        entities=[
            {"name_norm": "chris", "name": "Chris", "type": "PERSON", "mention_count": 7},
            {"name_norm": "chri", "name": "chri", "type": "", "mention_count": 1},
        ],
        speakers=["chris"],
    )
    result = await resolve_entities(store, "s1")
    # "chris" must not be singularized to "chri" and merged.
    assert "chris" in store.entities
    assert result["merged"] == 0


# ---------------------------------------------------------------------------
# Layers 4 + 5 — speaker-prefix merge and temporal drop (existing-graph repair)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_speaker_prefixed_entity_merges_into_the_speaker() -> None:
    """ "Caroline: Thanks, Melanie" heals into "Caroline" at flush.

    Repairs graphs ingested before the mint-point fix; on conv-26 these 27
    names carried 42.5% of REL edges.
    """
    store = FakeStore(
        entities=[
            {"name_norm": "caroline", "name": "Caroline", "type": "PERSON", "mention_count": 38},
            {
                "name_norm": "caroline: thanks, melanie",
                "name": "Caroline: Thanks, Melanie",
                "type": "PERSON",
                "mention_count": 3,
            },
        ],
        speakers=["caroline"],
    )
    await resolve_entities(store, "s1")
    assert "caroline: thanks, melanie" not in store.entities
    assert ("caroline", "caroline: thanks, melanie") in store.merges


@pytest.mark.asyncio
async def test_glued_group_without_a_clean_node_is_promoted() -> None:
    """When the glued span swallowed EVERY mention, the dominant member becomes X.

    The real conv-26 case: "Caroline" existed only as "Caroline: Thanks,
    Melanie" (63 mentions) plus untyped variants — no clean node to merge into.
    """
    store = FakeStore(
        entities=[
            {
                "name_norm": "caroline: thanks, melanie",
                "name": "Caroline: Thanks, Melanie",
                "type": "PERSON",
                "mention_count": 63,
            },
            {
                "name_norm": "caroline: nah, i",
                "name": "Caroline: Nah, I",
                "type": "",
                "mention_count": 0,
            },
            {"name_norm": "caroline:", "name": "Caroline:", "type": "", "mention_count": 0},
        ],
        speakers=[],
    )
    await resolve_entities(store, "s1")
    assert set(store.entities) == {"caroline"}
    winner = store.entities["caroline"]
    assert winner["name"] == "Caroline"
    assert winner["mention_count"] == 63
    assert "Caroline: Thanks, Melanie" in winner["aliases"]


@pytest.mark.asyncio
async def test_colon_name_without_a_person_anchor_is_kept() -> None:
    """ "Charlotte's Web: A Review" has no person anchor — never merged."""
    store = FakeStore(
        entities=[
            {
                "name_norm": "charlotte's web: a review",
                "name": "Charlotte's Web: A Review",
                "type": "WORK_OF_ART",
                "mention_count": 2,
            },
        ],
        speakers=["caroline"],
    )
    await resolve_entities(store, "s1")
    assert "charlotte's web: a review" in store.entities
    assert store.merges == []


@pytest.mark.asyncio
async def test_temporal_entities_from_old_ingests_are_dropped() -> None:
    """Date names that entered via the (formerly unfiltered) relation path go."""
    store = FakeStore(
        entities=[
            {"name_norm": "caroline", "name": "Caroline", "type": "PERSON", "mention_count": 38},
            {"name_norm": "8 may, 2023", "name": "8 May, 2023", "type": "", "mention_count": 2},
            {"name_norm": "last saturday", "name": "last Saturday", "type": "", "mention_count": 1},
            # Month-named person survives — the digit gate protects her.
            {"name_norm": "may", "name": "May", "type": "PERSON", "mention_count": 4},
        ],
        speakers=["caroline"],
    )
    await resolve_entities(store, "s1")
    assert "8 may, 2023" not in store.entities
    assert "last saturday" not in store.entities
    assert "may" in store.entities


@pytest.mark.asyncio
async def test_standardization_elects_the_singular_even_when_the_plural_leads() -> None:
    """The standardization layer must name the survivor for the SINGULAR form.

    ``_canonical`` breaks ties on the longest surface, which in a {style, styles}
    group is always the plural — so the layer whose stated job is to "collapse
    morphological (plural) variants" was standardising onto the plural.

    The observable damage is provenance: a MENTIONS edge points at a name its own
    chunk never used. Measured on conv-30 (namespace lg30v3) 38 of 2491 MENTIONS
    were in that state -- ``downs`` x8, ``dreams`` x7, ``customers``, ``plans``,
    ``projects``, ``styles``. ``downs``, typed EMOTION, was minted from "don't
    let it get you down".

    Here the plural also has MORE mentions, so this pins the rule rather than the
    tiebreak: singular wins on being the standard form, not on counting.
    """
    store = FakeStore(
        entities=[
            {"name_norm": "downs", "name": "downs", "type": "EMOTION", "mention_count": 8},
            {"name_norm": "down", "name": "down", "type": "EMOTION", "mention_count": 2},
        ],
        speakers=["gina"],
    )
    await resolve_entities(store, "conv-30")

    assert set(store.entities) == {"down"}, "the singular must be the surviving node"
    assert store.entities["down"]["name"] == "down"
    # The merge is not lost — the plural's mentions come across.
    assert store.entities["down"]["mention_count"] == 10


@pytest.mark.asyncio
async def test_all_plural_group_still_merges() -> None:
    """No singular member present -> fall back to the old rule rather than skip."""
    store = FakeStore(
        entities=[
            {"name_norm": "glasses", "name": "glasses", "type": "ARTIFACT", "mention_count": 5},
            {"name_norm": "glass", "name": "glass", "type": "ARTIFACT", "mention_count": 1},
        ],
        speakers=["gina"],
    )
    await resolve_entities(store, "conv-30")
    assert set(store.entities) == {"glass"}


@pytest.mark.asyncio
async def test_nickname_layer_still_prefers_the_longer_surface() -> None:
    """The singular rule is scoped to standardization; 'mel' must still lose to 'Melanie'."""
    store = _mel_caroline_store()
    await resolve_entities(store, "conv-26")
    assert "melanie" in store.entities
    assert "mel" not in store.entities

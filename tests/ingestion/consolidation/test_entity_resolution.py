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

"""T061/FR-018: recall commits the same-as candidates of the entities it recalled.

Mocked client, no ArcadeDB: what is pinned is that the candidate plane is read
back at recall time, that a recalled entity's candidates end at one committed
entity, and that the commitment is reported in the counters rather than logged.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

from graphknows.retrieval.retriever import SymbolRecall
from graphknows.storage.arcadedb._base import ArcadeStoreBase

_CONCEPTS = [{"uri": "demo:Person", "label": "Person"}]


def _row(subject: str, object_: str = "e2") -> dict[str, Any]:
    """One activation row: an assertion and the segment it was read from."""
    return {
        "id": f"f-{subject}",
        "subject": subject,
        "predicate": "demo:built",
        "object": object_,
        "polarity": "asserted",
        "modality": "",
        "confidence": 1.0,
        "valid_from": "",
        "valid_to": "",
        "text": "Ada designed the Engine.",
        "byte_start": 0,
        "byte_end": 24,
        "source_uri": "file:///notes.md",
    }


def _store(facts: list[dict], candidates: list[dict]) -> ArcadeStoreBase:
    """A store answering the concept lookup, activation, and the candidate plane."""

    async def answer(_db: str, cypher: str, params: dict | None = None) -> list[dict]:
        if "SAME_AS_CANDIDATE" in cypher:
            return candidates
        if "EVOKES" in cypher:
            return facts
        if "INSTANCE_OF" in cypher:
            return []
        return _CONCEPTS

    client = MagicMock()
    client.query = AsyncMock(side_effect=answer)
    return ArcadeStoreBase(client, "db")


def _candidate(source: str, target: str, at: str) -> dict[str, Any]:
    return {"source": source, "target": target, "at": at}


async def test_candidates_unify_into_one_committed_entity() -> None:
    """A chain of candidates off a recalled entity ends at a single entity."""
    store = _store(
        [_row("ada"), _row("ada-lovelace")],
        [
            _candidate("ada", "ada-lovelace", "2024-01-01T00:00:00+00:00"),
            _candidate("ada-lovelace", "ada-l", "2024-01-02T00:00:00+00:00"),
        ],
    )

    result = await SymbolRecall(store).recall("person")

    assert {item.fact.subject for item in result.facts} == {"ada-l"}
    assert result.counters.merges_committed == 2


async def test_no_candidates_commits_nothing() -> None:
    """Without a candidate edge the recalled entity is returned untouched."""
    store = _store([_row("ada")], [])

    result = await SymbolRecall(store).recall("person")

    assert [item.fact.subject for item in result.facts] == ["ada"]
    assert result.counters.merges_committed == 0

"""flush_memory runs its pipeline steps independently.

The steps used to share one try/except, so the first failure skipped every
later step — including ``consolidate_session``, the raw -> consolidated state
flip that is the point of the flush — and the reported error did not say which
stage had degraded.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from graphknows.memory import Memory


def _store() -> MagicMock:
    store = MagicMock()
    store._ensure_analytical_view = AsyncMock()
    # No buffered turns in these tests — the drain step is exercised separately
    # in tests/memory/test_turn_buffer.py (integration, live ArcadeDB).
    store.list_turns = AsyncMock(return_value=[])
    store.mark_turns_consolidated = AsyncMock(return_value=0)
    store.compute_and_persist_pagerank = AsyncMock(return_value=3)
    store.compute_graph_embeddings = AsyncMock(return_value=4)
    store.compute_communities = AsyncMock(return_value=5)
    store.consolidate_session = AsyncMock(return_value=7)
    store.consolidate_relations = AsyncMock(
        return_value={"deduped": 0, "contradictions": 0, "invalidated": 0}
    )
    store.verifier_coverage = AsyncMock(return_value={})
    store.entity_coverage = AsyncMock(return_value={"total": 10, "connected": 8, "isolated": 2})
    return store


def _memory(store: MagicMock) -> Memory:
    mem = Memory(settings=MagicMock(namespace=""), namespace="")
    mem._get_retriever = AsyncMock(return_value=MagicMock())
    mem._store = store
    return mem


@pytest.mark.asyncio
async def test_happy_path_reports_every_step() -> None:
    store = _store()
    with patch(
        "graphknows.ingestion.consolidation.resolve_entities",
        AsyncMock(return_value={"dropped": 1, "merged": 2}),
    ):
        out = await _memory(store).flush_memory("s1")

    assert out["errors"] == []
    assert out["nodes"] == 3
    assert out["graph_embeddings_computed"] == 4
    assert out["communities_computed"] == 5
    assert out["chunks"] == 7
    assert out["merged_entities"] == 2
    assert out["entity_coverage"] == {"total": 10, "connected": 8, "isolated": 2}


@pytest.mark.asyncio
async def test_analytics_failure_still_consolidates() -> None:
    """A failing analytics step must not skip the session state flip."""
    store = _store()
    store.compute_and_persist_pagerank = AsyncMock(side_effect=Exception("pagerank boom"))

    with patch(
        "graphknows.ingestion.consolidation.resolve_entities",
        AsyncMock(return_value={"dropped": 0, "merged": 0}),
    ):
        out = await _memory(store).flush_memory("s1")

    store.consolidate_session.assert_awaited_once()
    assert out["chunks"] == 7, "consolidate_session was skipped by an analytics failure"
    # Later analytics steps still ran too.
    assert out["communities_computed"] == 5
    # The error names the step that failed.
    assert any(e.startswith("pagerank:") for e in out["errors"]), out["errors"]
    assert out["nodes"] == 0


@pytest.mark.asyncio
async def test_drain_turns_per_turn_errors_are_surfaced() -> None:
    """A per-turn extraction failure inside drain_turns must not vanish.

    STMService catches per-turn errors into IngestResult.errors without
    raising, so the ``step()`` wrapper around drain_turns sees a clean
    return and cannot report it on its own — flush_memory must read
    ``errors`` off the drained result itself.
    """
    store = _store()
    store.list_turns = AsyncMock(return_value=[{"role": "user", "text": "hi"}])

    mem = _memory(store)
    mem._ingest_buffered_turns = AsyncMock(  # type: ignore[method-assign]
        return_value={
            "turns": 1,
            "chunks": 0,
            "abstentions": {},
            "errors": ["wordnet: Resource 'wordnet' not found"],
        }
    )

    with patch(
        "graphknows.ingestion.consolidation.resolve_entities",
        AsyncMock(return_value={"dropped": 0, "merged": 0}),
    ):
        out = await mem.flush_memory("s1")

    assert any(e == "drain_turns: wordnet: Resource 'wordnet' not found" for e in out["errors"]), (
        out["errors"]
    )


@pytest.mark.asyncio
async def test_connect_failure_reports_and_returns_defaults() -> None:
    """Without a store no step can run; the result says so rather than raising."""
    mem = Memory(settings=MagicMock(namespace=""), namespace="")
    mem._get_retriever = AsyncMock(side_effect=Exception("no db"))

    out = await mem.flush_memory("s1")

    assert any(e.startswith("connect:") for e in out["errors"]), out["errors"]
    assert out["chunks"] == 0

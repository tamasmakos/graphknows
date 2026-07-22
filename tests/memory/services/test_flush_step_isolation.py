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
    store.compute_and_persist_pagerank = AsyncMock(return_value=3)
    store.compute_graph_embeddings = AsyncMock(return_value=4)
    store.compute_communities = AsyncMock(return_value=5)
    store.compute_community_quality = AsyncMock(
        return_value={"modularity": 0.5, "communities": 2, "communities_scored": 2}
    )
    store.consolidate_session = AsyncMock(return_value=7)
    return store


def _memory(store: MagicMock) -> Memory:
    mem = Memory(settings=MagicMock(namespace=""), namespace="")
    mem._get_retriever = AsyncMock(return_value=MagicMock())
    mem._retriever_stores = [store]
    # Topics are their own stage now (Memory._flush_topics -> graphknows.topics),
    # so stub the stage rather than a store method. This test is about step
    # isolation in flush_memory, not about clustering.
    mem._flush_topics = AsyncMock(return_value=6)  # type: ignore[method-assign]
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
    assert out["topics_created"] == 6
    assert out["chunks"] == 7
    assert out["merged_entities"] == 2


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
async def test_connect_failure_reports_and_returns_defaults() -> None:
    """Without a store no step can run; the result says so rather than raising."""
    mem = Memory(settings=MagicMock(namespace=""), namespace="")
    mem._get_retriever = AsyncMock(side_effect=Exception("no db"))

    out = await mem.flush_memory("s1")

    assert any(e.startswith("connect:") for e in out["errors"]), out["errors"]
    assert out["chunks"] == 0

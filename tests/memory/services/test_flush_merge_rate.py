"""flush_memory must report ``merge_rate``, so a bare merge count becomes
interpretable against the population it was drawn from.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from graphknows.memory import Memory


def _store() -> MagicMock:
    store = MagicMock()
    store._ensure_analytical_view = AsyncMock()
    store.list_turns = AsyncMock(return_value=[])
    store.mark_turns_consolidated = AsyncMock(return_value=0)
    store.compute_and_persist_pagerank = AsyncMock(return_value=3)
    store.compute_graph_embeddings = AsyncMock(return_value=4)
    store.compute_communities = AsyncMock(return_value=5)
    store.consolidate_session = AsyncMock(return_value=7)
    store.verifier_coverage = AsyncMock(return_value={})
    return store


def _memory(store: MagicMock) -> Memory:
    mem = Memory(settings=MagicMock(namespace=""), namespace="")
    mem._get_retriever = AsyncMock(return_value=MagicMock())
    mem._store = store
    mem._flush_topics = AsyncMock(  # type: ignore[method-assign]
        return_value={"topics_created": 6, "modularity": 0.5}
    )
    return mem


@pytest.mark.asyncio
async def test_merge_rate_is_merged_over_considered() -> None:
    store = _store()
    with patch(
        "graphknows.ingestion.consolidation.resolve_entities",
        AsyncMock(return_value={"dropped": 1, "merged": 2, "considered": 10}),
    ):
        out = await _memory(store).flush_memory("s1")

    assert out["merge_rate"] == pytest.approx(0.2)


@pytest.mark.asyncio
async def test_merge_rate_degrades_safely_without_considered() -> None:
    """Existing callers patch resolve_entities without 'considered' — flush
    must still return, with merge_rate reported as 0.0 rather than raising.
    """
    store = _store()
    with patch(
        "graphknows.ingestion.consolidation.resolve_entities",
        AsyncMock(return_value={"dropped": 0, "merged": 0}),
    ):
        out = await _memory(store).flush_memory("s1")

    assert out["merge_rate"] == 0.0

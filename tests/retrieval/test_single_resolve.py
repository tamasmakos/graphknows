"""One ``retrieve()`` resolves the query's entities once (FR-019, SC-013).

The fact sheet and the frame re-rank both need the same names from the same
``words_graph``; they used to issue two identical graph reads. The read now
lives on the per-query ``ChannelContext``, so it is shared — and still lazy,
so a run where neither consumer is enabled never touches the store.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from graphknows.retrieval.retriever import DETRetriever
from graphknows.settings import get_settings


def _fake_embed(texts, model):
    return np.array([[0.1] * 4 for _ in texts])


def _make_retriever() -> tuple[DETRetriever, MagicMock]:
    store = MagicMock()
    store.query_entity_chunks = AsyncMock(return_value=[])
    store.full_text_search_chunks = AsyncMock(return_value=[])
    store.query_temporal_chunks = AsyncMock(return_value=[])
    store.query_dated_chunks_by_entities = AsyncMock(return_value=[])
    store.search_chunks_ann = AsyncMock(
        return_value=[{"chunk_id": "c1", "text": "jon in rome", "cosine_score": 0.9}]
    )
    store.get_chunks_by_ids = AsyncMock(return_value={})
    store.neighbor_chunks = AsyncMock(return_value={})
    store.resolve_query_entities = AsyncMock(return_value=["Jon"])
    store.entity_fact_context = AsyncMock(return_value=["Jon visiting Rome"])
    store.frame_rank_signals = AsyncMock(return_value=([], {}, 0))
    return DETRetriever(store=store), store


async def _retrieve(retriever: DETRetriever):
    """Run a retrieve with the frame re-rank forced past its model-backed parts."""
    with (
        patch("graphknows.retrieval.retriever.embed", side_effect=_fake_embed),
        patch("graphknows.ranking.frame_boost.enabled", return_value=True),
        patch(
            "graphknows.symbolic.framenet.index.match_frames",
            return_value=[("Motion", 0.4)],
        ),
        patch(
            "graphknows.symbolic.framenet.framenet.expand_frames",
            return_value={"Motion"},
        ),
    ):
        return await retriever.retrieve("where did jon go", session_id="s1", scope="both")


@pytest.mark.asyncio
async def test_both_consumers_share_one_resolve(monkeypatch) -> None:
    monkeypatch.setenv("GRAPHKNOWS_FACT_CONTEXT", "true")
    get_settings.cache_clear()
    retriever, store = _make_retriever()

    ctx = await _retrieve(retriever)

    store.resolve_query_entities.assert_awaited_once()
    # Both consumers were served, and served the SAME names.
    anchors = store.frame_rank_signals.await_args.args[2]
    assert anchors == ["Jon"]
    assert store.entity_fact_context.await_args.args[0] == ["Jon"]
    assert ctx.facts == ["Jon visiting Rome"]


@pytest.mark.asyncio
async def test_no_consumer_means_no_resolve(monkeypatch) -> None:
    monkeypatch.setenv("GRAPHKNOWS_FACT_CONTEXT", "false")
    get_settings.cache_clear()
    retriever, store = _make_retriever()

    with (
        patch("graphknows.retrieval.retriever.embed", side_effect=_fake_embed),
        patch("graphknows.ranking.frame_boost.enabled", return_value=False),
    ):
        await retriever.retrieve("where did jon go", session_id="s1", scope="both")

    store.resolve_query_entities.assert_not_awaited()

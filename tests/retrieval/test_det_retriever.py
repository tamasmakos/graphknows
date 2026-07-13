"""Tests for DETRetriever: collectors, RRF fusion, and fail-loud error propagation.

Golden-layer (Phase 1): the retriever reads a single ``GraphStore`` with a
lifecycle ``state`` filter. Collectors are mocked as AsyncMocks on that store.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from graphknows.retrieval.retriever import DETRetriever


def _fake_embed(texts, model):
    return np.array([[0.1] * 4 for _ in texts])


def _make_retriever():
    store = MagicMock()
    # Retrieval is fail-loud: every awaited store method a collector touches must
    # be an AsyncMock, or the (un-swallowed) await error propagates. Defaults
    # return empty; individual tests override the channels they exercise.
    store.query_entity_chunks = AsyncMock(return_value=[])
    store.full_text_search_chunks = AsyncMock(return_value=[])
    store.query_temporal_chunks = AsyncMock(return_value=[])
    store.search_topics = AsyncMock(return_value=[])
    store.search_chunks_ann = AsyncMock(return_value=[])
    store.get_chunks_by_ids = AsyncMock(return_value={})
    retriever = DETRetriever(store=store)
    return retriever, store


@pytest.mark.asyncio
async def test_all_collectors_contribute():
    retriever, store = _make_retriever()

    store.query_entity_chunks = AsyncMock(
        return_value=[{"chunk_id": "ent-c1", "text": "entity chunk", "entities": ["alice"]}]
    )
    store.full_text_search_chunks = AsyncMock(
        return_value=[{"chunk_id": "bm25-c1", "text": "alice policy text"}]
    )
    store.search_chunks_ann = AsyncMock(
        return_value=[{"chunk_id": "sem-c1", "text": "semantic chunk", "cosine_score": 0.9}]
    )

    with patch("graphknows.retrieval.retriever.embed", side_effect=_fake_embed):
        ctx = await retriever.retrieve("alice policy", session_id="s1", scope="both")

    chunk_ids = {fr["chunk_id"] for fr in ctx.fused_results}
    assert "ent-c1" in chunk_ids
    assert "bm25-c1" in chunk_ids
    assert "sem-c1" in chunk_ids
    assert not any(cid.startswith("topic:") for cid in chunk_ids)


@pytest.mark.asyncio
async def test_multi_source_chunk_ranks_higher():
    retriever, store = _make_retriever()

    store.query_entity_chunks = AsyncMock(
        return_value=[{"chunk_id": "shared", "text": "shared text", "entities": ["alice"]}]
    )
    store.search_chunks_ann = AsyncMock(
        return_value=[
            {"chunk_id": "shared", "text": "shared text", "cosine_score": 0.9},
            {"chunk_id": "solo", "text": "solo text", "cosine_score": 0.8},
        ]
    )

    with (
        patch("graphknows.retrieval.retriever.embed", side_effect=_fake_embed),
        # Test fusion order deterministically, independent of whether the
        # cross-encoder model happens to be cached in this environment.
        patch("graphknows.retrieval.retriever._ce_rerank", side_effect=lambda q, c, k: c[:k]),
    ):
        ctx = await retriever.retrieve("alice", session_id="s1", scope="stm")

    by_id = {fr["chunk_id"]: fr for fr in ctx.fused_results}
    assert by_id["shared"]["rrf_score"] > by_id["solo"]["rrf_score"]


@pytest.mark.asyncio
async def test_no_synthetic_topic_chunks_without_topics():
    retriever, store = _make_retriever()

    store.search_chunks_ann = AsyncMock(
        return_value=[
            {"chunk_id": "plain-hit", "text": "climate policy", "cosine_score": 0.9},
        ]
    )

    with (
        patch("graphknows.retrieval.retriever.embed", side_effect=_fake_embed),
        patch("graphknows.retrieval.retriever._ce_rerank", side_effect=lambda q, c, k: c[:k]),
    ):
        ctx = await retriever.retrieve("climate policy", session_id="s1", scope="stm")

    assert ctx.fused_results[0]["chunk_id"] == "plain-hit"
    assert not any(fr["chunk_id"].startswith("topic:") for fr in ctx.fused_results)


@pytest.mark.asyncio
async def test_top_k_is_respected():
    retriever, store = _make_retriever()

    store.query_entity_chunks = AsyncMock(
        return_value=[{"chunk_id": f"e{i}", "text": f"entity {i}"} for i in range(8)]
    )

    with patch("graphknows.retrieval.retriever.embed", side_effect=_fake_embed):
        ctx = await retriever.retrieve("query words here", session_id="s1", top_k=3)

    assert len(ctx.fused_results) <= 3


@pytest.mark.asyncio
async def test_collector_failure_raises_loudly():
    """A collector failure must propagate — no silent graceful degradation. A
    broken channel should surface immediately, not masquerade as thin recall."""
    retriever, store = _make_retriever()

    store.query_entity_chunks = AsyncMock(side_effect=RuntimeError("store down"))

    with (
        patch("graphknows.retrieval.retriever.embed", side_effect=_fake_embed),
        pytest.raises(RuntimeError, match="store down"),
    ):
        await retriever.retrieve("query text here", session_id="s1")


@pytest.mark.asyncio
async def test_scope_maps_to_state_filter():
    """scope stm/ltm/both must be threaded to the store as the state filter."""
    retriever, store = _make_retriever()

    with patch("graphknows.retrieval.retriever.embed", side_effect=_fake_embed):
        await retriever.retrieve("q text", session_id="s1", scope="ltm")

    # semantic_ann calls search_chunks_ann(emb, k, session_id, state).
    _emb, _k, _sid, state = store.search_chunks_ann.await_args.args
    assert state == "consolidated"

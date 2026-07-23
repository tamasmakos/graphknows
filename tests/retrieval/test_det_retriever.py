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
    store.search_topics_with_chunks = AsyncMock(return_value=[])
    store.search_chunks_ann = AsyncMock(return_value=[])
    store.get_chunks_by_ids = AsyncMock(return_value={})
    store.chunk_community_weights = AsyncMock(return_value={})
    store.resolve_query_entities = AsyncMock(return_value=[])
    store.entity_fact_context = AsyncMock(return_value=[])
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
async def test_fact_context_attached_when_enabled(monkeypatch):
    monkeypatch.setenv("GRAPHKNOWS_FACT_CONTEXT", "true")
    retriever, store = _make_retriever()
    store.resolve_query_entities = AsyncMock(return_value=["Jon"])
    store.entity_fact_context = AsyncMock(return_value=["Jon visiting Rome"])

    with patch("graphknows.retrieval.retriever.embed", side_effect=_fake_embed):
        ctx = await retriever.retrieve("where did jon go", session_id="s1", scope="both")

    assert ctx.facts == ["Jon visiting Rome"]
    store.resolve_query_entities.assert_awaited_once()


@pytest.mark.asyncio
async def test_fact_context_disabled_by_flag(monkeypatch):
    monkeypatch.setenv("GRAPHKNOWS_FACT_CONTEXT", "false")
    retriever, store = _make_retriever()
    store.resolve_query_entities = AsyncMock(return_value=["Jon"])
    store.entity_fact_context = AsyncMock(return_value=["Jon visiting Rome"])

    with patch("graphknows.retrieval.retriever.embed", side_effect=_fake_embed):
        ctx = await retriever.retrieve("where did jon go", session_id="s1", scope="both")

    assert ctx.facts == []
    store.resolve_query_entities.assert_not_awaited()


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


def _meta(**per_chunk):
    """chunk_rerank_meta shape: {cid: {emb, ents, file}} with defaults."""
    return {
        cid: {"emb": spec.get("emb"), "ents": spec.get("ents", []), "file": spec.get("file", "")}
        for cid, spec in per_chunk.items()
    }


@pytest.mark.asyncio
async def test_rerank_orders_the_pool_by_query_similarity():
    """Channels decide pool MEMBERSHIP; the rerank decides the ORDER.

    Measured on conv-26: recall@10 0.739 -> 0.848, multi-hop@10 0.548 -> 0.704.
    """
    retriever, store = _make_retriever()
    store.chunk_rerank_meta = AsyncMock(
        return_value=_meta(far={"emb": [0.0, 1.0]}, near={"emb": [1.0, 0.0]})
    )
    # RRF put the semantically-distant chunk first.
    pool = [{"chunk_id": "far", "rrf_score": 0.9}, {"chunk_id": "near", "rrf_score": 0.1}]

    out = await retriever._rerank_pool([1.0, 0.0], pool)

    assert [c["chunk_id"] for c in out] == ["near", "far"]
    assert out[0]["cosine"] == 1.0


@pytest.mark.asyncio
async def test_rerank_keeps_unscorable_candidates_last():
    """A chunk with no stored embedding is demoted, never dropped."""
    retriever, store = _make_retriever()
    store.chunk_rerank_meta = AsyncMock(return_value=_meta(has={"emb": [1.0, 0.0]}))
    pool = [{"chunk_id": "none", "rrf_score": 0.9}, {"chunk_id": "has", "rrf_score": 0.1}]

    out = await retriever._rerank_pool([1.0, 0.0], pool)

    assert [c["chunk_id"] for c in out] == ["has", "none"]


@pytest.mark.asyncio
async def test_rerank_falls_back_to_rrf_order_on_store_failure():
    retriever, store = _make_retriever()
    store.chunk_rerank_meta = AsyncMock(side_effect=RuntimeError("boom"))
    pool = [{"chunk_id": "a", "rrf_score": 0.9}, {"chunk_id": "b", "rrf_score": 0.1}]

    assert await retriever._rerank_pool([1.0, 0.0], pool) == pool


@pytest.mark.asyncio
async def test_rerank_coverage_prefers_new_entities_and_documents():
    """Structure breaks cosine ties: a near-duplicate about the SAME entities
    and document loses its slot to a slightly-less-similar chunk that brings
    NEW coverage — the beam multi-session fix.
    """
    retriever, store = _make_retriever()
    store.chunk_rerank_meta = AsyncMock(return_value=_meta(
        top={"emb": [1.0, 0.0], "ents": ["caroline"], "file": "f1"},
        dup={"emb": [0.99, 0.14], "ents": ["caroline"], "file": "f1"},
        new={"emb": [0.95, 0.31], "ents": ["melanie", "seattle"], "file": "f2"},
    ))
    pool = [
        {"chunk_id": "top", "rrf_score": 0.9},
        {"chunk_id": "dup", "rrf_score": 0.8},
        {"chunk_id": "new", "rrf_score": 0.1},
    ]

    out = await retriever._rerank_pool([1.0, 0.0], pool)

    # Pure cosine would give top, dup, new; coverage promotes `new` past `dup`.
    assert [c["chunk_id"] for c in out] == ["top", "new", "dup"]


@pytest.mark.asyncio
async def test_rerank_zero_weights_is_pure_cosine(monkeypatch):
    """The kill switch: weights 0/0 must reproduce plain cosine ordering."""
    from graphknows.retrieval import retriever as mod

    monkeypatch.setattr(mod, "_RERANK_W_ENT", 0.0)
    monkeypatch.setattr(mod, "_RERANK_W_DOC", 0.0)
    retriever, store = _make_retriever()
    store.chunk_rerank_meta = AsyncMock(return_value=_meta(
        a={"emb": [1.0, 0.0], "ents": ["x"], "file": "f1"},
        b={"emb": [0.9, 0.44], "ents": ["y", "z"], "file": "f2"},
    ))
    pool = [{"chunk_id": "b", "rrf_score": 0.9}, {"chunk_id": "a", "rrf_score": 0.1}]

    out = await retriever._rerank_pool([1.0, 0.0], pool)

    assert [c["chunk_id"] for c in out] == ["a", "b"]

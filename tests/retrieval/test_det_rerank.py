"""Tests for WS-2 additions: BM25 sparse channel and cross-encoder rerank.

Covers:
1. rerank() no-ops gracefully when the model library is absent (returns input unchanged).
2. BM25 fallback scorer ranks an exact-term document above a non-matching document.
3. BM25 channel: _stm_bm25 contributes a source entry for matching queries.
4. pagerank_boost / idf_weight helpers behave correctly at boundaries.
5. retrieve() signature remains (query, session_id, top_k, scope) -> RetrievalContext.
"""

from __future__ import annotations

import math
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from graphknows.memory.query._rrf_common import idf_weight, pagerank_boost
from graphknows.memory.query.rerank import rerank
from graphknows.memory.query.retriever import DETRetriever, _bm25_rank


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fake_embed(texts, model):
    return np.array([[0.1] * 4 for _ in texts])


def _make_retriever():
    pg = MagicMock()
    neo = MagicMock()
    mg = MagicMock()
    # Retrieval is fail-loud: every awaited store method a collector touches must
    # be an AsyncMock or the (un-swallowed) await error propagates. Empty defaults;
    # individual tests override the channels they exercise.
    mg.query_fact_subjects = AsyncMock(return_value=[])
    mg.query_entity_chunks = AsyncMock(return_value=[])
    mg.full_text_search_chunks = AsyncMock(return_value=[])
    neo._qry = AsyncMock(return_value=[])
    neo.full_text_search_chunks = AsyncMock(return_value=[])
    pg.search_topics = AsyncMock(return_value=[])
    pg.search_stm_chunks = AsyncMock(return_value=[])
    pg.search_stm_turns = AsyncMock(return_value=[])
    pg.search_ltm_chunks = AsyncMock(return_value=[])
    pg.search_ltm_turns = AsyncMock(return_value=[])
    pg.get_chunks_by_ids = AsyncMock(return_value={})
    return DETRetriever(pg=pg, neo=neo, mg=mg), pg, neo, mg


# ---------------------------------------------------------------------------
# rerank() — offline / no-op path
# ---------------------------------------------------------------------------


def test_rerank_noop_when_model_absent():
    """When sentence_transformers is not installed, rerank() returns input unchanged."""
    candidates = [
        {"chunk_id": "c1", "text": "hello world", "rrf_score": 0.9, "rank": 1},
        {"chunk_id": "c2", "text": "foo bar", "rrf_score": 0.5, "rank": 2},
    ]
    # Force the model-unavailable path by patching the cached model state.
    from graphknows.memory.query import rerank as rerank_mod

    sentinel = rerank_mod._NO_MODEL
    with patch.object(rerank_mod, "_model", sentinel), patch.object(
        rerank_mod, "_model_loaded", True
    ):
        result = rerank("what is hello world", candidates, top_k=2)

    assert result == candidates[:2]
    # Order preserved — no mutation.
    assert result[0]["chunk_id"] == "c1"
    assert result[1]["chunk_id"] == "c2"


def test_rerank_noop_empty_candidates():
    """rerank() with empty candidates returns an empty list immediately."""
    result = rerank("any query", [], top_k=5)
    assert result == []


def test_rerank_top_k_respected_in_noop():
    """In no-op mode, top_k is still respected."""
    candidates = [
        {"chunk_id": f"c{i}", "text": f"text {i}", "rrf_score": float(i), "rank": i + 1}
        for i in range(10)
    ]
    from graphknows.memory.query import rerank as rerank_mod

    sentinel = rerank_mod._NO_MODEL
    with patch.object(rerank_mod, "_model", sentinel), patch.object(
        rerank_mod, "_model_loaded", True
    ):
        result = rerank("query", candidates, top_k=3)

    assert len(result) == 3


def test_rerank_import_does_not_require_torch():
    """Importing from rerank must succeed even without torch installed."""
    # This test passes by virtue of the import at the top of this module.
    # Explicitly reload to simulate a fresh import.
    import importlib

    import graphknows.memory.query.rerank as mod

    importlib.reload(mod)
    assert callable(mod.rerank)


# ---------------------------------------------------------------------------
# BM25 fallback scorer
# ---------------------------------------------------------------------------


def test_bm25_fallback_ranks_exact_match_above_nonmatch():
    """The BM25 fallback must rank an exact-term document above a non-matching doc."""
    query_tokens = ["violin", "concert"]
    chunks = [
        {"chunk_id": "c_match", "text": "she played violin at the concert last night"},
        {"chunk_id": "c_miss", "text": "the weather is nice today in the city"},
    ]
    ranked = _bm25_rank(query_tokens, chunks)
    assert len(ranked) >= 1
    assert ranked[0][1]["chunk_id"] == "c_match"


def test_bm25_fallback_returns_empty_on_no_match():
    """When no document contains any query token, ranked list is empty."""
    query_tokens = ["xyzzy", "quux"]
    chunks = [
        {"chunk_id": "c1", "text": "hello world"},
        {"chunk_id": "c2", "text": "the quick brown fox"},
    ]
    ranked = _bm25_rank(query_tokens, chunks)
    assert ranked == []


def test_bm25_fallback_empty_inputs():
    """Empty inputs return empty results without error."""
    assert _bm25_rank([], [{"chunk_id": "c1", "text": "hello"}]) == []
    assert _bm25_rank(["hello"], []) == []
    assert _bm25_rank([], []) == []


def test_bm25_multi_term_document_scores_higher():
    """A document containing both query terms scores higher than one with just one."""
    query_tokens = ["adoption", "family"]
    chunks = [
        {"chunk_id": "both", "text": "the adoption process was handled by a caring family"},
        {"chunk_id": "one", "text": "the adoption was finalized this week"},
    ]
    ranked = _bm25_rank(query_tokens, chunks)
    scores_by_id = {c["chunk_id"]: s for s, c in ranked}
    assert scores_by_id["both"] > scores_by_id["one"]


# ---------------------------------------------------------------------------
# PageRank boost helper
# ---------------------------------------------------------------------------


def test_pagerank_boost_zero_is_noop():
    assert pagerank_boost(0.0) == 1.0


def test_pagerank_boost_none_is_noop():
    assert pagerank_boost(None) == 1.0  # type: ignore[arg-type]


def test_pagerank_boost_bounded():
    from graphknows.memory.query._rrf_common import PAGERANK_MAX_BOOST

    assert pagerank_boost(1.0) == pytest.approx(1.0 + PAGERANK_MAX_BOOST, abs=1e-9)
    assert 1.0 < pagerank_boost(0.5) < 1.0 + PAGERANK_MAX_BOOST


def test_pagerank_boost_monotone():
    pr_vals = [0.0, 0.1, 0.3, 0.7, 1.0]
    boosts = [pagerank_boost(p) for p in pr_vals]
    for i in range(len(boosts) - 1):
        assert boosts[i] <= boosts[i + 1]


# ---------------------------------------------------------------------------
# IDF weight helper
# ---------------------------------------------------------------------------


def test_idf_weight_rare_entity_is_one():
    """Entity in <50% of corpus gets weight 1.0."""
    assert idf_weight(5, 100) == 1.0  # 5% < 50%


def test_idf_weight_ubiquitous_entity_penalised():
    """Entity in 100% of corpus gets weight 0.5 (saturation_ratio / 1.0)."""
    w = idf_weight(100, 100)
    assert w < 1.0
    assert w == pytest.approx(0.5, abs=1e-9)


def test_idf_weight_at_boundary():
    """Entity at exactly 50% corpus frequency gets weight 1.0 (boundary)."""
    assert idf_weight(50, 100) == pytest.approx(1.0, abs=1e-9)


def test_idf_weight_zero_corpus():
    assert idf_weight(5, 0) == 1.0


def test_idf_weight_zero_freq():
    assert idf_weight(0, 100) == 1.0


# ---------------------------------------------------------------------------
# STM BM25 channel integration — _stm_bm25 contributes to fused results
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stm_bm25_contributes_source():
    """When STM full-text search returns a matching chunk, stm_text appears in sources."""
    retriever, pg, neo, mg = _make_retriever()

    mg.query_entity_chunks = AsyncMock(return_value=[])
    mg.full_text_search_chunks = AsyncMock(
        return_value=[
            {"chunk_id": "bm25-hit", "text": "violin concert performance", "metadata": {}},
        ]
    )
    pg.search_stm_chunks = AsyncMock(return_value=[])
    pg.search_stm_turns = AsyncMock(return_value=[])
    pg.search_ltm_chunks = AsyncMock(return_value=[])
    pg.search_ltm_turns = AsyncMock(return_value=[])
    pg.get_chunks_by_ids = AsyncMock(return_value={})

    with (
        patch("graphknows.memory.query.retriever.embed", side_effect=_fake_embed),
        patch.object(retriever, "_ltm_entity_traversal", new=AsyncMock(return_value={})),
    ):
        ctx = await retriever.retrieve("violin concert", session_id="s1", scope="stm")

    chunk_ids = {fr["chunk_id"] for fr in ctx.fused_results}
    assert "bm25-hit" in chunk_ids
    hit = next(fr for fr in ctx.fused_results if fr["chunk_id"] == "bm25-hit")
    assert "stm_text" in hit["sources"]


# ---------------------------------------------------------------------------
# retrieve() — signature unchanged
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_retrieve_signature_unchanged():
    """retrieve(query, session_id, top_k, scope) -> RetrievalContext must work."""
    from graphknows.memory.query.context import RetrievalContext

    retriever, pg, neo, mg = _make_retriever()
    mg.query_entity_chunks = AsyncMock(return_value=[])
    mg.full_text_search_chunks = AsyncMock(return_value=[])
    pg.search_stm_chunks = AsyncMock(return_value=[])
    pg.search_stm_turns = AsyncMock(return_value=[])
    pg.search_ltm_chunks = AsyncMock(return_value=[])
    pg.search_ltm_turns = AsyncMock(return_value=[])
    pg.get_chunks_by_ids = AsyncMock(return_value={})

    with (
        patch("graphknows.memory.query.retriever.embed", side_effect=_fake_embed),
        patch.object(retriever, "_ltm_entity_traversal", new=AsyncMock(return_value={})),
    ):
        ctx = await retriever.retrieve(
            query="test query",
            session_id="session-42",
            top_k=5,
            scope="both",
        )

    assert isinstance(ctx, RetrievalContext)
    assert ctx.query == "test query"
    assert ctx.session_id == "session-42"
    assert ctx.search_type == "DET"
    assert len(ctx.fused_results) <= 5

"""Tests for WS-2 additions: BM25 sparse channel and cross-encoder rerank.

Covers:
1. rerank() raises when the cross-encoder model cannot be loaded (no fallback).
2. BM25 fallback scorer ranks an exact-term document above a non-matching document.
3. BM25 channel: stm_bm25 contributes a source entry for matching queries.
4. retrieve() signature remains (query, session_id, top_k, scope) -> RetrievalContext.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from graphknows.ranking.rerank import rerank
from graphknows.retrieval.retriever import DETRetriever, _bm25_rank

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fake_embed(texts, model):
    return np.array([[0.1] * 4 for _ in texts])


def _make_retriever():
    store = MagicMock()
    # Retrieval is fail-loud: every awaited store method a collector touches must
    # be an AsyncMock or the (un-swallowed) await error propagates. Empty defaults;
    # individual tests override the channels they exercise.
    store.query_entity_chunks = AsyncMock(return_value=[])
    store.full_text_search_chunks = AsyncMock(return_value=[])
    store.query_temporal_chunks = AsyncMock(return_value=[])
    store.search_topics = AsyncMock(return_value=[])
    store.search_topics_with_chunks = AsyncMock(return_value=[])
    store.search_chunks_ann = AsyncMock(return_value=[])
    store.get_chunks_by_ids = AsyncMock(return_value={})
    store.chunk_community_weights = AsyncMock(return_value={})
    return DETRetriever(store=store), store


# ---------------------------------------------------------------------------
# rerank() — no fallback: raises when the model cannot be loaded
# ---------------------------------------------------------------------------


def test_rerank_raises_when_model_absent():
    """When the cross-encoder cannot be loaded, rerank() raises — no no-op fallback."""
    candidates = [
        {"chunk_id": "c1", "text": "hello world", "rrf_score": 0.9, "rank": 1},
        {"chunk_id": "c2", "text": "foo bar", "rrf_score": 0.5, "rank": 2},
    ]
    from graphknows.ranking import rerank as rerank_mod

    # Force a load failure and clear any cached model.
    with (
        patch.object(rerank_mod, "_load_model", side_effect=RuntimeError("model unavailable")),
        patch.object(rerank_mod, "_model", None),
        patch.object(rerank_mod, "_model_loaded", False),
        pytest.raises(RuntimeError),
    ):
        rerank("what is hello world", candidates, top_k=2)


def test_rerank_empty_candidates_returns_empty():
    """rerank() with empty candidates returns an empty list without loading a model."""
    result = rerank("any query", [], top_k=5)
    assert result == []


def test_rerank_scores_and_top_k(monkeypatch):
    """rerank() applies the model, sorts by descending score, and honours top_k."""
    candidates = [
        {"chunk_id": f"c{i}", "text": f"text {i}", "rrf_score": float(i), "rank": i + 1}
        for i in range(10)
    ]

    class _FakeCE:
        def predict(self, pairs, show_progress_bar=False):
            # Higher for later candidates so the last one ranks first.
            return np.array([float(i) for i in range(len(pairs))])

    from graphknows.ranking import rerank as rerank_mod

    monkeypatch.setattr(rerank_mod, "_model", _FakeCE())
    monkeypatch.setattr(rerank_mod, "_model_loaded", True)

    result = rerank("query", candidates, top_k=3)
    assert len(result) == 3
    assert result[0]["chunk_id"] == "c9"  # highest logit ranks first
    assert result[0]["rrf_score"] == 9.0  # raw cross-encoder logit, used as-is


def test_rerank_import_does_not_require_torch():
    """Importing from rerank must succeed even without torch installed."""
    # This test passes by virtue of the import at the top of this module.
    # Explicitly reload to simulate a fresh import.
    import importlib

    import graphknows.ranking.rerank as mod

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


def test_bm25_keeps_matches_whose_okapi_score_is_negative():
    """A common query term must not empty the channel.

    Regression: `_bm25_rank` filtered on `score > 0.0`, which is equivalent to
    "contains a query term" only for the fallback scorer, whose IDF
    (log((N+1)/(df+1)) + 1.0) is always positive. BM25Okapi's IDF is
    log((N - df + 0.5)/(df + 0.5)) and goes NEGATIVE once a term appears in more
    than about half the corpus — so every genuine match was discarded and the
    channel returned nothing.

    The bug was unreachable until `rank-bm25` was declared as a dependency:
    undeclared meant uninstalled, so `_HAS_BM25` was always False and only the
    fallback ever ran. Here "session" appears in every document, driving its IDF
    negative under Okapi.
    """
    query_tokens = ["session", "violin"]
    chunks = [
        {"chunk_id": "c_both", "text": "session about the violin recital"},
        {"chunk_id": "c_common", "text": "session about the weather"},
        {"chunk_id": "c_other", "text": "session about lunch plans"},
    ]

    ranked = _bm25_rank(query_tokens, chunks)

    ids = [c["chunk_id"] for _, c in ranked]
    assert ids, "every document contains a query term — the channel must not be empty"
    assert set(ids) == {"c_both", "c_common", "c_other"}
    # The doc matching the discriminative term ranks first.
    assert ids[0] == "c_both"


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

# ---------------------------------------------------------------------------
# STM BM25 channel integration — stm_bm25 contributes to fused results
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_bm25_contributes_source():
    """When full-text search returns a matching chunk, the text source appears."""
    retriever, store = _make_retriever()

    store.full_text_search_chunks = AsyncMock(
        return_value=[
            {"chunk_id": "bm25-hit", "text": "violin concert performance"},
        ]
    )

    with (
        patch("graphknows.retrieval.retriever.embed", side_effect=_fake_embed),
        patch("graphknows.retrieval.retriever._ce_rerank", side_effect=lambda q, c, k: c[:k]),
    ):
        ctx = await retriever.retrieve("violin concert", session_id="s1", scope="stm")

    chunk_ids = {fr["chunk_id"] for fr in ctx.fused_results}
    assert "bm25-hit" in chunk_ids
    hit = next(fr for fr in ctx.fused_results if fr["chunk_id"] == "bm25-hit")
    assert "text" in hit["sources"]


# ---------------------------------------------------------------------------
# retrieve() — signature unchanged
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_retrieve_signature_unchanged():
    """retrieve(query, session_id, top_k, scope) -> RetrievalContext must work."""
    from graphknows.retrieval.models.context import RetrievalContext

    retriever, store = _make_retriever()

    with patch("graphknows.retrieval.retriever.embed", side_effect=_fake_embed):
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

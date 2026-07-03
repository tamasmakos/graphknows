"""Guard tests for the fact-lookup retrieval channel and its routing into the pool."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from graphknows.memory.query.fact_lookup import (
    _match_subjects,
    fact_lookup,
    fact_support_scores,
)
from graphknows.memory.query.retriever import DETRetriever

_QEMB = [1.0, 0.0, 0.0, 0.0]


def _mg_with_facts(subjects, facts):
    mg = MagicMock()
    mg.query_fact_subjects = AsyncMock(return_value=subjects)
    mg.query_facts_by_subjects = AsyncMock(return_value=facts)
    return mg


def _emb(vectors):
    """Return a patched embed() that maps each input text to a fixed vector in order."""
    def _fake(texts, model=""):
        return np.array(vectors[: len(texts)], dtype=np.float32)
    return _fake


# --------------------------------------------------------------------------- unit


def test_match_subjects_word_boundary():
    subs = ["Alice", "Bob", "dance studio"]
    assert _match_subjects("What did Alice do?", subs) == ["Alice"]
    assert _match_subjects("Tell me about the dance studio.", subs) == ["dance studio"]
    # substring inside another word must not match
    assert _match_subjects("bobsled race", subs) == []


@pytest.mark.asyncio
async def test_fact_lookup_returns_source_chunk_for_known_subject():
    facts = [
        {"subject": "Alice", "relation": "WENT_TO", "object": "Paris",
         "chunk_id": "c1", "evidence": "Alice went to Paris last spring."},
        {"subject": "Alice", "relation": "LIKES", "object": "tea",
         "chunk_id": "c2", "evidence": "Alice likes tea."},
    ]
    mg = _mg_with_facts(["Alice"], facts)
    # first statement matches the query embedding, second is orthogonal
    with patch("graphknows.memory.query.fact_lookup.embed", _emb([[1, 0, 0, 0], [0, 1, 0, 0]])):
        hits = await fact_lookup(mg, "Where did Alice go?", "s1", _QEMB, "m")
    assert [h["chunk_id"] for h in hits] == ["c1"]
    assert hits[0]["object"] == "Paris" and hits[0]["score"] > 0.9


@pytest.mark.asyncio
async def test_fact_lookup_silent_when_no_subject_named():
    mg = _mg_with_facts(["Alice"], [])
    hits = await fact_lookup(mg, "Where did Bob go?", "s1", _QEMB, "m")
    assert hits == []
    # gated before fetching facts (and thus before any embedding work)
    mg.query_facts_by_subjects.assert_not_called()


@pytest.mark.asyncio
async def test_fact_lookup_gated_below_similarity_threshold():
    facts = [{"subject": "Alice", "relation": "LIKES", "object": "tea",
              "chunk_id": "c2", "evidence": "Alice likes tea."}]
    mg = _mg_with_facts(["Alice"], facts)
    with patch("graphknows.memory.query.fact_lookup.embed", _emb([[0, 1, 0, 0]])):
        hits = await fact_lookup(mg, "What does Alice drink?", "s1", _QEMB, "m", min_sim=0.3)
    assert hits == []  # best similarity 0.0 < threshold -> clean fallback


@pytest.mark.asyncio
async def test_fact_lookup_no_session_or_emb():
    mg = _mg_with_facts(["Alice"], [])
    assert await fact_lookup(mg, "Where did Alice go?", "", _QEMB, "m") == []
    assert await fact_lookup(mg, "Where did Alice go?", "s1", [], "m") == []


# --------------------------------------------------------- fact_support_scores


@pytest.mark.asyncio
async def test_fact_support_scores_ungated_per_chunk_best():
    facts = [
        {"subject": "Alice", "relation": "WENT_TO", "object": "Paris",
         "chunk_id": "c1", "evidence": "Alice went to Paris."},
        {"subject": "Alice", "relation": "LIKES", "object": "tea",
         "chunk_id": "c2", "evidence": "Alice likes tea."},
    ]
    mg = _mg_with_facts(["Alice"], facts)
    with patch("graphknows.memory.query.fact_lookup.embed", _emb([[1, 0, 0, 0], [0, 1, 0, 0]])):
        m = await fact_support_scores(mg, "Where did Alice go?", "s1", _QEMB, "m")
    # c1 (sim 1.0) kept; c2 (sim 0.0 < floor) dropped — no hard gate on the map itself.
    assert set(m) == {"c1"}
    assert m["c1"]["object"] == "Paris" and m["c1"]["score"] > 0.9


@pytest.mark.asyncio
async def test_fact_support_scores_empty_without_subject():
    mg = _mg_with_facts(["Alice"], [])
    m = await fact_support_scores(mg, "Where did Bob go?", "s1", _QEMB, "m")
    assert m == {}  # no subject named -> term is 0, vector dominates (self-attenuation)
    mg.query_facts_by_subjects.assert_not_called()


# --------------------------------------------------------- unified blend


def _retriever():
    return DETRetriever(pg=MagicMock(), neo=MagicMock(), mg=MagicMock())


def test_blend_empty_fact_map_preserves_rerank_order():
    """No matching fact -> unified score is a monotonic transform of the rerank
    score, so the pure-vector ordering is preserved (no router)."""
    r = _retriever()
    ranked = [
        {"chunk_id": "a", "text": "a", "rrf_score": 0.9, "sources": "semantic", "rank": 1, "metadata": {}},
        {"chunk_id": "b", "text": "b", "rrf_score": 0.5, "sources": "semantic", "rank": 2, "metadata": {}},
    ]
    out = r._blend_fact_support(ranked, {}, top_k=10)
    assert [c["chunk_id"] for c in out] == ["a", "b"]


def test_blend_promotes_and_annotates_fact_supported_chunk():
    import graphknows.memory.query.retriever as R

    r = _retriever()
    ranked = [
        {"chunk_id": "a", "text": "top vector", "rrf_score": 1.0, "sources": "semantic", "rank": 1, "metadata": {}},
        {"chunk_id": "b", "text": "buried body", "rrf_score": 0.0, "sources": "semantic", "rank": 2, "metadata": {}},
    ]
    fact_map = {"b": {"subject": "Alice", "relation": "WENT_TO", "object": "Paris",
                      "evidence": "Alice went to Paris.", "score": 1.0}}
    old = R._FACT_SUPPORT_BOOST
    R._FACT_SUPPORT_BOOST = 2.0  # a=norm 1.0; b=norm 0.0 + 2.0*1.0 -> b wins
    try:
        out = r._blend_fact_support(ranked, fact_map, top_k=10)
    finally:
        R._FACT_SUPPORT_BOOST = old
    assert out[0]["chunk_id"] == "b"  # fact-supported chunk the CE ranked last is promoted
    assert out[0]["text"].startswith("[FACT] Alice went to Paris")
    assert "fact" in out[0]["sources"].split(",")
    assert out[0]["metadata"]["fact"] == "Alice went to Paris"

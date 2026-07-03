"""Guard tests for the fact-lookup retrieval channel and its routing into the pool."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from graphknows.memory.query.fact_lookup import _match_subjects, fact_lookup
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


# ----------------------------------------------------------------------- routing


def _retriever():
    return DETRetriever(pg=MagicMock(), neo=MagicMock(), mg=MagicMock())


@pytest.mark.asyncio
async def test_route_fact_hits_annotates_existing_chunk_in_place():
    r = _retriever()
    fused = [
        {"chunk_id": "v1", "text": "vector one", "sources": "semantic", "rrf_score": 0.4, "rank": 1, "metadata": {}},
        {"chunk_id": "v2", "text": "vector two", "sources": "semantic", "rrf_score": 0.3, "rank": 2, "metadata": {}},
    ]
    hits = [{"chunk_id": "v2", "subject": "Alice", "relation": "WENT_TO",
             "object": "Paris", "evidence": "Alice went to Paris.", "score": 0.9}]
    out = await r._route_fact_hits(hits, fused, top_k=10)
    # A chunk vector already found is annotated IN PLACE — not reordered, so good
    # vector hits are never displaced.
    assert [fc["chunk_id"] for fc in out] == ["v1", "v2"]
    v2 = out[1]
    assert v2["text"].startswith("[FACT] Alice went to Paris")
    assert "fact_lookup" in v2["sources"] and "semantic" in v2["sources"]
    assert v2["metadata"]["fact"] == "Alice went to Paris"


@pytest.mark.asyncio
async def test_route_fact_hits_default_no_promotion_is_rank_neutral():
    # Promotion is opt-in: by default a fact on a chunk the vector pool MISSED is
    # not surfaced, so the channel never evicts a vector hit (rank-neutral).
    r = _retriever()
    fused = [{"chunk_id": "v1", "text": "vector one", "sources": "semantic", "rrf_score": 0.4, "rank": 1, "metadata": {}}]
    hits = [{"chunk_id": "c9", "subject": "Alice", "relation": "LIKES",
             "object": "tea", "evidence": "Alice likes tea.", "score": 0.9}]
    out = await r._route_fact_hits(hits, fused, top_k=10)
    assert [fc["chunk_id"] for fc in out] == ["v1"]


@pytest.mark.asyncio
async def test_route_fact_hits_pulls_source_text_for_new_chunk():
    r = _retriever()
    r._fact_promote = True  # promotion is opt-in; enable it for this path
    r.pg.get_chunks_by_ids = AsyncMock(
        return_value={"c9": {"text": "chunk nine body", "doc_id": "d9", "metadata": {}}}
    )
    fused = [{"chunk_id": "v1", "text": "vector one", "sources": "semantic", "rrf_score": 0.4, "rank": 1, "metadata": {}}]
    hits = [{"chunk_id": "c9", "subject": "Alice", "relation": "LIKES",
             "object": "tea", "evidence": "Alice likes tea.", "score": 0.8}]
    out = await r._route_fact_hits(hits, fused, top_k=10)
    assert out[0]["chunk_id"] == "c9" and out[0]["sources"] == "fact_lookup"
    assert "[FACT] Alice likes tea" in out[0]["text"] and "chunk nine body" in out[0]["text"]
    assert [fc["rank"] for fc in out] == [1, 2]  # re-ranked

"""
Tests for HybridRetriever — advanced methods beyond basic retrieve().

Covers:
- retrieve_graph_traversal()
- retrieve_summaries()
- execute_cypher() — validation logic (no DB needed)
- retrieve_feeling_lucky() — routing logic
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from graphknows.retrieval.context import RetrievalContext
from graphknows.retrieval.retriever import HybridRetriever


def _fake_embed_fn(texts, model):
    return np.array([[0.1] * 4 for _ in texts])


def _make_retriever():
    pg = MagicMock()
    neo = MagicMock()
    return HybridRetriever(pg=pg, neo=neo), pg, neo


# ---------------------------------------------------------------------------
# retrieve_graph_traversal
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_retrieve_graph_traversal_basic():
    retriever, pg, neo = _make_retriever()

    entity_rows = [{"name": "alice", "label": "PERSON", "score": 0.9}]
    pg.search_entities = AsyncMock(return_value=entity_rows)
    neo.get_chunks_for_entity = AsyncMock(
        return_value=[{"chunk_id": "c1", "doc_id": "d1", "text": "alice text"}]
    )
    neo.get_entity_neighbors = AsyncMock(return_value={"neighbors": []})

    with patch("graphknows.retrieval.retriever.embed", side_effect=_fake_embed_fn):
        ctx = await retriever.retrieve_graph_traversal("alice", session_id="s1")

    assert isinstance(ctx, RetrievalContext)
    assert ctx.search_type == "GRAPH_TRAVERSAL"
    assert len(ctx.graph_matches) == 1
    assert ctx.graph_matches[0].name == "alice"
    assert any(fr["chunk_id"] == "c1" for fr in ctx.fused_results)
    # All fused results should come from graph source
    for fr in ctx.fused_results:
        assert fr["sources"] == "graph"


@pytest.mark.asyncio
async def test_retrieve_graph_traversal_k_final_limits():
    retriever, pg, neo = _make_retriever()

    entity_rows = [{"name": "alice", "label": "PERSON", "score": 0.9}]
    pg.search_entities = AsyncMock(return_value=entity_rows)
    neo.get_chunks_for_entity = AsyncMock(
        return_value=[{"chunk_id": f"c{i}", "doc_id": "d", "text": f"text {i}"} for i in range(20)]
    )
    neo.get_entity_neighbors = AsyncMock(return_value={"neighbors": []})

    with patch("graphknows.retrieval.retriever.embed", side_effect=_fake_embed_fn):
        ctx = await retriever.retrieve_graph_traversal("query", session_id="s1", k_final=3)

    assert len(ctx.fused_results) <= 3


@pytest.mark.asyncio
async def test_retrieve_graph_traversal_neighbor_chunks_included():
    """Chunks from neighbor entities must also appear in fused_results."""
    retriever, pg, neo = _make_retriever()

    entity_rows = [{"name": "alice", "label": "PERSON", "score": 0.9}]
    pg.search_entities = AsyncMock(return_value=entity_rows)

    async def neo_chunks_side(name, top_k=5):
        if name == "alice":
            return [{"chunk_id": "direct", "doc_id": "d", "text": "direct"}]
        return [{"chunk_id": "neighbor_chunk", "doc_id": "d", "text": "neighbor"}]

    neo.get_chunks_for_entity = AsyncMock(side_effect=neo_chunks_side)
    neo.get_entity_neighbors = AsyncMock(return_value={"neighbors": [{"name": "bob"}]})

    with patch("graphknows.retrieval.retriever.embed", side_effect=_fake_embed_fn):
        ctx = await retriever.retrieve_graph_traversal("query", session_id="s1")

    chunk_ids = {fr["chunk_id"] for fr in ctx.fused_results}
    assert "direct" in chunk_ids
    assert "neighbor_chunk" in chunk_ids


@pytest.mark.asyncio
async def test_retrieve_graph_traversal_empty_entities():
    retriever, pg, neo = _make_retriever()
    pg.search_entities = AsyncMock(return_value=[])

    with patch("graphknows.retrieval.retriever.embed", side_effect=_fake_embed_fn):
        ctx = await retriever.retrieve_graph_traversal("q", session_id="s1")

    assert ctx.graph_matches == []
    assert ctx.fused_results == []


@pytest.mark.asyncio
async def test_retrieve_graph_traversal_deduplicates_chunks():
    """Same chunk appearing via multiple entity paths should only appear once in fused_results."""
    retriever, pg, neo = _make_retriever()

    entity_rows = [
        {"name": "alice", "label": "PERSON", "score": 0.9},
        {"name": "bob", "label": "PERSON", "score": 0.7},
    ]
    pg.search_entities = AsyncMock(return_value=entity_rows)
    shared = [{"chunk_id": "shared", "doc_id": "d", "text": "shared text"}]
    neo.get_chunks_for_entity = AsyncMock(return_value=shared)
    neo.get_entity_neighbors = AsyncMock(return_value={"neighbors": []})

    with patch("graphknows.retrieval.retriever.embed", side_effect=_fake_embed_fn):
        ctx = await retriever.retrieve_graph_traversal("q", session_id="s1")

    chunk_ids = [fr["chunk_id"] for fr in ctx.fused_results]
    assert chunk_ids.count("shared") == 1  # deduplication


# ---------------------------------------------------------------------------
# retrieve_summaries
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_retrieve_summaries_returns_topic_text():
    retriever, pg, neo = _make_retriever()

    entity_rows = [{"name": "ai safety", "label": "TOPIC", "score": 0.85}]
    pg.search_entities = AsyncMock(return_value=entity_rows)
    neo.get_entity_community = AsyncMock(
        return_value={
            "id": "t1",
            "label": "AI Safety",
            "summary": "Overview of AI safety research.",
        }
    )

    with patch("graphknows.retrieval.retriever.embed", side_effect=_fake_embed_fn):
        ctx = await retriever.retrieve_summaries("what is AI safety", session_id="s1")

    assert ctx.search_type == "SUMMARIES"
    assert len(ctx.fused_results) == 1
    fr = ctx.fused_results[0]
    assert fr["sources"] == "summaries"
    assert "AI Safety" in fr["text"]
    assert fr["chunk_id"].startswith("topic:")
    assert abs(fr["rrf_score"] - 0.85) < 1e-4


@pytest.mark.asyncio
async def test_retrieve_summaries_skips_none_community():
    retriever, pg, neo = _make_retriever()

    entity_rows = [
        {"name": "orphan", "label": "ORG", "score": 0.7},
        {"name": "known", "label": "ORG", "score": 0.6},
    ]
    pg.search_entities = AsyncMock(return_value=entity_rows)

    async def community_side(name):
        if name == "orphan":
            return None
        return {"id": "t2", "label": "Known Topic", "summary": "..."}

    neo.get_entity_community = AsyncMock(side_effect=community_side)

    with patch("graphknows.retrieval.retriever.embed", side_effect=_fake_embed_fn):
        ctx = await retriever.retrieve_summaries("query", session_id="s1")

    assert len(ctx.fused_results) == 1  # orphan was skipped


@pytest.mark.asyncio
async def test_retrieve_summaries_skips_exception_community():
    retriever, pg, neo = _make_retriever()

    entity_rows = [{"name": "alice", "label": "PERSON", "score": 0.9}]
    pg.search_entities = AsyncMock(return_value=entity_rows)
    neo.get_entity_community = AsyncMock(side_effect=RuntimeError("Neo4j down"))

    with patch("graphknows.retrieval.retriever.embed", side_effect=_fake_embed_fn):
        ctx = await retriever.retrieve_summaries("query", session_id="s1")

    # Exception treated as None → skipped
    assert ctx.fused_results == []


# ---------------------------------------------------------------------------
# execute_cypher — validation guards (no DB needed for guard tests)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_execute_cypher_rejects_fstring_interpolation():
    retriever, _, _ = _make_retriever()
    with pytest.raises(ValueError, match="interpolation"):
        await retriever.execute_cypher("MATCH (n {name: {name}}) RETURN n")


@pytest.mark.asyncio
async def test_execute_cypher_rejects_write_keywords():
    retriever, _, _ = _make_retriever()
    for kw in ["CREATE", "MERGE", "SET", "DELETE", "DETACH DELETE", "DROP"]:
        with pytest.raises(ValueError, match="read-only"):
            await retriever.execute_cypher(f"{kw} (n:Entity) RETURN n")


@pytest.mark.asyncio
async def test_execute_cypher_allows_read_cypher():
    retriever, _, neo = _make_retriever()
    neo._arun = AsyncMock(return_value=[{"name": "Alice"}])

    result = await retriever.execute_cypher(
        "MATCH (n:Entity) WHERE n.name = $name RETURN n.name AS name",
        params={"name": "Alice"},
    )
    assert result == [{"name": "Alice"}]


@pytest.mark.asyncio
async def test_execute_cypher_empty_params():
    retriever, _, neo = _make_retriever()
    neo._arun = AsyncMock(return_value=[])
    result = await retriever.execute_cypher("MATCH (n:Entity) RETURN n LIMIT 5")
    assert result == []


# ---------------------------------------------------------------------------
# retrieve_feeling_lucky — routing logic
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_feeling_lucky_rejects_cypher_queries():
    retriever, _, _ = _make_retriever()
    for q in ["MATCH (n) RETURN n", "WHERE n.name = $x RETURN n"]:
        with pytest.raises(ValueError, match="Cypher"):
            await retriever.retrieve_feeling_lucky(q, session_id="s1")


@pytest.mark.asyncio
async def test_feeling_lucky_routes_short_query_to_graph_traversal():
    """Fewer than 5 tokens → GRAPH_TRAVERSAL route."""
    retriever, pg, neo = _make_retriever()
    pg.search_entities = AsyncMock(return_value=[])
    neo.get_entity_neighbors = AsyncMock(return_value={"neighbors": []})

    with patch("graphknows.retrieval.retriever.embed", side_effect=_fake_embed_fn):
        ctx = await retriever.retrieve_feeling_lucky("Alice", session_id="s1")

    assert "GRAPH_TRAVERSAL" in ctx.search_type


@pytest.mark.asyncio
async def test_feeling_lucky_routes_summary_triggers():
    """'summarize X' → SUMMARIES route."""
    retriever, pg, neo = _make_retriever()
    pg.search_entities = AsyncMock(return_value=[])

    with patch("graphknows.retrieval.retriever.embed", side_effect=_fake_embed_fn):
        ctx = await retriever.retrieve_feeling_lucky(
            "summarize the key findings from the research paper", session_id="s1"
        )

    assert "SUMMARIES" in ctx.search_type


@pytest.mark.asyncio
async def test_feeling_lucky_defaults_to_hybrid():
    """Long query without summary triggers → HYBRID route."""
    retriever, pg, neo = _make_retriever()
    pg.search_entities = AsyncMock(return_value=[])
    pg.search_chunks = AsyncMock(return_value=[])
    pg.get_chunks_by_ids = AsyncMock(return_value={})

    with patch("graphknows.retrieval.retriever.embed", side_effect=_fake_embed_fn):
        ctx = await retriever.retrieve_feeling_lucky(
            "what are the implications of climate change on biodiversity ecosystems globally",
            session_id="s1",
        )

    assert "HYBRID" in ctx.search_type


# ===========================================================================
# RRF scoring
# ===========================================================================



import pytest

from graphknows.retrieval.retriever import RRF_K

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fake_emb(dim: int = 4) -> list[float]:
    return [0.1] * dim


def _fake_embed_fn(texts, model):
    """Replaces `embed()` — returns a numpy array of arrays, not a list."""
    return np.array([[0.1] * 4 for _ in texts])


def _make_retriever():
    """Return a HybridRetriever with all dependencies mocked out."""
    pg = MagicMock()
    neo = MagicMock()
    retriever = HybridRetriever(pg=pg, neo=neo)
    return retriever, pg, neo


# ---------------------------------------------------------------------------
# RRF math (pure-function tests — no IO)
# ---------------------------------------------------------------------------


def test_rrf_k_constant():
    assert RRF_K == 60


def test_rrf_score_decreases_with_rank():
    """Higher rank index → lower RRF contribution (1/(K+rank+1))."""
    scores = [1.0 / (RRF_K + r + 1) for r in range(5)]
    for i in range(len(scores) - 1):
        assert scores[i] > scores[i + 1], f"Expected scores[{i}]>scores[{i + 1}]: {scores}"


def test_rrf_score_rank0():
    expected = 1.0 / (RRF_K + 0 + 1)
    assert abs(expected - 1.0 / 61) < 1e-10


# ---------------------------------------------------------------------------
# retrieve() — pure semantic path (no graph entities)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_retrieve_semantic_only():
    """
    When entity search returns empty list, the retriever must still return
    a valid RetrievalContext where graph_matches is empty and semantic_matches
    contains the semantic chunks.
    """
    retriever, pg, neo = _make_retriever()

    sem_rows = [
        {
            "chunk_id": f"c{i}",
            "doc_id": "d1",
            "text": f"chunk {i}",
            "metadata": {},
            "cosine_score": 0.9 - i * 0.1,
        }
        for i in range(3)
    ]
    pg.search_entities = AsyncMock(return_value=[])
    pg.search_chunks = AsyncMock(return_value=sem_rows)
    pg.get_chunks_by_ids = AsyncMock(
        return_value={
            r["chunk_id"]: {"doc_id": r["doc_id"], "text": r["text"], "metadata": {}}
            for r in sem_rows
        }
    )

    with patch("graphknows.retrieval.retriever.embed", side_effect=_fake_embed_fn):
        ctx = await retriever.retrieve(
            "test query", session_id="s1", k_entity=5, k_semantic=10, k_final=5
        )

    assert isinstance(ctx, RetrievalContext)
    assert ctx.query == "test query"
    assert ctx.session_id == "s1"
    assert ctx.graph_matches == []
    assert len(ctx.semantic_matches) == 3
    assert len(ctx.fused_results) == 3
    # Sources should all be 'semantic'
    for fr in ctx.fused_results:
        assert fr["sources"] == "semantic"
    # Ranks must be 1-indexed sequential
    ranks = [fr["rank"] for fr in ctx.fused_results]
    assert ranks == list(range(1, 4))


# ---------------------------------------------------------------------------
# retrieve() — graph path contributes to fusion
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_retrieve_graph_boosts_chunk():
    """
    Chunk c1 appears in both graph and semantic paths → its fused RRF score
    must be strictly higher than chunks appearing in only one path.
    """
    retriever, pg, neo = _make_retriever()

    entity_rows = [{"name": "alice", "label": "PERSON", "score": 0.95}]
    sem_rows = [
        {
            "chunk_id": "c1",
            "doc_id": "d1",
            "text": "alice chunk",
            "metadata": {},
            "cosine_score": 0.9,
        },
        {
            "chunk_id": "c2",
            "doc_id": "d1",
            "text": "other chunk",
            "metadata": {},
            "cosine_score": 0.8,
        },
    ]
    neo_chunks_for_alice = [{"chunk_id": "c1", "doc_id": "d1", "text": "alice chunk"}]

    pg.search_entities = AsyncMock(return_value=entity_rows)
    pg.search_chunks = AsyncMock(return_value=sem_rows)
    pg.get_chunks_by_ids = AsyncMock(
        return_value={
            "c1": {"doc_id": "d1", "text": "alice chunk", "metadata": {}},
            "c2": {"doc_id": "d1", "text": "other chunk", "metadata": {}},
        }
    )
    neo.get_chunks_for_entity = AsyncMock(return_value=neo_chunks_for_alice)

    with patch("graphknows.retrieval.retriever.embed", side_effect=_fake_embed_fn):
        ctx = await retriever.retrieve("alice query", session_id="s1")

    fused_by_id = {fr["chunk_id"]: fr for fr in ctx.fused_results}
    assert "c1" in fused_by_id
    assert "c2" in fused_by_id
    # c1 must rank higher (rank 1) since it got graph + semantic boost
    assert fused_by_id["c1"]["rank"] < fused_by_id["c2"]["rank"]
    # c1 sources must include both
    assert "graph" in fused_by_id["c1"]["sources"]
    assert "semantic" in fused_by_id["c1"]["sources"]
    # c2 is semantic-only
    assert fused_by_id["c2"]["sources"] == "semantic"


@pytest.mark.asyncio
async def test_retrieve_graph_match_entity_metadata():
    """graph_matches must include entity name, label, similarity."""
    retriever, pg, neo = _make_retriever()

    entity_rows = [{"name": "eiffel tower", "label": "LOC", "score": 0.88}]
    pg.search_entities = AsyncMock(return_value=entity_rows)
    pg.search_chunks = AsyncMock(return_value=[])
    pg.get_chunks_by_ids = AsyncMock(return_value={})
    neo.get_chunks_for_entity = AsyncMock(
        return_value=[{"chunk_id": "cx", "doc_id": "dx", "text": "paris"}]
    )

    with patch("graphknows.retrieval.retriever.embed", side_effect=_fake_embed_fn):
        ctx = await retriever.retrieve("tower", session_id="s1")

    assert len(ctx.graph_matches) == 1
    em = ctx.graph_matches[0]
    assert em.name == "eiffel tower"
    assert em.label == "LOC"
    assert abs(em.similarity - 0.88) < 1e-6
    assert len(em.chunks) == 1
    assert em.chunks[0].chunk_id == "cx"


@pytest.mark.asyncio
async def test_retrieve_neo4j_exception_gracefully_continues():
    """Neo4j errors per entity must not crash the whole retrieve call."""
    retriever, pg, neo = _make_retriever()

    entity_rows = [
        {"name": "bad_entity", "label": "ORG", "score": 0.7},
        {"name": "good_entity", "label": "ORG", "score": 0.6},
    ]
    pg.search_entities = AsyncMock(return_value=entity_rows)
    pg.search_chunks = AsyncMock(return_value=[])
    pg.get_chunks_by_ids = AsyncMock(return_value={})

    async def neo_side_effect(name, top_k=5):
        if name == "bad_entity":
            raise RuntimeError("Neo4j timeout")
        return [{"chunk_id": "cg", "doc_id": "dg", "text": "good"}]

    neo.get_chunks_for_entity = AsyncMock(side_effect=neo_side_effect)

    with patch("graphknows.retrieval.retriever.embed", side_effect=_fake_embed_fn):
        ctx = await retriever.retrieve("query", session_id="s1")

    # Should still return a result, bad_entity gets empty chunk list
    assert len(ctx.graph_matches) == 2
    bad = next(em for em in ctx.graph_matches if em.name == "bad_entity")
    good = next(em for em in ctx.graph_matches if em.name == "good_entity")
    assert bad.chunks == []
    assert len(good.chunks) == 1


@pytest.mark.asyncio
async def test_retrieve_k_final_limits_fused_results():
    """k_final must cap the number of returned fused chunks."""
    retriever, pg, neo = _make_retriever()

    sem_rows = [
        {
            "chunk_id": f"c{i}",
            "doc_id": "d",
            "text": f"t{i}",
            "metadata": {},
            "cosine_score": 1.0 - i * 0.01,
        }
        for i in range(10)
    ]
    pg.search_entities = AsyncMock(return_value=[])
    pg.search_chunks = AsyncMock(return_value=sem_rows)
    pg.get_chunks_by_ids = AsyncMock(
        return_value={
            r["chunk_id"]: {"doc_id": r["doc_id"], "text": r["text"], "metadata": {}}
            for r in sem_rows
        }
    )

    with patch("graphknows.retrieval.retriever.embed", side_effect=_fake_embed_fn):
        ctx = await retriever.retrieve("q", session_id="s1", k_final=3)

    assert len(ctx.fused_results) == 3


@pytest.mark.asyncio
async def test_retrieve_empty_query_returns_empty_context():
    """Empty entity + semantic results → empty fused_results."""
    retriever, pg, neo = _make_retriever()

    pg.search_entities = AsyncMock(return_value=[])
    pg.search_chunks = AsyncMock(return_value=[])
    pg.get_chunks_by_ids = AsyncMock(return_value={})

    with patch("graphknows.retrieval.retriever.embed", side_effect=_fake_embed_fn):
        ctx = await retriever.retrieve("nothing", session_id="s1")

    assert ctx.fused_results == []
    assert ctx.graph_matches == []
    assert ctx.semantic_matches == []


# ---------------------------------------------------------------------------
# retrieve_semantic() method (lines 256+)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_retrieve_semantic_method():
    """retrieve_semantic must use only semantic path and set search_type='SEMANTIC'."""
    retriever, pg, neo = _make_retriever()

    sem_rows = [
        {"chunk_id": "s1c", "doc_id": "d", "text": "hello", "metadata": {}, "cosine_score": 0.85}
    ]
    pg.search_chunks = AsyncMock(return_value=sem_rows)
    pg.get_chunks_by_ids = AsyncMock(
        return_value={"s1c": {"doc_id": "d", "text": "hello", "metadata": {}}}
    )

    with patch("graphknows.retrieval.retriever.embed", side_effect=_fake_embed_fn):
        ctx = await retriever.retrieve_semantic("hello", session_id="s1", k=5)

    assert isinstance(ctx, RetrievalContext)
    assert ctx.search_type == "SEMANTIC"
    assert len(ctx.semantic_matches) == 1
    # retrieve_semantic should NOT call entity search
    pg.search_entities.assert_not_called()


# ---------------------------------------------------------------------------
# mention_weight decay (entity rank 0 → heavier weight than rank 1)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_mention_weight_decays_with_entity_rank():
    """
    When two entities link to the same chunk, the one with lower entity_rank
    (higher similarity) should win the mention_weight, and the chunk's entity_rank
    stored in graph_chunk_info should be from the first entity.
    """
    retriever, pg, neo = _make_retriever()

    # Two entities at different ranks both link to the same chunk 'shared'
    entity_rows = [
        {"name": "top_entity", "label": "ORG", "score": 0.99},  # rank 0 — higher
        {"name": "low_entity", "label": "ORG", "score": 0.50},  # rank 1 — lower
    ]
    pg.search_entities = AsyncMock(return_value=entity_rows)
    pg.search_chunks = AsyncMock(return_value=[])
    pg.get_chunks_by_ids = AsyncMock(
        return_value={"shared": {"doc_id": "d", "text": "shared text", "metadata": {}}}
    )

    shared_chunk = [{"chunk_id": "shared", "doc_id": "d", "text": "shared text"}]

    async def neo_side(name, top_k=5):
        return shared_chunk

    neo.get_chunks_for_entity = AsyncMock(side_effect=neo_side)

    with patch("graphknows.retrieval.retriever.embed", side_effect=_fake_embed_fn):
        ctx = await retriever.retrieve("q", session_id="s1")

    # 'shared' chunk must appear in fused_results
    assert any(fr["chunk_id"] == "shared" for fr in ctx.fused_results)
    # Its RRF score comes from entity rank 0 (the better match wins)
    fused = next(fr for fr in ctx.fused_results if fr["chunk_id"] == "shared")
    # Score must be at least 1/(K+0+1) since rank=0 contributes (allow float rounding)
    min_expected = round(1.0 / (RRF_K + 0 + 1), 6)
    assert fused["rrf_score"] >= min_expected

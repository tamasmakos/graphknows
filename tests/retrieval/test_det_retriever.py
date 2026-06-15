"""Tests for DETRetriever — all 4 sources, RRF fusion, and graceful degradation."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from graphknows.memory.query._rrf_common import RRF_K
from graphknows.memory.query.retriever import DETRetriever


def _fake_embed(texts, model):
    return np.array([[0.1] * 4 for _ in texts])


def _make_retriever():
    pg = MagicMock()
    neo = MagicMock()
    mg = MagicMock()
    retriever = DETRetriever(pg=pg, neo=neo, mg=mg)
    return retriever, pg, neo, mg


# ---------------------------------------------------------------------------
# Test 1: all 4 sources contribute to fused_results
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_all_four_sources_contribute():
    retriever, pg, neo, mg = _make_retriever()

    # Source A — STM entity boost (Memgraph)
    mg.query_entity_chunks = AsyncMock(return_value=[
        {"chunk_id": "stm-c1", "text": "stm entity chunk", "entity_names": ["alice"],
         "file_id": "", "metadata": {}}
    ])

    # Source B — Topic recall (Neo4j TOPIC, via to_thread)
    topic_rows = [{"topic_id": "t1", "label": "AI Policy", "summary": "alice policy topic",
                   "entities": ["alice"], "score": 1}]

    # Source C — LTM entity traversal (Neo4j, via to_thread)
    ltm_rows = [{"chunk_id": "ltm-c1", "text": "ltm entity chunk", "entities": ["alice"]}]

    # Source D — semantic ANN
    pg.search_stm_chunks = AsyncMock(return_value=[
        {"id": "sem-c1", "text": "semantic chunk", "cosine_score": 0.9,
         "file_id": "", "heading_path": "", "metadata": {}}
    ])
    pg.search_stm_turns = AsyncMock(return_value=[])
    pg.search_ltm_chunks = AsyncMock(return_value=[])
    pg.search_ltm_turns = AsyncMock(return_value=[])
    pg.get_chunks_by_ids = AsyncMock(return_value={})

    def _topic_cypher(words, top_k):
        return topic_rows

    def _ltm_traversal(words, session_filter, top_k):
        return {r["chunk_id"]: (1.0 / (RRF_K + i + 1), {
            "text": r["text"], "entities": r["entities"],
            "sources": {"ltm_graph"}, "doc_id": "", "metadata": {},
        }) for i, r in enumerate(ltm_rows)}

    with (
        patch("graphknows.memory.query.retriever.embed", side_effect=_fake_embed),
        patch.object(retriever, "_run_topic_cypher", side_effect=_topic_cypher),
        patch.object(retriever, "_run_entity_traversal_sync", side_effect=_ltm_traversal),
    ):
        ctx = await retriever.retrieve("alice policy", session_id="s1", scope="both")

    chunk_ids = {fr["chunk_id"] for fr in ctx.fused_results}
    assert "stm-c1" in chunk_ids, "STM entity boost chunk missing"
    assert "topic:t1" in chunk_ids, "Topic recall result missing"
    assert "ltm-c1" in chunk_ids, "LTM entity traversal chunk missing"
    assert "sem-c1" in chunk_ids, "Semantic ANN chunk missing"


# ---------------------------------------------------------------------------
# Test 2: chunk in multiple sources gets higher RRF than single-source chunk
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_multi_source_chunk_ranks_higher():
    retriever, pg, neo, mg = _make_retriever()

    # "shared" chunk appears in both STM and semantic sources
    mg.query_entity_chunks = AsyncMock(return_value=[
        {"chunk_id": "shared", "text": "shared text", "entity_names": ["alice"],
         "file_id": "", "metadata": {}}
    ])
    pg.search_stm_chunks = AsyncMock(return_value=[
        {"id": "shared", "text": "shared text", "cosine_score": 0.9,
         "file_id": "", "heading_path": "", "metadata": {}},
        {"id": "solo", "text": "solo text", "cosine_score": 0.8,
         "file_id": "", "heading_path": "", "metadata": {}},
    ])
    pg.search_stm_turns = AsyncMock(return_value=[])
    pg.search_ltm_chunks = AsyncMock(return_value=[])
    pg.search_ltm_turns = AsyncMock(return_value=[])
    pg.get_chunks_by_ids = AsyncMock(return_value={})

    with (
        patch("graphknows.memory.query.retriever.embed", side_effect=_fake_embed),
        patch.object(retriever, "_run_topic_cypher", return_value=[]),
        patch.object(retriever, "_run_entity_traversal_sync", return_value={}),
    ):
        ctx = await retriever.retrieve("alice", session_id="s1", scope="stm")

    by_id = {fr["chunk_id"]: fr for fr in ctx.fused_results}
    assert by_id["shared"]["rrf_score"] > by_id["solo"]["rrf_score"], (
        "shared chunk (stm + semantic) should outscore solo (semantic only)"
    )


# ---------------------------------------------------------------------------
# Test 3: topic hits appear with synthetic "topic:..." chunk_id
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_topic_hits_have_synthetic_chunk_id():
    retriever, pg, neo, mg = _make_retriever()

    mg.query_entity_chunks = AsyncMock(return_value=[])
    pg.search_stm_chunks = AsyncMock(return_value=[])
    pg.search_stm_turns = AsyncMock(return_value=[])
    pg.search_ltm_chunks = AsyncMock(return_value=[])
    pg.search_ltm_turns = AsyncMock(return_value=[])
    pg.get_chunks_by_ids = AsyncMock(return_value={})

    topic_rows = [
        {"topic_id": "t42", "label": "Climate Policy", "summary": "climate energy policy",
         "entities": ["EU", "Paris"], "score": 2},
    ]

    with (
        patch("graphknows.memory.query.retriever.embed", side_effect=_fake_embed),
        patch.object(retriever, "_run_topic_cypher", return_value=topic_rows),
        patch.object(retriever, "_run_entity_traversal_sync", return_value={}),
    ):
        ctx = await retriever.retrieve("climate policy", session_id="s1")

    topic_fused = [fr for fr in ctx.fused_results if fr["chunk_id"].startswith("topic:")]
    assert len(topic_fused) == 1
    assert topic_fused[0]["chunk_id"] == "topic:t42"


# ---------------------------------------------------------------------------
# Test 4: top_k caps fused_results
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_top_k_is_respected():
    retriever, pg, neo, mg = _make_retriever()

    mg.query_entity_chunks = AsyncMock(return_value=[
        {"chunk_id": f"e{i}", "text": f"entity {i}", "entity_names": [],
         "file_id": "", "metadata": {}}
        for i in range(8)
    ])
    pg.search_stm_chunks = AsyncMock(return_value=[])
    pg.search_stm_turns = AsyncMock(return_value=[])
    pg.search_ltm_chunks = AsyncMock(return_value=[])
    pg.search_ltm_turns = AsyncMock(return_value=[])
    pg.get_chunks_by_ids = AsyncMock(return_value={})

    with (
        patch("graphknows.memory.query.retriever.embed", side_effect=_fake_embed),
        patch.object(retriever, "_run_topic_cypher", return_value=[]),
        patch.object(retriever, "_run_entity_traversal_sync", return_value={}),
    ):
        ctx = await retriever.retrieve("query words here", session_id="s1", top_k=3)

    assert len(ctx.fused_results) <= 3


# ---------------------------------------------------------------------------
# Test 5: Memgraph failure degrades gracefully
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_memgraph_failure_degrades_gracefully():
    retriever, pg, neo, mg = _make_retriever()

    mg.query_entity_chunks = AsyncMock(side_effect=RuntimeError("memgraph down"))
    pg.search_stm_chunks = AsyncMock(return_value=[
        {"id": "sem-ok", "text": "semantic chunk", "cosine_score": 0.85,
         "file_id": "", "heading_path": "", "metadata": {}}
    ])
    pg.search_stm_turns = AsyncMock(return_value=[])
    pg.search_ltm_chunks = AsyncMock(return_value=[])
    pg.search_ltm_turns = AsyncMock(return_value=[])
    pg.get_chunks_by_ids = AsyncMock(return_value={})

    with (
        patch("graphknows.memory.query.retriever.embed", side_effect=_fake_embed),
        patch.object(retriever, "_run_topic_cypher", return_value=[]),
        patch.object(retriever, "_run_entity_traversal_sync", return_value={}),
    ):
        ctx = await retriever.retrieve("query text here", session_id="s1")

    assert ctx.search_type == "DET"
    chunk_ids = {fr["chunk_id"] for fr in ctx.fused_results}
    assert "sem-ok" in chunk_ids, "Semantic source should still return results when Memgraph fails"

"""Tests for DETRetriever: collectors, RRF fusion, and fail-loud error propagation."""

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
    # Retrieval is fail-loud: every awaited store method a collector touches must
    # be an AsyncMock, or the (now un-swallowed) await error propagates. Defaults
    # return empty; individual tests override the channels they exercise.
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
    retriever = DETRetriever(pg=pg, neo=neo, mg=mg)
    return retriever, pg, neo, mg


@pytest.mark.asyncio
async def test_all_three_collectors_contribute():
    retriever, pg, neo, mg = _make_retriever()

    mg.query_entity_chunks = AsyncMock(
        return_value=[
            {
                "chunk_id": "stm-c1",
                "text": "stm entity chunk",
                "entity_names": ["alice"],
                "file_id": "",
                "metadata": {},
            }
        ]
    )

    ltm_rows = [{"chunk_id": "ltm-c1", "text": "ltm entity chunk", "entities": ["alice"]}]

    pg.search_stm_chunks = AsyncMock(
        return_value=[
            {
                "id": "sem-c1",
                "text": "semantic chunk",
                "cosine_score": 0.9,
                "file_id": "",
                "heading_path": "",
                "metadata": {},
            }
        ]
    )
    pg.search_stm_turns = AsyncMock(return_value=[])
    pg.search_ltm_chunks = AsyncMock(return_value=[])
    pg.search_ltm_turns = AsyncMock(return_value=[])
    pg.get_chunks_by_ids = AsyncMock(return_value={})

    async def _ltm_traversal(words, session_id, top_k):
        return {
            r["chunk_id"]: (
                1.0 / (RRF_K + i + 1),
                {
                    "text": r["text"],
                    "entities": r["entities"],
                    "sources": {"ltm_graph"},
                    "doc_id": "",
                    "metadata": {},
                },
            )
            for i, r in enumerate(ltm_rows)
        }

    with (
        patch("graphknows.memory.query.retriever.embed", side_effect=_fake_embed),
        patch.object(retriever, "_ltm_entity_traversal", side_effect=_ltm_traversal),
    ):
        ctx = await retriever.retrieve("alice policy", session_id="s1", scope="both")

    chunk_ids = {fr["chunk_id"] for fr in ctx.fused_results}
    assert "stm-c1" in chunk_ids
    assert "ltm-c1" in chunk_ids
    assert "sem-c1" in chunk_ids
    assert not any(cid.startswith("topic:") for cid in chunk_ids)


@pytest.mark.asyncio
async def test_multi_source_chunk_ranks_higher():
    retriever, pg, neo, mg = _make_retriever()

    mg.query_entity_chunks = AsyncMock(
        return_value=[
            {
                "chunk_id": "shared",
                "text": "shared text",
                "entity_names": ["alice"],
                "file_id": "",
                "metadata": {},
            }
        ]
    )
    pg.search_stm_chunks = AsyncMock(
        return_value=[
            {
                "id": "shared",
                "text": "shared text",
                "cosine_score": 0.9,
                "file_id": "",
                "heading_path": "",
                "metadata": {},
            },
            {
                "id": "solo",
                "text": "solo text",
                "cosine_score": 0.8,
                "file_id": "",
                "heading_path": "",
                "metadata": {},
            },
        ]
    )
    pg.search_stm_turns = AsyncMock(return_value=[])
    pg.search_ltm_chunks = AsyncMock(return_value=[])
    pg.search_ltm_turns = AsyncMock(return_value=[])
    pg.get_chunks_by_ids = AsyncMock(return_value={})

    with (
        patch("graphknows.memory.query.retriever.embed", side_effect=_fake_embed),
        patch.object(retriever, "_ltm_entity_traversal", new=AsyncMock(return_value={})),
        # Test fusion/blend order deterministically, independent of whether the
        # cross-encoder model happens to be cached in this environment.
        patch("graphknows.memory.query.retriever._ce_rerank", side_effect=lambda q, c, k: c[:k]),
    ):
        ctx = await retriever.retrieve("alice", session_id="s1", scope="stm")

    by_id = {fr["chunk_id"]: fr for fr in ctx.fused_results}
    assert by_id["shared"]["rrf_score"] > by_id["solo"]["rrf_score"]


@pytest.mark.asyncio
async def test_community_weight_boosts_semantic_hit_without_topic_chunk():
    retriever, pg, neo, mg = _make_retriever()

    mg.query_entity_chunks = AsyncMock(return_value=[])
    pg.search_stm_chunks = AsyncMock(
        return_value=[
            {
                "id": "community-hit",
                "text": "climate energy policy",
                "cosine_score": 0.8,
                "file_id": "",
                "heading_path": "",
                "metadata": {"community_id": 7, "community_weight": 3.0},
            },
            {
                "id": "plain-hit",
                "text": "climate policy",
                "cosine_score": 0.9,
                "file_id": "",
                "heading_path": "",
                "metadata": {},
            },
        ]
    )
    pg.search_stm_turns = AsyncMock(return_value=[])
    pg.search_ltm_chunks = AsyncMock(return_value=[])
    pg.search_ltm_turns = AsyncMock(return_value=[])
    pg.get_chunks_by_ids = AsyncMock(return_value={})

    with (
        patch("graphknows.memory.query.retriever.embed", side_effect=_fake_embed),
        patch.object(retriever, "_ltm_entity_traversal", new=AsyncMock(return_value={})),
        patch("graphknows.memory.query.retriever._ce_rerank", side_effect=lambda q, c, k: c[:k]),
    ):
        ctx = await retriever.retrieve("climate policy", session_id="s1", scope="stm")

    assert ctx.fused_results[0]["chunk_id"] == "community-hit"
    assert not any(fr["chunk_id"].startswith("topic:") for fr in ctx.fused_results)


@pytest.mark.asyncio
async def test_top_k_is_respected():
    retriever, pg, neo, mg = _make_retriever()

    mg.query_entity_chunks = AsyncMock(
        return_value=[
            {"chunk_id": f"e{i}", "text": f"entity {i}", "entity_names": [], "file_id": ""}
            for i in range(8)
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
        ctx = await retriever.retrieve("query words here", session_id="s1", top_k=3)

    assert len(ctx.fused_results) <= 3


@pytest.mark.asyncio
async def test_collector_failure_raises_loudly():
    """A collector failure must propagate — no silent graceful degradation. A
    broken channel should surface immediately, not masquerade as thin recall."""
    retriever, pg, neo, mg = _make_retriever()

    mg.query_entity_chunks = AsyncMock(side_effect=RuntimeError("stm down"))
    pg.search_stm_chunks = AsyncMock(return_value=[])
    pg.search_stm_turns = AsyncMock(return_value=[])
    pg.search_ltm_chunks = AsyncMock(return_value=[])
    pg.search_ltm_turns = AsyncMock(return_value=[])
    pg.get_chunks_by_ids = AsyncMock(return_value={})

    with (
        patch("graphknows.memory.query.retriever.embed", side_effect=_fake_embed),
        patch.object(retriever, "_ltm_entity_traversal", new=AsyncMock(return_value={})),
        pytest.raises(RuntimeError, match="stm down"),
    ):
        await retriever.retrieve("query text here", session_id="s1")

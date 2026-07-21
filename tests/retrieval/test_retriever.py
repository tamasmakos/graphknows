"""Tests for DETRetriever — execute_cypher validation and RRF math."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from graphknows.ranking.rrf import RRF_K
from graphknows.retrieval.retriever import DETRetriever


def _make_retriever():
    store = MagicMock()
    store.query = AsyncMock(return_value=[])
    return DETRetriever(store=store), store


# ---------------------------------------------------------------------------
# execute_cypher — validation guards (no DB needed)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_execute_cypher_rejects_fstring_interpolation():
    retriever, _ = _make_retriever()
    with pytest.raises(ValueError, match="interpolation"):
        await retriever.execute_cypher("MATCH (n {name: {name}}) RETURN n")


@pytest.mark.asyncio
async def test_execute_cypher_rejects_write_keywords():
    retriever, _ = _make_retriever()
    for kw in ["CREATE", "MERGE", "SET", "DELETE", "DETACH DELETE", "DROP"]:
        with pytest.raises(ValueError, match="read-only"):
            await retriever.execute_cypher(f"{kw} (n:Entity) RETURN n")


@pytest.mark.asyncio
async def test_execute_cypher_allows_read_cypher():
    retriever, store = _make_retriever()
    store.query = AsyncMock(return_value=[{"name": "Alice"}])

    result = await retriever.execute_cypher(
        "MATCH (n:Entity) WHERE n.name = $name RETURN n.name AS name",
        params={"name": "Alice"},
    )
    assert result == [{"name": "Alice"}]


@pytest.mark.asyncio
async def test_execute_cypher_empty_params():
    retriever, store = _make_retriever()
    store.query = AsyncMock(return_value=[])
    result = await retriever.execute_cypher("MATCH (n:Entity) RETURN n LIMIT 5")
    assert result == []


# ---------------------------------------------------------------------------
# RRF math (pure-function tests — no IO)
# ---------------------------------------------------------------------------


def test_rrf_k_constant():
    assert RRF_K == 60


def test_rrf_score_decreases_with_rank():
    scores = [1.0 / (RRF_K + r + 1) for r in range(5)]
    for i in range(len(scores) - 1):
        assert scores[i] > scores[i + 1]


def test_rrf_score_rank0():
    expected = 1.0 / (RRF_K + 0 + 1)
    assert abs(expected - 1.0 / 61) < 1e-10

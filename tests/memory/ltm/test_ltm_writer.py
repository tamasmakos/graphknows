"""Tests for LTMWriter with mocked Neo4j store."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from graphknows.memory.ltm.writer import LTMWriter


def _make_neo_store(nodes: int = 3, edges: int = 2, rel_edges: int = 1) -> MagicMock:
    store = MagicMock()
    store.write_session_subgraph = AsyncMock(return_value={"nodes": nodes, "edges": edges})
    store.extract_and_write_relations = AsyncMock(return_value=rel_edges)
    return store


@pytest.mark.asyncio
async def test_write_returns_correct_counts() -> None:
    store = _make_neo_store(nodes=5, edges=4, rel_edges=2)
    result = await LTMWriter.write(
        session_id="sess-1",
        stm_data={"entities": [], "files": [], "chunks": [], "dialogs": [], "turns": []},
        neo_store=store,
        openrouter_key="test-key",  # key required to trigger relation extraction
    )
    assert result["nodes"] == 5
    assert result["edges"] == 6  # 4 + 2 rel edges


@pytest.mark.asyncio
async def test_write_appends_errors_on_store_failure() -> None:
    store = MagicMock()
    store.write_session_subgraph = AsyncMock(side_effect=RuntimeError("db down"))
    store.extract_and_write_relations = AsyncMock(return_value=0)
    errors: list[str] = []
    result = await LTMWriter.write(
        session_id="sess-2",
        stm_data={},
        neo_store=store,
        errors=errors,
    )
    assert result["nodes"] == 0
    assert any("write_ltm_graph" in e for e in errors)


@pytest.mark.asyncio
async def test_write_without_errors_list() -> None:
    store = _make_neo_store()
    result = await LTMWriter.write(
        session_id="sess-3",
        stm_data={},
        neo_store=store,
    )
    assert "nodes" in result
    assert "edges" in result


@pytest.mark.asyncio
async def test_write_passes_session_id() -> None:
    store = _make_neo_store()
    await LTMWriter.write("my-session", stm_data={}, neo_store=store)
    store.write_session_subgraph.assert_called_once_with("my-session", {})

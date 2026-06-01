"""Tests for MemgraphAdapter — uses mocked MemgraphStore."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from graphknows.config.adapters.memgraph_adapter import MemgraphAdapter
from graphknows.config.adapters.ports.session_db import ISessionDb


@pytest.fixture()
def mock_store() -> MagicMock:
    """Create a mock MemgraphStore with async methods."""
    store = MagicMock()
    store.close = MagicMock()
    store.ensure_indexes = AsyncMock()
    store.write_session_metadata = AsyncMock()
    store.write_file_node = AsyncMock()
    store.write_chunk_node = AsyncMock()
    store.write_entity_node = AsyncMock()
    store.write_chunk_entity_edge = AsyncMock()
    store.write_dialog_node = AsyncMock()
    store.write_turn_node = AsyncMock()
    store.write_turn_sequence = AsyncMock()
    store.write_turn_entity_edge = AsyncMock()
    store.write_tool_call_node = AsyncMock()
    store.query_entity_chunks = AsyncMock(return_value=[{"chunk_id": "c1", "text": "hi"}])
    store.delete_session = AsyncMock()
    store.get_session_nodes = AsyncMock(return_value=[{"name": "Alice"}])
    store.get_session_edges = AsyncMock(return_value=[])
    store.get_session_graph = AsyncMock(return_value={"nodes": [], "edges": []})
    store.read_session_subgraph = AsyncMock(return_value={"entities": []})
    store.get_session_node_count = AsyncMock(return_value=7)
    store.check_chain_integrity = AsyncMock(return_value={"chain_ok": True})
    store.write_relation_edge = AsyncMock()
    return store


@pytest.fixture()
def adapter(mock_store: MagicMock) -> MemgraphAdapter:
    return MemgraphAdapter(mock_store)


def test_memgraph_adapter_implements_protocol(adapter: MemgraphAdapter) -> None:
    """MemgraphAdapter must satisfy the ISessionDb protocol at runtime."""
    assert isinstance(adapter, ISessionDb)


def test_close_delegates(adapter: MemgraphAdapter, mock_store: MagicMock) -> None:
    adapter.close()
    mock_store.close.assert_called_once()


@pytest.mark.asyncio
async def test_ensure_indexes(adapter: MemgraphAdapter, mock_store: MagicMock) -> None:
    await adapter.ensure_indexes()
    mock_store.ensure_indexes.assert_awaited_once()


@pytest.mark.asyncio
async def test_write_entity_node(adapter: MemgraphAdapter, mock_store: MagicMock) -> None:
    await adapter.write_entity_node("sess-1", "Alice", "PERSON", 0.9)
    mock_store.write_entity_node.assert_awaited_once_with("sess-1", "Alice", "PERSON", 0.9)


@pytest.mark.asyncio
async def test_get_session_node_count(adapter: MemgraphAdapter, mock_store: MagicMock) -> None:
    result = await adapter.get_session_node_count("sess-1")
    assert result == 7


@pytest.mark.asyncio
async def test_read_session_subgraph(adapter: MemgraphAdapter, mock_store: MagicMock) -> None:
    result = await adapter.read_session_subgraph("sess-1")
    assert result == {"entities": []}


@pytest.mark.asyncio
async def test_check_chain_integrity(adapter: MemgraphAdapter, mock_store: MagicMock) -> None:
    result = await adapter.check_chain_integrity("sess-1")
    assert result["chain_ok"] is True

"""Tests for PgVectorAdapter — thin delegating wrapper around PgVectorStore."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from graphknows.adapters.pgvector_adapter import PgVectorAdapter
from graphknows.adapters.ports.vector_db import IVectorDb


@pytest.fixture()
def mock_store() -> MagicMock:
    store = MagicMock()
    for method in (
        "connect", "close", "ensure_schema", "upsert_chunks", "upsert_entities",
        "search_chunks", "search_entities", "write_stm_chunk", "write_stm_turn",
        "search_stm_chunks", "search_stm_turns", "promote_to_ltm",
        "search_ltm_chunks", "search_ltm_turns", "write_ltm_topic", "clear_stm",
    ):
        setattr(store, method, AsyncMock(return_value=None))
    return store


@pytest.fixture()
def adapter(mock_store: MagicMock) -> PgVectorAdapter:
    return PgVectorAdapter(mock_store)


def test_implements_ivectordb_protocol(adapter: PgVectorAdapter) -> None:
    assert isinstance(adapter, IVectorDb)


@pytest.mark.asyncio
async def test_connect_delegates(adapter: PgVectorAdapter, mock_store: MagicMock) -> None:
    await adapter.connect()
    mock_store.connect.assert_awaited_once()


@pytest.mark.asyncio
async def test_close_delegates(adapter: PgVectorAdapter, mock_store: MagicMock) -> None:
    await adapter.close()
    mock_store.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_ensure_schema_delegates(adapter: PgVectorAdapter, mock_store: MagicMock) -> None:
    await adapter.ensure_schema()
    mock_store.ensure_schema.assert_awaited_once()


@pytest.mark.asyncio
async def test_upsert_chunks_delegates(adapter: PgVectorAdapter, mock_store: MagicMock) -> None:
    chunks = [{"chunk_id": "c1", "text": "hello"}]
    await adapter.upsert_chunks("sess-1", chunks)
    mock_store.upsert_chunks.assert_awaited_once_with("sess-1", chunks)


@pytest.mark.asyncio
async def test_search_chunks_delegates(adapter: PgVectorAdapter, mock_store: MagicMock) -> None:
    mock_store.search_chunks.return_value = []
    result = await adapter.search_chunks([0.1, 0.2], "sess-1", k=3)
    mock_store.search_chunks.assert_awaited_once()
    assert result == []


@pytest.mark.asyncio
async def test_clear_stm_delegates(adapter: PgVectorAdapter, mock_store: MagicMock) -> None:
    await adapter.clear_stm("sess-1")
    mock_store.clear_stm.assert_awaited_once_with("sess-1")

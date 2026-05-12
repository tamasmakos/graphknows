"""Tests for PgVectorClient — async pgvector client with session_id namespacing."""
from __future__ import annotations

import sys
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from services.agent.graph.state import Chunk
from services.agent.infra.pgvector_client import PgVectorClient


@pytest.fixture
def mock_pool():
    """Return a mock asyncpg pool with async context manager acquire."""
    pool = MagicMock()
    pool.close = AsyncMock()

    conn = AsyncMock()
    conn.execute = AsyncMock(return_value=None)
    conn.fetch = AsyncMock(return_value=[])
    conn.fetchval = AsyncMock(return_value=1)

    acquire_cm = MagicMock()
    acquire_cm.__aenter__ = AsyncMock(return_value=conn)
    acquire_cm.__aexit__ = AsyncMock(return_value=None)
    pool.acquire = MagicMock(return_value=acquire_cm)
    return pool


@pytest.fixture
def client(mock_pool):
    """Return a PgVectorClient with a mocked pool and embedder."""
    c = PgVectorClient(dsn="postgresql://test:***@localhost:5432/testdb")
    c._pool = mock_pool
    # Mock embedder to avoid loading the model
    mock_emb = MagicMock()
    mock_emb.encode.return_value = MagicMock()
    mock_emb.encode.return_value.tolist.return_value = [[0.1] * 384]
    c._embedder = mock_emb
    return c


@pytest.mark.asyncio
async def test_ensure_schema_idempotent(client, mock_pool):
    """Test that ensure_schema runs without error and is idempotent."""
    conn = mock_pool.acquire.return_value.__aenter__.return_value
    conn.fetchval.return_value = 1  # session_id column already exists
    await client.ensure_schema()
    assert conn.execute.call_count >= 3  # extension, table, indexes


@pytest.mark.asyncio
async def test_ensure_schema_adds_missing_columns(client, mock_pool):
    """Test that ensure_schema adds session_id and metadata if missing."""
    conn = mock_pool.acquire.return_value.__aenter__.return_value
    conn.fetchval.side_effect = [0, 0]  # session_id missing, metadata missing
    await client.ensure_schema()
    # Check ALTER TABLE statements were issued
    alter_calls = [call for call in conn.execute.call_args_list if "ALTER TABLE" in str(call)]
    assert len(alter_calls) == 2


@pytest.mark.asyncio
async def test_upsert_chunks_empty(client):
    """Test upsert_chunks returns 0 for empty list."""
    result = await client.upsert_chunks("sess-1", [])
    assert result == 0


@pytest.mark.asyncio
async def test_upsert_chunks_calls_insert(client, mock_pool):
    """Test that upsert_chunks embeds texts and inserts."""
    chunks = [
        Chunk(id="c1", doc_id="d1", text="Hello world", metadata={"src": "test"}),
    ]
    result = await client.upsert_chunks("sess-1", chunks)
    conn = mock_pool.acquire.return_value.__aenter__.return_value
    assert result == 1
    conn.execute.assert_awaited_once()


@pytest.mark.asyncio
async def test_search_chunks_correct_sql(client, mock_pool):
    """Test that search_chunks includes WHERE session_id filter."""
    conn = mock_pool.acquire.return_value.__aenter__.return_value
    conn.fetch.return_value = []
    results = await client.search_chunks("sess-1", "test query", k=5)
    assert results == []
    call_sql = conn.fetch.call_args[0][0]
    assert "WHERE session_id = $1" in call_sql


@pytest.mark.asyncio
async def test_search_chunks_returns_results(client, mock_pool):
    """Test that search_chunks returns formatted dicts."""
    conn = mock_pool.acquire.return_value.__aenter__.return_value
    conn.fetch.return_value = [
        {"id": "c1", "doc_id": "d1", "text": "test text", "score": 0.95},
    ]
    results = await client.search_chunks("sess-1", "test query", k=5)
    assert len(results) == 1
    assert results[0]["id"] == "c1"
    assert results[0]["score"] == 0.95


@pytest.mark.asyncio
async def test_connect_creates_pool():
    """Test connect creates pool via asyncpg.create_pool."""
    with patch("services.agent.infra.pgvector_client.asyncpg.create_pool", new_callable=AsyncMock) as mock_create:
        client = PgVectorClient(dsn="postgresql://test:***@localhost:5432/testdb")
        await client.connect()
        mock_create.assert_awaited_once_with(
            "postgresql://test:***@localhost:5432/testdb",
            min_size=1,
            max_size=4,
        )
        await client.close()


@pytest.mark.asyncio
async def test_close_releases_pool(client, mock_pool):
    """Test close calls pool.close()."""
    await client.close()
    mock_pool.close.assert_awaited_once()
    assert client._pool is None


@pytest.mark.asyncio
async def test_embedder_lazy_loaded(client):
    """Test embedder is loaded on first access."""
    client._embedder = None
    with patch("services.agent.infra.pgvector_client.SentenceTransformer") as mock_st:
        _ = client.embedder
        mock_st.assert_called_once_with("BAAI/bge-small-en-v1.5")

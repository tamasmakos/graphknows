"""Tests for Neo4jAdapter — uses mocked Neo4jStore."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from graphknows.config.adapters.neo4j_adapter import Neo4jAdapter
from graphknows.config.adapters.ports.graph_db import IGraphDb


@pytest.fixture()
def mock_store() -> MagicMock:
    """Create a mock Neo4jStore with async methods."""
    store = MagicMock()
    store.ensure_indexes = AsyncMock()
    store.upsert_chunks = AsyncMock()
    store.upsert_entities_and_mentions = AsyncMock()
    store.upsert_topics = AsyncMock()
    store.upsert_co_occurs = AsyncMock()
    store.upsert_session_entities = AsyncMock()
    store.seed_session_from_corpus = AsyncMock(return_value=3)
    store.merge_session_entities_to_corpus = AsyncMock(return_value=(5, 2))
    store.preview_session_merge = AsyncMock(return_value={"node_count": 5})
    store.get_chunk_entities = AsyncMock(return_value=[{"name": "Alice", "label": "PERSON"}])
    store.get_entity_neighbors = AsyncMock(return_value={"neighbors": []})
    store.get_chunks_for_entity = AsyncMock(return_value=[])
    store.get_entity_community = AsyncMock(return_value=None)
    store.write_session_subgraph = AsyncMock(return_value={"nodes": 1, "edges": 0})
    store.extract_and_write_relations = AsyncMock(return_value=4)
    store.summarize_topics = AsyncMock(return_value=2)
    store.run_gds_enrichment = AsyncMock(return_value={"entity_count": 10})
    store.count_nodes = AsyncMock(return_value={"chunks": 1, "entities": 2})
    store.get_segment_info = AsyncMock(return_value=None)
    store.list_segments = AsyncMock(return_value=[])
    return store


@pytest.fixture()
def adapter(mock_store: MagicMock) -> Neo4jAdapter:
    return Neo4jAdapter(mock_store)


def test_neo4j_adapter_implements_protocol(adapter: Neo4jAdapter) -> None:
    """Neo4jAdapter must satisfy the IGraphDb protocol at runtime."""
    assert isinstance(adapter, IGraphDb)


@pytest.mark.asyncio
async def test_ensure_indexes_delegates(adapter: Neo4jAdapter, mock_store: MagicMock) -> None:
    await adapter.ensure_indexes()
    mock_store.ensure_indexes.assert_awaited_once()


@pytest.mark.asyncio
async def test_upsert_chunks_delegates(adapter: Neo4jAdapter, mock_store: MagicMock) -> None:
    chunks: list[dict[str, Any]] = [{"id": "c1", "text": "hello"}]
    await adapter.upsert_chunks(chunks)
    mock_store.upsert_chunks.assert_awaited_once_with(chunks)


@pytest.mark.asyncio
async def test_seed_session_from_corpus(adapter: Neo4jAdapter, mock_store: MagicMock) -> None:
    result = await adapter.seed_session_from_corpus("sess-1", ["Alice", "Bob"])
    assert result == 3
    mock_store.seed_session_from_corpus.assert_awaited_once_with("sess-1", ["Alice", "Bob"])


@pytest.mark.asyncio
async def test_get_chunk_entities(adapter: Neo4jAdapter, mock_store: MagicMock) -> None:
    result = await adapter.get_chunk_entities("chunk-123")
    assert result == [{"name": "Alice", "label": "PERSON"}]


@pytest.mark.asyncio
async def test_run_gds_enrichment(adapter: Neo4jAdapter, mock_store: MagicMock) -> None:
    result = await adapter.run_gds_enrichment("sess-1")
    assert result["entity_count"] == 10


@pytest.mark.asyncio
async def test_extract_and_write_relations(adapter: Neo4jAdapter, mock_store: MagicMock) -> None:
    result = await adapter.extract_and_write_relations({"chunks": []}, "key-xxx")
    assert result == 4

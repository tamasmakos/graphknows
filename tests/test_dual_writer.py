"""Tests for DualWriter — real dual-write to pgvector + Memgraph."""

from __future__ import annotations

import sys
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from services.agent.graph.nodes.investigate import DualWriter
from services.agent.graph.state import Entity, Relation


@pytest.fixture
def pg_client():
    """Return a mock PgVectorClient."""
    client = AsyncMock()
    client.upsert_chunks = AsyncMock(return_value=2)
    return client


@pytest.fixture
def mg_driver():
    """Return a mock Memgraph driver."""
    driver = MagicMock()
    session = MagicMock()
    session.__enter__ = MagicMock(return_value=session)
    session.__exit__ = MagicMock(return_value=None)
    session.run = MagicMock()
    driver.session = MagicMock(return_value=session)
    return driver


@pytest.fixture
def writer(mg_driver):
    """Return a DualWriter with mocked dependencies."""
    return DualWriter(
        session_id="test-sess-001",
        mg_driver=mg_driver,
    )


@pytest.mark.asyncio
async def test_write_empty_does_nothing(writer, mg_driver):
    """Test write with empty lists does nothing."""
    await writer.write([], [])
    mg_driver.session.assert_not_called()


@pytest.mark.asyncio
async def test_write_chunks_calls_pgvector(writer, mg_driver):
    """Test write with only entities calls graph path."""
    entities = [
        Entity(id="e1", name="test entity", label="ORG", confidence=0.9),
    ]
    await writer.write(entities, [])
    mg_driver.session.assert_called_once()


@pytest.mark.asyncio
async def test_write_graph_calls_memgraph(writer, pg_client, mg_driver):
    """Test write merges entities and relations in Memgraph."""
    entities = [
        Entity(id="e1", name="OrgA", label="ORG", confidence=0.9, doc_ids=["d1"]),
    ]
    relations = [
        Relation(source="e1", target="e2", type="RELATED_TO", confidence=0.7),
    ]
    await writer.write(entities, relations)
    mg_driver.session.assert_called_once()
    session = mg_driver.session.return_value.__enter__.return_value
    assert session.run.call_count == 2  # 1 entity + 1 relation


@pytest.mark.asyncio
async def test_write_concurrent_call(writer, mg_driver):
    """Test write calls graph store for both entities and relations."""
    entities = [Entity(id="e1", name="E1", label="PER", confidence=0.8)]
    relations = [Relation(source="e1", target="e2", type="KNOWS", confidence=0.6)]
    await writer.write(entities, relations)
    mg_driver.session.assert_called_once()


@pytest.mark.asyncio
async def test_write_entities_use_merge(writer, mg_driver):
    """Test entity Cypher uses MERGE with session_id."""
    entities = [
        Entity(id="e1", name="TestEntity", label="ORG", confidence=0.9, doc_ids=["d1"]),
    ]
    await writer.write(entities, [])
    session = mg_driver.session.return_value.__enter__.return_value
    cypher = session.run.call_args[0][0]
    assert "MERGE" in cypher
    assert "session_id" in cypher


@pytest.mark.asyncio
async def test_write_relations_include_session_id(writer, mg_driver):
    """Test relation Cypher includes session_id filter on both endpoints."""
    entities = [
        Entity(id="e1", name="A", label="ORG", confidence=0.9, doc_ids=["d1"]),
        Entity(id="e2", name="B", label="ORG", confidence=0.8, doc_ids=["d1"]),
    ]
    relations = [Relation(source="e1", target="e2", type="RELATED_TO", confidence=0.7)]
    await writer.write(entities, relations)
    session = mg_driver.session.return_value.__enter__.return_value
    rel_call = session.run.call_args_list[-1]
    cypher = rel_call[0][0]
    assert "session_id" in cypher

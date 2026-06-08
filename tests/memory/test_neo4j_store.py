"""Tests for Neo4jStore — relation_edges consumed in write_session_subgraph."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


def _make_mock_neo4j_session():
    """Return a mock neo4j driver session that records run() calls."""
    mock_session = MagicMock()
    mock_session.run.return_value = []

    ctx = MagicMock()
    ctx.__enter__ = MagicMock(return_value=mock_session)
    ctx.__exit__ = MagicMock(return_value=False)

    mock_driver = MagicMock()
    mock_driver.session.return_value = ctx
    return mock_driver, mock_session


def test_write_session_subgraph_consumes_relation_edges():
    """Unit: write_session_subgraph writes relation_edges when present in stm_data."""
    from graphknows.memory.stores.neo4j import Neo4jStore, _write_relation_edges

    stm_data = {
        "entities": [
            {"name": "Alice", "type": "PERSON", "confidence": 0.9},
            {"name": "Bob", "type": "PERSON", "confidence": 0.9},
        ],
        "files": [],
        "chunks": [],
        "dialogs": [],
        "turns": [],
        "chunk_entity_edges": [],
        "turn_entity_edges": [],
        "next_edges": [],
        "relation_edges": [
            {"source": "Alice", "relation": "KNOWS", "target": "Bob", "confidence": 0.9}
        ],
    }

    written_triplets: list = []

    def fake_write_relation_edges(s, triplets, entity_by_name):
        written_triplets.extend(triplets)
        return len(triplets)

    mock_driver, _mock_session = _make_mock_neo4j_session()

    store = Neo4jStore.__new__(Neo4jStore)
    store._driver = mock_driver
    store._relation_extractor = None

    import asyncio

    with (
        patch(
            "graphknows.memory.stores.neo4j._write_evidence_graph",
            return_value=(2, 0, {}),
        ),
        patch(
            "graphknows.memory.stores.neo4j._write_interaction_graph",
            return_value=(0, 0),
        ),
        patch(
            "graphknows.memory.stores.neo4j._write_relation_edges",
            side_effect=fake_write_relation_edges,
        ),
    ):
        result = asyncio.run(store.write_session_subgraph("test-session", stm_data))

    assert len(written_triplets) == 1
    assert written_triplets[0] == ("Alice", "KNOWS", "Bob")
    assert result["edges"] >= 1


def test_write_session_subgraph_no_relation_edges_ok():
    """Unit: write_session_subgraph works without relation_edges key."""
    from graphknows.memory.stores.neo4j import Neo4jStore

    stm_data = {
        "entities": [{"name": "Alice", "type": "PERSON", "confidence": 0.9}],
        "files": [],
        "chunks": [],
        "dialogs": [],
        "turns": [],
        "chunk_entity_edges": [],
        "turn_entity_edges": [],
        "next_edges": [],
    }

    mock_driver, _ = _make_mock_neo4j_session()
    store = Neo4jStore.__new__(Neo4jStore)
    store._driver = mock_driver
    store._relation_extractor = None

    import asyncio

    with (
        patch(
            "graphknows.memory.stores.neo4j._write_evidence_graph",
            return_value=(1, 0, {}),
        ),
        patch(
            "graphknows.memory.stores.neo4j._write_interaction_graph",
            return_value=(0, 0),
        ),
    ):
        result = asyncio.run(store.write_session_subgraph("test-session", stm_data))

    assert "nodes" in result
    assert "edges" in result


@pytest.mark.asyncio
async def test_extract_and_write_relations_delegates_text_only_to_dspy():
    """DSPy owns RDF ontology inputs; Neo4j passes only chunk text."""
    from graphknows.memory.stores.neo4j import Neo4jStore

    stm_data = {
        "entities": [
            {"name": "Alice", "type": "PERSON", "confidence": 0.9},
            {"name": "Bob", "type": "PERSON", "confidence": 0.9},
        ],
        "chunks": [{"id": "c1", "text": "Alice supports Bob."}],
        "chunk_entity_edges": [
            {"chunk_id": "c1", "entity_name": "Alice"},
            {"chunk_id": "c1", "entity_name": "Bob"},
        ],
    }
    extractor = MagicMock()
    extractor.extract_relations = AsyncMock(
        return_value=([("Alice", "SUPPORTS", "Bob", {"confidence": 0.9})], [], {})
    )
    mock_driver, _mock_session = _make_mock_neo4j_session()
    store = Neo4jStore.__new__(Neo4jStore)
    store._driver = mock_driver
    store._relation_extractor = extractor

    with patch("graphknows.memory.stores.neo4j._write_relation_edges", return_value=1):
        result = await store.extract_and_write_relations(stm_data, "key")

    assert result == 1
    extractor.extract_relations.assert_awaited_once()
    assert extractor.extract_relations.await_args.kwargs == {"text": "Alice supports Bob."}


@pytest.mark.asyncio
async def test_extract_and_write_relations_preserves_relation_metadata():
    """DSPy relation metadata must reach graph edge writes."""
    from graphknows.memory.stores.neo4j import Neo4jStore

    stm_data = {
        "entities": [
            {"name": "Alice", "type": "PERSON", "confidence": 0.9},
            {"name": "Bob", "type": "PERSON", "confidence": 0.9},
        ],
        "chunks": [{"id": "c1", "text": "Alice supports Bob."}],
        "chunk_entity_edges": [
            {"chunk_id": "c1", "entity_name": "Alice"},
            {"chunk_id": "c1", "entity_name": "Bob"},
        ],
    }
    props = {
        "confidence": 0.9,
        "evidence": "Alice supports Bob.",
        "reasoning": "The sentence explicitly states support.",
        "source_type": "PERSON",
        "target_type": "PERSON",
    }
    extractor = MagicMock()
    extractor.extract_relations = AsyncMock(
        return_value=([("Alice", "SUPPORTS", "Bob", props)], [], {})
    )
    mock_driver, _mock_session = _make_mock_neo4j_session()
    store = Neo4jStore.__new__(Neo4jStore)
    store._driver = mock_driver
    store._relation_extractor = extractor

    written_triplets = []

    def fake_write_relation_edges(s, triplets, entity_by_name):
        written_triplets.extend(triplets)
        return len(triplets)

    with patch(
        "graphknows.memory.stores.neo4j._write_relation_edges",
        side_effect=fake_write_relation_edges,
    ):
        result = await store.extract_and_write_relations(stm_data, "key")

    assert result == 1
    assert written_triplets == [("Alice", "SUPPORTS", "Bob", props)]


@pytest.mark.asyncio
async def test_extract_and_write_relations_skips_when_no_chunks():
    """No chunks means there is no text to send to DSPy."""
    from graphknows.memory.stores.neo4j import Neo4jStore

    extractor = MagicMock()
    extractor.extract_relations = AsyncMock()
    store = Neo4jStore.__new__(Neo4jStore)
    store._relation_extractor = extractor

    result = await store.extract_and_write_relations(
        {"entities": [], "chunks": [], "chunk_entity_edges": []},
        "key",
    )

    assert result == 0
    extractor.extract_relations.assert_not_called()

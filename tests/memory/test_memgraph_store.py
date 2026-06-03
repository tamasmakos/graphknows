"""Tests for MemgraphStore — relation_edges in read_session_subgraph."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


@pytest.mark.integration
def test_read_session_subgraph_includes_relation_edges():
    """Integration: read_session_subgraph returns relation_edges with REL edges written."""
    from graphknows.memory.stores.memgraph import MemgraphStore

    store = MemgraphStore(uri="bolt://localhost:7687")
    session_id = "test-rel-edges-session"

    import asyncio

    async def run():
        await store.write_entity_node(
            session_id=session_id, name="Alice", entity_type="PERSON", confidence=0.9
        )
        await store.write_entity_node(
            session_id=session_id, name="Bob", entity_type="PERSON", confidence=0.9
        )
        await store.write_relation_edge(
            session_id=session_id,
            subject_name="Alice",
            relation_name="KNOWS",
            object_name="Bob",
            confidence=0.9,
        )
        subgraph = await store.read_session_subgraph(session_id)
        await store.delete_session(session_id)
        return subgraph

    subgraph = asyncio.run(run())
    assert "relation_edges" in subgraph
    rel_edges = subgraph["relation_edges"]
    assert len(rel_edges) >= 1
    edge = next(
        (e for e in rel_edges if e["source"] == "Alice" and e["target"] == "Bob"),
        None,
    )
    assert edge is not None, f"Expected Alice->Bob edge, got: {rel_edges}"
    assert edge["relation"] == "KNOWS"


def test_read_session_subgraph_relation_edges_key_present_unit():
    """Unit: read_session_subgraph always returns relation_edges key (may be empty)."""
    from graphknows.memory.stores.memgraph import MemgraphStore

    mock_session = MagicMock()
    mock_session.run.return_value = []

    mock_driver_session = MagicMock()
    mock_driver_session.__enter__ = MagicMock(return_value=mock_session)
    mock_driver_session.__exit__ = MagicMock(return_value=False)

    mock_driver = MagicMock()
    mock_driver.session.return_value = mock_driver_session

    store = MemgraphStore.__new__(MemgraphStore)
    store._driver = mock_driver

    import asyncio

    subgraph = asyncio.run(store.read_session_subgraph("any-session"))
    assert "relation_edges" in subgraph
    assert isinstance(subgraph["relation_edges"], list)

"""Integration tests for SessionGraphManager and graph-view shaping."""

import os
import socket
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from services.agent.main import _build_graph_views
from services.agent.session_graph import SessionGraphManager

MEMGRAPH_URI = "bolt://127.0.0.1:17687"
NEO4J_URI = "bolt://172.18.0.1:7687"
NEO4J_USER = "neo4j"
NEO4J_PASS = "testpassword123"
SESSION_ID = "test-sess-001"


def _memgraph_available() -> bool:
    """Return True if Memgraph bolt port is reachable."""
    try:
        s = socket.socket()
        s.settimeout(2)
        result = s.connect_ex(("127.0.0.1", 17687))
        s.close()
        return result == 0
    except OSError:
        return False


memgraph_skip = pytest.mark.skipif(
    not _memgraph_available(),
    reason="Memgraph not reachable at 127.0.0.1:17687 — service not running",
)


def make_mgr():
    return SessionGraphManager(MEMGRAPH_URI, NEO4J_URI, NEO4J_USER, NEO4J_PASS)


@memgraph_skip
def test_create_session():
    mgr = make_mgr()
    sid = mgr.create_session(SESSION_ID)
    assert sid == SESSION_ID
    mgr.close()


@memgraph_skip
def test_seed_from_neo4j():
    mgr = make_mgr()
    mgr.create_session(SESSION_ID)
    count = mgr.seed_from_neo4j(SESSION_ID, ["UKRAINE", "EU", "CEASEFIRE"])
    assert isinstance(count, int)
    assert count >= 0
    mgr.close()


@memgraph_skip
def test_add_reasoning_step():
    mgr = make_mgr()
    mgr.create_session(SESSION_ID)
    mgr.add_reasoning_step(
        SESSION_ID,
        {
            "step_num": 1,
            "thought": "test",
            "action": "search_chunks(test)",
            "observation": "[]",
            "chunk_ids": [],
        },
    )
    mgr.close()


@memgraph_skip
def test_get_session_graph():
    mgr = make_mgr()
    mgr.create_session(SESSION_ID)
    graph = mgr.get_session_graph(SESSION_ID)
    assert "nodes" in graph
    assert "edges" in graph
    assert isinstance(graph["nodes"], list)
    assert isinstance(graph["edges"], list)
    mgr.close()


@memgraph_skip
def test_close_session():
    mgr = make_mgr()
    mgr.create_session(SESSION_ID)
    mgr.add_reasoning_step(
        SESSION_ID,
        {
            "step_num": 99,
            "thought": "cleanup",
            "action": "none",
            "observation": "",
            "chunk_ids": [],
        },
    )
    mgr.close_session(SESSION_ID)
    # After close, session graph should be empty
    graph = mgr.get_session_graph(SESSION_ID)
    assert graph["nodes"] == []
    assert graph["edges"] == []
    mgr.close()


def test_build_graph_views_preserves_existing_edge_chunk_ids() -> None:
    """Existing edge provenance must survive even without get_triplets steps."""
    subgraph = {
        "nodes": [
            {
                "id": "entity-a",
                "label": "Entity",
                "type": "Entity",
                "name": "ALPHA",
                "properties": {"name": "ALPHA"},
            },
            {
                "id": "entity-b",
                "label": "Entity",
                "type": "Entity",
                "name": "BETA",
                "properties": {"name": "BETA"},
            },
        ],
        "edges": [
            {
                "id": "edge-1",
                "source": "entity-a",
                "target": "entity-b",
                "type": "CO_OCCURS",
                "properties": {"chunk_ids": ["chunk-123"], "weight": 2},
            }
        ],
    }

    graph_views = _build_graph_views(subgraph, reasoning_steps=[], sources=[])

    knowledge_edges = graph_views["knowledge"]["edges"]
    assert len(knowledge_edges) == 1
    edge_props = knowledge_edges[0]["properties"]
    assert edge_props["chunk_ids"] == ["chunk-123"]
    assert edge_props["chunk_count"] == 1
    assert edge_props["source_mode"] == "edge_provenance"


@memgraph_skip
def test_memgraph_cooccurs_preserves_chunk_ids() -> None:
    """Memgraph-backed session graph should keep CO_OCCURS provenance."""
    mgr = make_mgr()
    mgr.create_session(SESSION_ID)
    mgr.add_cooccurs_edges(SESSION_ID, [("EU", "UKRAINE", 5)])
    mgr.add_cooccurs_edges(SESSION_ID, [("EU", "UKRAINE", 7)])

    graph = mgr.get_session_graph(SESSION_ID)
    co_occurs_edges = [e for e in graph["edges"] if e["type"] == "CO_OCCURS"]
    assert co_occurs_edges
    props = co_occurs_edges[0]["properties"]
    assert props["weight"] == 7
    assert "chunk_ids" in props
    assert isinstance(props["chunk_ids"], list)
    assert mgr._memgraph_ok is True

    mgr.close_session(SESSION_ID)
    mgr.close()

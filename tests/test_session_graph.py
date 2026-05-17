"""Integration tests for SessionGraphManager."""
import sys
import os
import socket

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from services.agent.session.manager import SessionGraphManager

MEMGRAPH_URI = "bolt://172.18.0.1:17687"
NEO4J_URI = "bolt://172.18.0.1:7687"
NEO4J_USER = "neo4j"
NEO4J_PASS = "testpassword123"
SESSION_ID = "test-sess-001"


def _memgraph_available() -> bool:
    """Return True if Memgraph bolt port is reachable."""
    try:
        s = socket.socket()
        s.settimeout(2)
        result = s.connect_ex(("172.18.0.1", 17687))
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
        {"step_num": 99, "thought": "cleanup", "action": "none", "observation": "", "chunk_ids": []},
    )
    mgr.close_session(SESSION_ID)
    # After close, session graph should be empty
    graph = mgr.get_session_graph(SESSION_ID)
    assert graph["nodes"] == []
    assert graph["edges"] == []
    mgr.close()

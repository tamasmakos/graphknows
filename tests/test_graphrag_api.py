"""tests/test_graphrag_api.py

Integration tests for the GraphRAG REST API.

Requires live Neo4j and pgvector services (running locally on 172.18.0.1).
Run from the repo root:

    .venv/bin/python -m pytest tests/test_graphrag_api.py -v
"""
from __future__ import annotations

import os
import sys

import pytest
from fastapi.testclient import TestClient

# Ensure repo root is on path so `services.graphrag.api` is importable,
# and ensure the graphrag service directory itself is also on path so the
# agent's relative imports (graph_store, agent) resolve correctly.
_REPO = os.path.join(os.path.dirname(__file__), "..")
_SVC = os.path.join(_REPO, "services", "graphrag")
for _p in (_REPO, _SVC):
    if _p not in sys.path:
        sys.path.insert(0, _p)


@pytest.fixture(scope="module")
def client():
    """Create a TestClient that triggers the FastAPI lifespan (startup/shutdown)."""
    from services.graphrag.api import app  # noqa: PLC0415

    with TestClient(app, raise_server_exceptions=True) as c:
        yield c


# ---------------------------------------------------------------------------
# Health endpoint
# ---------------------------------------------------------------------------


def test_health_returns_200(client: TestClient):
    resp = client.get("/health")
    assert resp.status_code == 200, resp.text


def test_health_payload(client: TestClient):
    data = client.get("/health").json()
    assert data["status"] == "ok"
    assert data["service"] == "graphknows-graphrag"


# ---------------------------------------------------------------------------
# Retrieve endpoint — real query against live databases
# ---------------------------------------------------------------------------


def test_retrieve_returns_200(client: TestClient):
    payload = {"query": "What is the EU position on rule of law?", "max_steps": 3}
    resp = client.post("/retrieve", json=payload)
    assert resp.status_code == 200, resp.text


def test_retrieve_response_has_non_empty_answer(client: TestClient):
    payload = {"query": "What did EU Parliament debate about climate?", "max_steps": 3}
    data = client.post("/retrieve", json=payload).json()
    assert "answer" in data
    assert len(data["answer"]) > 0, "answer should not be empty"


def test_retrieve_response_schema(client: TestClient):
    payload = {"query": "Tell me about EU sanctions policy.", "max_steps": 2}
    data = client.post("/retrieve", json=payload).json()

    assert "query" in data
    assert "answer" in data
    assert "citations" in data
    assert "reasoning_steps" in data
    assert "n_steps" in data
    assert isinstance(data["citations"], list)
    assert isinstance(data["reasoning_steps"], list)
    assert isinstance(data["n_steps"], int)


def test_retrieve_echo_session_id(client: TestClient):
    payload = {
        "query": "What is the EU position on rule of law?",
        "max_steps": 2,
        "session_id": "test-session-42",
    }
    data = client.post("/retrieve", json=payload).json()
    assert data.get("session_id") == "test-session-42"

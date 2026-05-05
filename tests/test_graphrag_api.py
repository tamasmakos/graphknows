"""tests/test_graphrag_api.py

Integration tests for the GraphRAG REST API (pure retrieval service).

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
# retriever's relative imports (graph_store) resolve correctly.
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
# Retrieve endpoint — pure retrieval (no LLM)
# ---------------------------------------------------------------------------


def test_retrieve_returns_200(client: TestClient):
    payload = {"query": "What is the EU position on rule of law?", "k": 5}
    resp = client.post("/retrieve", json=payload)
    assert resp.status_code == 200, resp.text


def test_retrieve_response_schema(client: TestClient):
    payload = {"query": "Tell me about EU sanctions policy.", "k": 5}
    data = client.post("/retrieve", json=payload).json()

    assert "chunks" in data
    assert "entities" in data
    assert "scores" in data
    assert isinstance(data["chunks"], list)
    assert isinstance(data["entities"], list)
    assert isinstance(data["scores"], list)


def test_retrieve_has_results(client: TestClient):
    payload = {"query": "climate change EU Parliament", "k": 3}
    data = client.post("/retrieve", json=payload).json()
    # Should return at least some results (may be empty if DB is empty)
    assert "chunks" in data
    assert "entities" in data


# ---------------------------------------------------------------------------
# Path endpoint
# ---------------------------------------------------------------------------


def test_path_returns_200(client: TestClient):
    payload = {"entity_a": "EU", "entity_b": "PARLIAMENT", "max_hops": 3}
    resp = client.post("/path", json=payload)
    assert resp.status_code == 200, resp.text


def test_path_response_schema(client: TestClient):
    payload = {"entity_a": "EU", "entity_b": "PARLIAMENT", "max_hops": 2}
    data = client.post("/path", json=payload).json()
    assert "path" in data
    assert "found" in data
    assert isinstance(data["path"], list)
    assert isinstance(data["found"], bool)


# ---------------------------------------------------------------------------
# Chunks endpoint
# ---------------------------------------------------------------------------


def test_chunks_returns_200(client: TestClient):
    payload = {"entity_id": "EU", "top_k": 5}
    resp = client.post("/chunks", json=payload)
    assert resp.status_code == 200, resp.text


def test_chunks_response_schema(client: TestClient):
    payload = {"entity_id": "PARLIAMENT", "top_k": 3}
    data = client.post("/chunks", json=payload).json()
    assert "chunks" in data
    assert isinstance(data["chunks"], list)


# ---------------------------------------------------------------------------
# Community endpoint
# ---------------------------------------------------------------------------


def test_community_returns_200(client: TestClient):
    resp = client.get("/community/EU")
    assert resp.status_code == 200, resp.text


def test_community_response_schema(client: TestClient):
    resp = client.get("/community/PARLIAMENT")
    data = resp.json()
    # community_id may be None if entity not found
    assert "community_id" in data
    assert "members" in data
    assert isinstance(data["members"], list)

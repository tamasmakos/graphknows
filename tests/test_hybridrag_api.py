"""tests/test_hybridrag_api.py

Integration tests for the HybridRAG REST API.

These tests require live Neo4j and pgvector services.
Skip in CI by running: pytest -m "not integration"
"""
from __future__ import annotations

import os
import sys

import pytest

_REPO = os.path.join(os.path.dirname(__file__), "..")
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def client():
    """Create a TestClient against the hybridrag FastAPI app."""
    from fastapi.testclient import TestClient
    from services.hybridrag.api import app  # noqa: PLC0415

    with TestClient(app, raise_server_exceptions=True) as c:
        yield c


# ---------------------------------------------------------------------------
# Health endpoint
# ---------------------------------------------------------------------------


def test_health_returns_200(client):
    resp = client.get("/health")
    assert resp.status_code == 200, resp.text


def test_health_payload(client):
    data = client.get("/health").json()
    assert data["status"] == "ok"
    assert data["service"] == "hybridrag"


# ---------------------------------------------------------------------------
# Retrieve endpoint — returns XML string
# ---------------------------------------------------------------------------


def test_retrieve_returns_200(client):
    payload = {
        "query": "What is the EU position on rule of law?",
        "session_id": "test-session-001",
        "k_entity": 5,
        "k_semantic": 10,
        "k_final": 5,
    }
    resp = client.post("/retrieve", json=payload)
    assert resp.status_code == 200, resp.text


def test_retrieve_response_is_xml_string(client):
    payload = {
        "query": "EU AI regulation",
        "session_id": "test-session-001",
        "k_entity": 5,
        "k_semantic": 10,
        "k_final": 5,
    }
    data = client.post("/retrieve", json=payload).json()
    assert isinstance(data, str)
    assert "<retrieval_context" in data


# ---------------------------------------------------------------------------
# Ingest endpoint
# ---------------------------------------------------------------------------


def test_ingest_returns_200(client):
    payload = {
        "chunks": [{"id": "t1", "doc_id": "d1", "text": "Test chunk for integration testing."}],
        "session_id": "test-session-001",
    }
    resp = client.post("/ingest", json=payload)
    assert resp.status_code == 200, resp.text


def test_ingest_response_schema(client):
    payload = {
        "chunks": [{"id": "t2", "doc_id": "d2", "text": "Another test chunk."}],
        "session_id": "test-session-002",
    }
    data = client.post("/ingest", json=payload).json()
    assert "chunks" in data
    assert "entities" in data
    assert isinstance(data["chunks"], int)
    assert isinstance(data["entities"], int)


# ---------------------------------------------------------------------------
# Community endpoint
# ---------------------------------------------------------------------------


def test_community_returns_200_or_404(client):
    resp = client.get("/community/EU")
    assert resp.status_code in (200, 404), resp.text

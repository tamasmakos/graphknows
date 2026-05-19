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
    """Create a TestClient against the graphknows FastAPI app."""
    from fastapi.testclient import TestClient
    from graphknows.api import app  # noqa: PLC0415

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
    assert data["service"] == "graphknows"


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
    if resp.status_code == 503:
        pytest.skip("Retriever not initialised — no backend available")
    assert resp.status_code == 200, resp.text


def test_retrieve_response_is_xml_string(client):
    payload = {
        "query": "EU AI regulation",
        "session_id": "test-session-001",
        "k_entity": 5,
        "k_semantic": 10,
        "k_final": 5,
    }
    resp = client.post("/retrieve", json=payload)
    if resp.status_code == 503:
        pytest.skip("Retriever not initialised — no backend available")
    data = resp.json()
    assert isinstance(data, str)
    assert "<retrieval_context" in data


# ---------------------------------------------------------------------------
# Ingest endpoint
# ---------------------------------------------------------------------------


def test_ingest_returns_200(client):
    payload = {
        "documents": [{"text": "Test document for integration testing.", "title": "Test Doc"}],
    }
    resp = client.post("/ingest", json=payload)
    if resp.status_code == 503:
        pytest.skip("Stores not initialised — no backend available")
    assert resp.status_code in (200, 202), resp.text


def test_ingest_response_schema(client):
    payload = {
        "documents": [{"text": "Another test document.", "title": "Test Doc 2"}],
    }
    resp = client.post("/ingest", json=payload)
    if resp.status_code == 503:
        pytest.skip("Stores not initialised — no backend available")
    data = resp.json()
    assert "run_id" in data
    assert "status" in data


# ---------------------------------------------------------------------------
# Community endpoint
# ---------------------------------------------------------------------------


def test_community_returns_200_or_404(client):
    resp = client.get("/community/EU")
    assert resp.status_code in (200, 404), resp.text

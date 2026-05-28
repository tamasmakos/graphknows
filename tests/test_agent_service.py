"""
Basic HTTP tests for the Graphknows Agent Service.

These tests use httpx.AsyncClient with FastAPI's ASGI transport so no real
network calls are made.  SessionGraphManager and the graphrag HTTP call are
both mocked so the tests are fully self-contained.
"""
from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from fastapi.testclient import TestClient

# ---------------------------------------------------------------------------
# Fixtures — mock heavy singletons BEFORE importing the app
# ---------------------------------------------------------------------------


def _make_mock_sgm() -> MagicMock:
    """Return a MagicMock that satisfies SessionGraphManager's interface."""
    sgm = MagicMock()
    sgm.create_session.return_value = None
    sgm.close_session.return_value = None
    sgm.seed_from_neo4j.return_value = 0
    sgm.get_session_graph.return_value = {
        "nodes": [
            {"id": "n1", "label": "Entity", "type": "Entity", "properties": {"name": "EU Parliament"}},
        ],
        "edges": [],
    }
    sgm.close.return_value = None
    return sgm


MOCK_GRAPHRAG_RESPONSE = {
    "query": "Who leads the EU Parliament?",
    "answer": "The EU Parliament is led by its President.",
    "citations": [{"entity": "EU Parliament", "text": "...", "score": 0.95}],
    "reasoning_steps": [],
    "n_steps": 1,
    "session_id": None,
    "entities_mentioned": ["EU Parliament", "President"],
}


@pytest.fixture()
def agent_app():
    """
    Return the FastAPI app with SessionGraphManager and httpx client mocked.

    We patch both constructors so that when the lifespan runs (TestClient
    enters the context) it instantiates our mocks instead of real drivers.
    """
    import services.agent.main as main_module  # noqa: PLC0415

    mock_sgm = _make_mock_sgm()

    mock_resp = MagicMock(spec=httpx.Response)
    mock_resp.status_code = 200
    mock_resp.json.return_value = MOCK_GRAPHRAG_RESPONSE
    mock_resp.raise_for_status.return_value = None

    mock_http = MagicMock(spec=httpx.AsyncClient)
    mock_http.post = AsyncMock(return_value=mock_resp)
    mock_http.aclose = AsyncMock()
    # httpx.AsyncClient is used as a context manager via 'async with' — not
    # needed here, but make __aenter__/__aexit__ safe.
    mock_http.__aenter__ = AsyncMock(return_value=mock_http)
    mock_http.__aexit__ = AsyncMock(return_value=False)

    # Patch SessionGraphManager constructor and httpx.AsyncClient constructor
    # so the lifespan instantiates our mocks.
    with (
        patch("services.agent.main.SessionGraphManager", return_value=mock_sgm) as _,
        patch("services.agent.main.httpx.AsyncClient", return_value=mock_http) as _,
    ):
        main_module._active_sessions = set()
        yield main_module.app, mock_sgm, mock_http

    # Cleanup singletons after test
    main_module._sgm = None
    main_module._http_client = None
    main_module._active_sessions = set()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestHealthEndpoint:
    def test_health_returns_ok(self, agent_app):
        app, *_ = agent_app
        with TestClient(app, raise_server_exceptions=True) as client:
            resp = client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert body["service"] == "graphknows-agent"


class TestChatEndpoint:
    def test_chat_creates_new_session(self, agent_app):
        app, mock_sgm, mock_http = agent_app
        with TestClient(app, raise_server_exceptions=True) as client:
            resp = client.post("/chat", json={"message": "Who leads the EU Parliament?"})

        assert resp.status_code == 200
        body = resp.json()
        assert body["answer"] == MOCK_GRAPHRAG_RESPONSE["answer"]
        assert "session_id" in body
        assert isinstance(body["session_id"], str)
        assert "graph" in body
        assert "nodes" in body["graph"]
        assert "edges" in body["graph"]

        # create_session must have been called for a brand-new session
        mock_sgm.create_session.assert_called_once()

    def test_chat_continues_existing_session(self, agent_app):
        import services.agent.main as main_module  # noqa: PLC0415

        app, mock_sgm, mock_http = agent_app
        existing_id = "test-session-existing"
        main_module._active_sessions.add(existing_id)

        with TestClient(app, raise_server_exceptions=True) as client:
            resp = client.post(
                "/chat",
                json={"message": "Follow-up question?", "session_id": existing_id},
            )

        assert resp.status_code == 200
        body = resp.json()
        assert body["session_id"] == existing_id
        # create_session should NOT be called for an existing session
        mock_sgm.create_session.assert_not_called()

    def test_chat_empty_message_rejected(self, agent_app):
        app, *_ = agent_app
        with TestClient(app, raise_server_exceptions=False) as client:
            resp = client.post("/chat", json={"message": "   "})
        assert resp.status_code == 422

    def test_chat_graphrag_failure_returns_502(self, agent_app):
        app, mock_sgm, mock_http = agent_app

        error_resp = MagicMock(spec=httpx.Response)
        error_resp.status_code = 500
        error_resp.text = "Internal Server Error"
        mock_http.post = AsyncMock(
            side_effect=httpx.HTTPStatusError(
                "500", request=MagicMock(), response=error_resp
            )
        )

        with TestClient(app, raise_server_exceptions=False) as client:
            resp = client.post("/chat", json={"message": "test"})
        assert resp.status_code == 502


class TestSessionGraphEndpoint:
    def test_get_graph_existing_session(self, agent_app):
        import services.agent.main as main_module  # noqa: PLC0415

        app, mock_sgm, _ = agent_app
        sid = "graph-test-session"
        main_module._active_sessions.add(sid)

        with TestClient(app, raise_server_exceptions=True) as client:
            resp = client.get(f"/session/{sid}/graph")

        assert resp.status_code == 200
        body = resp.json()
        assert "nodes" in body
        assert "edges" in body
        mock_sgm.get_session_graph.assert_called_with(sid)

    def test_get_graph_unknown_session_returns_404(self, agent_app):
        app, *_ = agent_app
        with TestClient(app, raise_server_exceptions=False) as client:
            resp = client.get("/session/nonexistent-xyz/graph")
        assert resp.status_code == 404


class TestDeleteSessionEndpoint:
    def test_delete_existing_session(self, agent_app):
        import services.agent.main as main_module  # noqa: PLC0415

        app, mock_sgm, _ = agent_app
        sid = "delete-me-session"
        main_module._active_sessions.add(sid)

        with TestClient(app, raise_server_exceptions=True) as client:
            resp = client.delete(f"/session/{sid}")

        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "deleted"
        assert body["session_id"] == sid
        assert sid not in main_module._active_sessions
        mock_sgm.close_session.assert_called_with(sid)

    def test_delete_unknown_session_returns_404(self, agent_app):
        app, *_ = agent_app
        with TestClient(app, raise_server_exceptions=False) as client:
            resp = client.delete("/session/does-not-exist")
        assert resp.status_code == 404

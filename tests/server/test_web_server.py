"""Unit tests for the Memory Mirror web server.

Everything here is DB-free: the pure normalisers/query builders take plain
dicts, and the TestClient cases only hit paths that reject before any store is
built (or, for ``/healthz``, touch no backend at all).
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from starlette.testclient import TestClient

from graphknows.server.mcp import _state
from graphknows.server.web import server as web
from graphknows.storage.namespace import resolve


@pytest.fixture
def client(fake_settings: object) -> TestClient:
    with patch.object(web, "get_settings", return_value=fake_settings):
        yield TestClient(web.app)


@pytest.fixture
def fake_settings() -> object:
    return type("S", (), {"namespace": "tenant_default"})()


# ---------------------------------------------------------------------------
# _normalise_stm_graph
# ---------------------------------------------------------------------------


def test_normalise_stm_graph_defaults_and_dedup() -> None:
    raw = {
        "session_id": "s1",
        "nodes": [
            {"id": "a", "name": "Alice", "type": "Person", "confidence": 0.5},
            {"id": "a", "name": "Alice again"},  # duplicate id → dropped
            {"name": "Bob"},  # id falls back to name
            {},  # no identity at all → dropped
        ],
        "edges": [
            {"source": "a", "target": "Bob", "rel_type": "KNOWS", "weight": 2},
            {"from": "a", "to": "missing"},  # unknown endpoint → dropped
        ],
    }
    out = web._normalise_stm_graph(raw, limit=10)

    assert out["scope"] == "stm"
    assert out["session_id"] == "s1"
    assert [n["id"] for n in out["nodes"]] == ["a", "Bob"]
    assert out["nodes"][0] == {
        "id": "a",
        "label": "Alice",
        "type": "Person",
        "score": 0.5,
        "scope": "stm",
    }
    assert out["nodes"][1]["type"] == "Entity"  # fallback when type/label absent
    assert out["links"] == [
        {"source": "a", "target": "Bob", "label": "KNOWS", "weight": 2.0}
    ]
    assert (out["node_count"], out["edge_count"]) == (2, 1)


def test_normalise_stm_graph_applies_node_limit() -> None:
    raw = {"nodes": [{"id": f"n{i}"} for i in range(5)], "edges": []}
    out = web._normalise_stm_graph(raw, limit=2)
    assert out["node_count"] == 2
    assert out["session_id"] == ""


def test_normalise_stm_graph_empty() -> None:
    out = web._normalise_stm_graph({}, limit=10)
    assert out == {
        "scope": "stm",
        "session_id": "",
        "nodes": [],
        "links": [],
        "node_count": 0,
        "edge_count": 0,
    }


# ---------------------------------------------------------------------------
# _ltm_queries
# ---------------------------------------------------------------------------


def test_ltm_queries_scoped_to_session() -> None:
    node_cypher, edge_cypher, params = web._ltm_queries("s1", 25)
    assert params == {"session_id": "s1", "limit": 25}
    assert "$session_id" in node_cypher and "$session_id" in edge_cypher
    assert "SESSION" in node_cypher


def test_ltm_queries_global() -> None:
    node_cypher, edge_cypher, params = web._ltm_queries("", 25)
    assert params == {"limit": 25}
    assert "$session_id" not in node_cypher and "$session_id" not in edge_cypher
    assert "MATCH (e:ENTITY)" in node_cypher


# ---------------------------------------------------------------------------
# _error_payload
# ---------------------------------------------------------------------------


def test_error_payload_shape() -> None:
    assert web._error_payload(ValueError("boom")) == {
        "error": "ValueError",
        "message": "boom",
    }


# ---------------------------------------------------------------------------
# HTTP smoke tests (no live backend)
# ---------------------------------------------------------------------------


def test_healthz_needs_no_backend(client: TestClient) -> None:
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_graph_stm_without_session_id_is_400(client: TestClient) -> None:
    response = client.get("/api/graph?scope=stm")
    assert response.status_code == 400
    assert response.json()["error"] == "session_id_required"


def test_graph_invalid_scope_is_400(client: TestClient) -> None:
    response = client.get("/api/graph?scope=nope")
    assert response.status_code == 400
    assert response.json()["error"] == "invalid_scope"


def test_recall_without_session_id_is_400(client: TestClient) -> None:
    response = client.get("/api/recall?q=hello")
    assert response.status_code == 400
    assert response.json()["error"] == "session_id_required"


def test_recall_without_query_returns_empty_hits(client: TestClient) -> None:
    response = client.get("/api/recall?q=")
    assert response.status_code == 200
    assert response.json() == {"hits": []}


def test_backend_failure_becomes_503(client: TestClient) -> None:
    runtime = MagicMock()
    runtime.doctor.side_effect = RuntimeError("arcadedb down")
    with patch.object(web, "get_runtime", return_value=runtime):
        response = client.get("/api/health")
    assert response.status_code == 503
    assert response.json() == {"error": "RuntimeError", "message": "arcadedb down"}


# ---------------------------------------------------------------------------
# Namespace resolution — one rule across both transports
# ---------------------------------------------------------------------------


def test_resolve_prefers_explicit_namespace(fake_settings: object) -> None:
    assert resolve("acme", fake_settings) == "acme"
    assert resolve("", fake_settings) == "tenant_default"


def test_web_get_runtime_falls_back_to_settings_namespace(fake_settings: object) -> None:
    runtimes: dict[str, object] = {}
    with patch.object(web, "get_settings", return_value=fake_settings), patch.object(
        web, "_runtimes", runtimes
    ), patch.object(web, "Memory") as memory:
        web.get_runtime("")
        web.get_runtime("acme")
    assert memory.call_args_list[0].kwargs["namespace"] == "tenant_default"
    assert memory.call_args_list[1].kwargs["namespace"] == "acme"
    assert sorted(runtimes) == ["acme", "tenant_default"]  # cached under the resolved name


def test_mcp_and_web_resolve_identically(fake_settings: object) -> None:
    with patch.object(_state, "get_settings", return_value=fake_settings):
        assert _state._resolve("") == resolve("", fake_settings) == "tenant_default"
        assert _state._resolve("acme") == resolve("acme", fake_settings) == "acme"

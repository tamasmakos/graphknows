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
# _state_filter — golden-layer CHUNK.state clause per scope
# ---------------------------------------------------------------------------


def test_state_filter_stm_is_raw() -> None:
    assert web._state_filter("stm") == "c.state = 'raw'"


def test_state_filter_ltm_is_consolidated() -> None:
    assert web._state_filter("ltm") == "c.state = 'consolidated'"


def test_state_filter_unknown_scope_is_unfiltered() -> None:
    assert web._state_filter("nope") == ""


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


def test_graph_invalid_scope_is_400(client: TestClient) -> None:
    response = client.get("/api/graph?scope=nope")
    assert response.status_code == 400
    assert response.json()["error"] == "invalid_scope"


class _FakeGraphStore:
    """Minimal async GraphStore double — records queries, returns canned rows."""

    def __init__(self, node_rows: list[dict], edge_rows: list[dict]) -> None:
        self._node_rows = node_rows
        self._edge_rows = edge_rows
        self.queries: list[tuple[str, dict]] = []

    async def connect(self) -> None:
        return None

    async def close(self) -> None:
        return None

    async def query(self, cypher: str, **params: object) -> list[dict]:
        self.queries.append((cypher, params))
        if "MENTIONS" in cypher:
            return self._node_rows
        return self._edge_rows


def test_graph_stm_scope_no_longer_requires_session_id(client: TestClient) -> None:
    """scope=stm is now just a CHUNK.state filter — no session_id needed."""
    fake = _FakeGraphStore(
        node_rows=[
            {"id": "Alice", "label": "Alice", "type": "Person", "topic": None, "pagerank": 0.9}
        ],
        edge_rows=[],
    )
    with patch.object(web, "build_graph_store", return_value=fake):
        response = client.get("/api/graph?scope=stm")
    assert response.status_code == 200
    body = response.json()
    assert "error" not in body
    assert body["scope"] == "stm"
    assert body["node_count"] == 1
    # the state filter for stm must be threaded into the node query
    assert any("c.state = 'raw'" in q for q, _ in fake.queries)


@pytest.mark.asyncio
async def test_golden_graph_filters_links_to_known_nodes(fake_settings: object) -> None:
    """Links whose endpoints aren't both in the returned node set are dropped."""
    fake = _FakeGraphStore(
        node_rows=[
            {"id": "Alice", "label": "Alice", "type": "Person", "topic": 1, "pagerank": 0.9},
            {"id": "Bob", "label": "Bob", "type": "Person", "topic": 1, "pagerank": 0.5},
        ],
        edge_rows=[
            {"source": "Alice", "target": "Bob", "label": "KNOWS", "weight": 0.8},
            {"source": "Alice", "target": "Ghost", "label": "KNOWS", "weight": 0.1},
        ],
    )
    with (
        patch.object(web, "get_settings", return_value=fake_settings),
        patch.object(web, "build_graph_store", return_value=fake),
    ):
        out = await web._golden_graph("ltm", "", 200, "")

    assert out["scope"] == "ltm"
    assert [n["id"] for n in out["nodes"]] == ["Alice", "Bob"]
    assert out["links"] == [{"source": "Alice", "target": "Bob", "label": "KNOWS", "weight": 0.8}]
    assert (out["node_count"], out["edge_count"]) == (2, 1)
    # the state filter for ltm and the session scope must both reach the node query
    node_query, node_params = fake.queries[0]
    assert "c.state = 'consolidated'" in node_query
    assert "session_id" not in node_params


@pytest.mark.asyncio
async def test_golden_graph_session_scoping_threads_session_id(fake_settings: object) -> None:
    fake = _FakeGraphStore(node_rows=[], edge_rows=[])
    with (
        patch.object(web, "get_settings", return_value=fake_settings),
        patch.object(web, "build_graph_store", return_value=fake),
    ):
        await web._golden_graph("stm", "sess-1", 50, "")

    node_query, node_params = fake.queries[0]
    assert "c.session_id = $session_id" in node_query
    assert node_params["session_id"] == "sess-1"


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
    with (
        patch.object(web, "get_settings", return_value=fake_settings),
        patch.object(web, "_runtimes", runtimes),
        patch.object(web, "Memory") as memory,
    ):
        web.get_runtime("")
        web.get_runtime("acme")
    assert memory.call_args_list[0].kwargs["namespace"] == "tenant_default"
    assert memory.call_args_list[1].kwargs["namespace"] == "acme"
    assert sorted(runtimes) == ["acme", "tenant_default"]  # cached under the resolved name


def test_mcp_and_web_resolve_identically(fake_settings: object) -> None:
    with patch.object(_state, "get_settings", return_value=fake_settings):
        assert _state._resolve("") == resolve("", fake_settings) == "tenant_default"
        assert _state._resolve("acme") == resolve("acme", fake_settings) == "acme"

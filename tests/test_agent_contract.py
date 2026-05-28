"""Targeted tests for agent frontend contract endpoints."""

from __future__ import annotations

import os
import sys

import pytest

_REPO = os.path.join(os.path.dirname(__file__), "..")
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)


@pytest.fixture()
def agent_main_module():
    from services.agent import main as agent_main

    return agent_main


def test_relation_sources_filters_client_chunk_ids(agent_main_module) -> None:
    agent_main = agent_main_module
    original_sgm = agent_main._sgm
    original_active_sessions = set(agent_main._active_sessions)
    original_views = dict(agent_main._session_graph_views)

    captured: dict[str, list[str] | str] = {}

    def fake_get_relation_sources(
        session_id: str,
        relation_type: str,
        source_name: str,
        target_name: str,
        chunk_ids: list[str] | None = None,
    ) -> list[dict[str, str]]:
        safe_ids = [str(cid) for cid in (chunk_ids or [])]
        captured["call"] = [session_id, relation_type, source_name, target_name, *safe_ids]
        return [{"chunk_id": cid, "text": f"text:{cid}"} for cid in safe_ids]

    agent_main._sgm = type(
        "StubSGM",
        (),
        {"get_relation_sources": staticmethod(fake_get_relation_sources)},
    )()
    agent_main._active_sessions = {"sess-1"}
    agent_main._session_graph_views = {
        "sess-1": {
            "knowledge": {
                "nodes": [],
                "edges": [
                    {
                        "id": "edge-1",
                        "source": "entity-a-sess-1",
                        "target": "entity-b-sess-1",
                        "type": "KG_RELATION",
                        "properties": {
                            "relation_type": "CO_OCCURS",
                            "source_name": "Entity A",
                            "target_name": "Entity B",
                            "chunk_ids": ["allowed-1", "allowed-2"],
                        },
                    }
                ],
            }
        }
    }

    try:
        import asyncio

        payload = asyncio.run(
            agent_main.get_relation_sources(
                session_id="sess-1",
                relation_type="CO_OCCURS",
                source_name="Entity A",
                target_name="Entity B",
                chunk_ids="allowed-2,forged-9",
            )
        )
    finally:
        agent_main._sgm = original_sgm
        agent_main._active_sessions = original_active_sessions
        agent_main._session_graph_views = original_views

    assert [item["chunk_id"] for item in payload["sources"]] == ["allowed-2"]
    assert captured["call"] == [
        "sess-1",
        "CO_OCCURS",
        "Entity A",
        "Entity B",
        "allowed-2",
    ]

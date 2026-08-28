"""GraphStore.write_frame_elements does not exist (issue #143).

The FE plane (coreness, Requires/Excludes, CoreSets, semantic types) has
nowhere to land in the graph today — only FRAME and FRAME_INSTANCE vertices
exist. This pins the writer's contract: idempotent MERGEs keyed on the FE's
stable id and on the SEMTYPE name, FRAME/SEMTYPE linkage, and Requires/
Excludes edges.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from graphknows.storage.arcadedb.graph_store import GraphStore

pytestmark = pytest.mark.unit


def _mock_client() -> MagicMock:
    client = MagicMock()
    client.command = AsyncMock(return_value=[])
    client.query = AsyncMock(return_value=[])
    client.create_database = AsyncMock()
    client.drop_database = AsyncMock()
    return client


def _command_strings(client: MagicMock) -> list[str]:
    return [call.args[1] for call in client.command.await_args_list]


def _fe_rows() -> list[dict]:
    return [
        {
            "frame": "Attaching",
            "fe": "Item",
            "coreness": "Core",
            "requires": "Goal",
            "excludes": None,
            "sem_type": None,
        },
        {
            "frame": "Attaching",
            "fe": "Goal",
            "coreness": "Core",
            "requires": "Item",
            "excludes": None,
            "sem_type": None,
        },
        {
            "frame": "Aiming",
            "fe": "Agent",
            "coreness": "Core",
            "requires": None,
            "excludes": None,
            "sem_type": "Sentient",
        },
    ]


@pytest.mark.asyncio
async def test_write_frame_elements_noop_on_empty() -> None:
    client = _mock_client()
    store = GraphStore(client=client, db="mem")
    written = await store.write_frame_elements([])
    assert written == 0
    assert not client.command.await_args_list


@pytest.mark.asyncio
async def test_write_frame_elements_merges_keyed_on_stable_fe_id() -> None:
    client = _mock_client()
    store = GraphStore(client=client, db="mem")
    written = await store.write_frame_elements(_fe_rows())

    assert written == 3
    fe_merges = [c for c in client.command.await_args_list if "MERGE (n:FE" in c.args[1]]
    assert fe_merges, "expected a MERGE keyed on an FE vertex"
    ids = [r.get("id") for c in fe_merges for r in c.kwargs["params"]["rows"]]
    assert all(ids), "every FE row must carry a stable id, not an auto id"
    # Same (frame, fe) pair must produce the same key every time it's written.
    assert len(ids) == len(set(ids)) or len(set(ids)) == len(
        {(r["frame"], r["fe"]) for r in _fe_rows()}
    )


@pytest.mark.asyncio
async def test_write_frame_elements_links_fe_to_frame_and_semtype() -> None:
    client = _mock_client()
    store = GraphStore(client=client, db="mem")
    await store.write_frame_elements(_fe_rows())

    cmds = _command_strings(client)
    assert any("FE" in s and "FRAME" in s for s in cmds if "MERGE" in s and "-[" in s), (
        "expected an edge merging an FE onto its owning FRAME"
    )
    semtype_merges = [c for c in client.command.await_args_list if "MERGE (n:SEMTYPE" in c.args[1]]
    assert semtype_merges, "expected a MERGE keyed on a SEMTYPE vertex for 'Sentient'"
    names = [
        r.get("name") or r.get("id") for c in semtype_merges for r in c.kwargs["params"]["rows"]
    ]
    assert "Sentient" in names


@pytest.mark.asyncio
async def test_write_frame_elements_emits_requires_and_excludes_edges() -> None:
    client = _mock_client()
    store = GraphStore(client=client, db="mem")
    await store.write_frame_elements(
        [
            {
                "frame": "Attaching",
                "fe": "Item",
                "coreness": "Core",
                "requires": "Goal",
                "excludes": None,
                "sem_type": None,
            },
            {
                "frame": "Aiming",
                "fe": "Targeted",
                "coreness": "Core",
                "requires": None,
                "excludes": "Target_location",
                "sem_type": None,
            },
        ]
    )
    cmds = _command_strings(client)
    assert any("REQUIRES" in s for s in cmds)
    assert any("EXCLUDES" in s for s in cmds)


@pytest.mark.asyncio
async def test_write_frame_elements_is_idempotent() -> None:
    client = _mock_client()
    store = GraphStore(client=client, db="mem")
    rows = _fe_rows()

    await store.write_frame_elements(rows)
    first_call_count = len(client.command.await_args_list)
    client.command.reset_mock()
    await store.write_frame_elements(rows)
    second_call_count = len(client.command.await_args_list)

    # Same shape of statements issued each time (all MERGE, no CREATE), and no
    # growth in statement count for repeating the identical input.
    assert second_call_count == first_call_count
    cmds = _command_strings(client)
    assert not any(s.strip().upper().startswith("CREATE VERTEX") for s in cmds)
    assert all("MERGE" in s for s in cmds if "FE" in s or "SEMTYPE" in s)

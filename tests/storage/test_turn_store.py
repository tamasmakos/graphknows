"""Integration tests for the TURN short-term-memory buffer on GraphStore.

Live-ArcadeDB only (mirrors test_graph_store.py's scratch-database round
trip): TURN is written with no extraction/embedding, so there is nothing
useful to unit-test against a mocked client beyond what test_graph_store.py
already covers for the write_chunks_bulk-style MERGE pattern — these tests
exercise the real MERGE/idempotency/state-flip semantics against a live DB.
"""

from __future__ import annotations

import uuid

import pytest

from graphknows.storage.arcadedb.graph_store import GraphStore


def _arcadedb_reachable() -> bool:
    """Return True when ArcadeDB is up at the configured URL."""
    import httpx

    from graphknows.settings import GraphKnowsSettings

    try:
        s = GraphKnowsSettings()
        r = httpx.get(
            f"{s.arcadedb_url}/api/v1/exists/ltm",
            auth=(s.arcadedb_user, s.arcadedb_password),
            timeout=3,
        )
    except Exception:
        return False
    else:
        return r.status_code == 200


@pytest.fixture
async def store():
    if not _arcadedb_reachable():
        pytest.skip("ArcadeDB not reachable")

    from graphknows.settings import GraphKnowsSettings
    from graphknows.storage import build_arcadedb_client

    s = GraphKnowsSettings()
    client = build_arcadedb_client(s)
    db = f"mem_test_{uuid.uuid4().hex[:8]}"
    gs = GraphStore(client=client, db=db)
    await gs.connect()
    await gs.ensure_schema(dims=8)
    try:
        yield gs
    finally:
        try:
            await gs.drop_database()
        finally:
            await gs.close()


pytestmark = [pytest.mark.integration, pytest.mark.asyncio]


async def test_write_turn_then_list_turns_returns_it(store: GraphStore) -> None:
    await store.ensure_session("sess-1", started_at="2026-07-26T00:00:00Z")
    turn_id = await store.write_turn(
        "sess-1", "user", "hello there", seq=1, ts="2026-07-26T00:00:01Z"
    )

    assert turn_id == "sess-1:1"
    turns = await store.list_turns("sess-1")
    assert len(turns) == 1
    row = turns[0]
    assert row["turn_id"] == "sess-1:1"
    assert row["session_id"] == "sess-1"
    assert row["role"] == "user"
    assert row["text"] == "hello there"
    assert row["seq"] == 1
    assert row["ts"] == "2026-07-26T00:00:01Z"
    assert row["state"] == "raw"


async def test_next_turn_seq_starts_at_one_and_increments(store: GraphStore) -> None:
    await store.ensure_session("sess-2")
    assert await store.next_turn_seq("sess-2") == 1

    await store.write_turn("sess-2", "user", "first", seq=1)
    assert await store.next_turn_seq("sess-2") == 2

    await store.write_turn("sess-2", "assistant", "second", seq=2)
    assert await store.next_turn_seq("sess-2") == 3


async def test_list_turns_ordered_newest_first_and_respects_limit(store: GraphStore) -> None:
    await store.ensure_session("sess-3")
    # Write out of order to prove ordering is by seq, not insertion order.
    await store.write_turn("sess-3", "user", "t1", seq=1)
    await store.write_turn("sess-3", "assistant", "t3", seq=3)
    await store.write_turn("sess-3", "user", "t2", seq=2)

    turns = await store.list_turns("sess-3")
    assert [t["seq"] for t in turns] == [3, 2, 1]

    limited = await store.list_turns("sess-3", limit=1)
    assert len(limited) == 1
    assert limited[0]["seq"] == 3


async def test_list_turns_is_session_scoped(store: GraphStore) -> None:
    await store.ensure_session("sess-a")
    await store.ensure_session("sess-b")
    await store.write_turn("sess-a", "user", "in a", seq=1)
    await store.write_turn("sess-b", "user", "in b", seq=1)

    turns_a = await store.list_turns("sess-a")
    assert [t["turn_id"] for t in turns_a] == ["sess-a:1"]

    turns_b = await store.list_turns("sess-b")
    assert [t["turn_id"] for t in turns_b] == ["sess-b:1"]


async def test_mark_turns_consolidated_flips_state(store: GraphStore) -> None:
    await store.ensure_session("sess-4")
    await store.write_turn("sess-4", "user", "one", seq=1)
    await store.write_turn("sess-4", "assistant", "two", seq=2)

    flipped = await store.mark_turns_consolidated("sess-4")
    assert flipped == 2

    assert await store.list_turns("sess-4", state="raw") == []
    consolidated = await store.list_turns("sess-4", state="consolidated")
    assert sorted(t["seq"] for t in consolidated) == [1, 2]
    assert all(t["state"] == "consolidated" for t in consolidated)

    # Re-running on an already-consolidated session flips nothing further.
    assert await store.mark_turns_consolidated("sess-4") == 0


async def test_write_turn_is_idempotent_on_same_seq(store: GraphStore) -> None:
    await store.ensure_session("sess-5")
    await store.write_turn("sess-5", "user", "draft", seq=1)
    turn_id = await store.write_turn("sess-5", "user", "final", seq=1)

    assert turn_id == "sess-5:1"
    turns = await store.list_turns("sess-5")
    assert len(turns) == 1
    assert turns[0]["text"] == "final"

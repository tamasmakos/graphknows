"""Integration tests for the single-valued-predicate supersede-then-MERGE.

``GraphStore.write_relation`` tombstones the old current edge of a
single-valued predicate (``_supersede_edge``) before MERGE-ing the new one.
As two independent HTTP calls, a failure between them — or a swallowed
exception from the tombstone itself — could leave two current edges (old and
new both live) or a live current edge and no replacement. T035 wraps both
writes in one ArcadeDB transaction (T013's ``self.transaction()`` context
manager) so they commit or roll back together, and lets a tombstone failure
propagate instead of being swallowed at debug level.

Live-ArcadeDB only (mirrors test_turn_store.py's scratch-database pattern):
proving atomicity means forcing a real mid-sequence failure and checking what
survived on the actual server, which a mocked client cannot exercise.
"""

from __future__ import annotations

import uuid
from typing import Any

import pytest

from graphknows.exceptions import StoreError
from graphknows.storage.arcadedb.graph_store import GraphStore, RelationWrite

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]


@pytest.fixture
async def store(arcadedb_required: None):
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


async def _current_rel_rows(
    store: GraphStore, src_norm: str, predicate: str
) -> list[dict[str, Any]]:
    rows = await store.query(
        "MATCH (a:ENTITY {name_norm: $src})-[r:REL]->(b:ENTITY) "
        "WHERE r.predicate = $predicate "
        "RETURN b.name_norm AS dst, r.is_current AS cur",
        src=src_norm,
        predicate=predicate,
    )
    return [r for r in rows if r.get("cur")]


async def test_supersede_then_merge_leaves_exactly_one_current_edge(store: GraphStore) -> None:
    """Baseline: two writes of a single-valued predicate never coexist as current."""
    await store.write_relation(
        RelationWrite(src_name="Melanie", predicate="LIVES_IN", dst_name="Toronto", chunk_id="c1")
    )
    await store.write_relation(
        RelationWrite(src_name="Melanie", predicate="LIVES_IN", dst_name="Vancouver", chunk_id="c2")
    )

    current = await _current_rel_rows(store, "melanie", "LIVES_IN")
    assert len(current) == 1
    assert current[0]["dst"] == "vancouver"


async def test_merge_failure_rolls_back_the_tombstone_too(store: GraphStore) -> None:
    """A failure on the MERGE half of the pair must undo the tombstone half.

    Forces the second of the two HTTP calls the single-valued path issues (the
    MERGE) to fail after the first (the tombstone) already reached ArcadeDB.
    One shared transaction rolls both back — the old edge must still be
    current, not stranded as a tombstone with no live replacement.
    """
    await store.write_relation(
        RelationWrite(src_name="Priya", predicate="LIVES_IN", dst_name="Boston", chunk_id="c1")
    )

    real_command = store.command
    calls = {"n": 0}

    async def _fail_on_merge(cypher: str, **params: Any) -> list[dict]:
        calls["n"] += 1
        if calls["n"] == 2:
            raise StoreError("simulated mid-transaction failure")
        return await real_command(cypher, **params)

    store.command = _fail_on_merge  # type: ignore[method-assign]
    try:
        with pytest.raises(StoreError):
            await store.write_relation(
                RelationWrite(
                    src_name="Priya", predicate="LIVES_IN", dst_name="Seattle", chunk_id="c2"
                )
            )
    finally:
        store.command = real_command  # type: ignore[method-assign]

    current = await _current_rel_rows(store, "priya", "LIVES_IN")
    assert len(current) == 1
    assert current[0]["dst"] == "boston"


async def test_tombstone_failure_propagates_instead_of_leaving_two_current_edges(
    store: GraphStore,
) -> None:
    """A failed tombstone must abort the write, not pass silently at debug level.

    Swallowing the tombstone's exception used to let the MERGE go on to write
    a second current edge beside the one the tombstone failed to retire — the
    exact double-current-edge state a single-valued predicate must never
    reach.
    """
    await store.write_relation(
        RelationWrite(src_name="Omar", predicate="LIVES_IN", dst_name="Denver", chunk_id="c1")
    )

    real_command = store.command

    async def _fail_tombstone(cypher: str, **params: Any) -> list[dict]:
        if "r.invalid_at = $now" in cypher:
            raise StoreError("simulated tombstone failure")
        return await real_command(cypher, **params)

    store.command = _fail_tombstone  # type: ignore[method-assign]
    try:
        with pytest.raises(StoreError):
            await store.write_relation(
                RelationWrite(
                    src_name="Omar", predicate="LIVES_IN", dst_name="Austin", chunk_id="c2"
                )
            )
    finally:
        store.command = real_command  # type: ignore[method-assign]

    current = await _current_rel_rows(store, "omar", "LIVES_IN")
    assert len(current) == 1
    assert current[0]["dst"] == "denver"

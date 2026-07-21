"""Failure semantics for ArcadeDBSTMStore.delete_session.

The store previously swallowed every per-label delete error, so a partial
delete — ENTITY nodes left behind because their statement failed — was reported
to ``purge_memory`` as a clean purge. A missing vertex type is still benign (the
label was simply never written), but anything else must surface.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from graphknows.exceptions import StoreError
from graphknows.storage.arcadedb.stm_store import ArcadeDBSTMStore

_LABELS = ("TURN", "DIALOG", "CHUNK", "ENTITY", "FILE", "SESSION")


def _store(command: AsyncMock) -> ArcadeDBSTMStore:
    client = MagicMock()
    client.command = command
    return ArcadeDBSTMStore(client=client, db="stm")


@pytest.mark.asyncio
async def test_deletes_every_label_including_session() -> None:
    """One DETACH DELETE per label; SESSION is keyed by id, the rest by session_id."""
    command = AsyncMock()
    await _store(command).delete_session("s1")

    assert command.await_count == len(_LABELS)
    statements = [c.args[1] for c in command.await_args_list]
    for label in _LABELS:
        assert any(f"(n:{label} " in s for s in statements), f"{label} not deleted"
    session_stmt = next(s for s in statements if "(n:SESSION " in s)
    assert "{id: $sid}" in session_stmt


@pytest.mark.asyncio
async def test_missing_vertex_type_is_ignored() -> None:
    """A label that was never created is a no-op, not a failure."""
    command = AsyncMock(side_effect=Exception("Type with name 'DIALOG' was not found"))
    await _store(command).delete_session("s1")  # must not raise


@pytest.mark.asyncio
async def test_real_failure_raises_and_names_the_label() -> None:
    """A genuine delete failure must not be reported as a successful purge."""

    async def _fail_on_entity(_db, statement, **_kw):
        if "(n:ENTITY " in statement:
            raise RuntimeError("connection reset")

    command = AsyncMock(side_effect=_fail_on_entity)
    with pytest.raises(StoreError) as excinfo:
        await _store(command).delete_session("s1")

    assert "ENTITY" in str(excinfo.value)
    assert "connection reset" in str(excinfo.value)
    # The remaining labels are still attempted before raising.
    assert command.await_count == len(_LABELS)


# ---------------------------------------------------------------------------
# GraphStore is the store purge_memory actually calls in the golden layer, and
# it carried the same swallow-everything bug.
# ---------------------------------------------------------------------------


def _graph_store(command: AsyncMock):
    from graphknows.storage.arcadedb.graph_store import GraphStore

    client = MagicMock()
    client.command = command
    return GraphStore(client=client, db="mem")


@pytest.mark.asyncio
async def test_graph_store_missing_type_is_ignored() -> None:
    command = AsyncMock(side_effect=Exception("Type with name 'FRAME' was not found"))
    await _graph_store(command).delete_session("s1")  # must not raise


@pytest.mark.asyncio
async def test_graph_store_real_failure_raises() -> None:
    """purge_memory must not report a clean purge over a failed delete."""

    async def _fail_on_chunk(_db, statement, **_kw):
        if "(n:CHUNK " in statement:
            raise RuntimeError("disk full")

    command = AsyncMock(side_effect=_fail_on_chunk)
    with pytest.raises(StoreError) as excinfo:
        await _graph_store(command).delete_session("s1")

    assert "CHUNK" in str(excinfo.value)
    assert "disk full" in str(excinfo.value)

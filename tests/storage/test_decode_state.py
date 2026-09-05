"""CHUNK.decode_state: the write side and the re-drive query (FR-023).

A nullable marker ('' | 'ok' | 'failed') on the existing CHUNK vertex — no new
vertex type, no new edge type. ``failed_decode_chunks`` is the only reader, and
without it the marker would be a write-only channel.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from graphknows.storage.arcadedb.graph_store import GraphStore


def _mock_client(query_rows: list[dict] | None = None) -> MagicMock:
    client = MagicMock()
    client.command = AsyncMock(return_value=[])
    client.query = AsyncMock(return_value=query_rows or [])
    return client


def _chunk_merge_call(client: MagicMock):
    calls = [
        c
        for c in client.command.await_args_list
        if "MERGE (c:CHUNK {chunk_id: row.cid})" in c.args[1]
    ]
    assert len(calls) == 1
    return calls[0]


@pytest.mark.asyncio
async def test_write_chunk_carries_decode_state() -> None:
    client = _mock_client()
    store = GraphStore(client=client)

    await store.write_chunk({"chunk_id": "c1", "session_id": "s1", "decode_state": "failed"})

    call = _chunk_merge_call(client)
    assert "c.decode_state = row.ds" in call.args[1]
    assert call.kwargs["params"]["rows"][0]["ds"] == "failed"


@pytest.mark.asyncio
async def test_write_chunks_bulk_defaults_decode_state_to_empty() -> None:
    """A chunk that never saw an LLM decode carries '' — absent, not 'ok'."""
    client = _mock_client()
    store = GraphStore(client=client)

    await store.write_chunks_bulk(
        [
            {"chunk_id": "c1", "session_id": "s1"},
            {"chunk_id": "c2", "session_id": "s1", "decode_state": "ok"},
        ]
    )

    rows = _chunk_merge_call(client).kwargs["params"]["rows"]
    assert [r["ds"] for r in rows] == ["", "ok"]


@pytest.mark.asyncio
async def test_failed_decode_chunks_filters_by_session_and_marker() -> None:
    client = _mock_client(
        query_rows=[
            {
                "chunk_id": "c1",
                "session_id": "s1",
                "text": "t1",
                "speaker": None,
                "ts": None,
                "seq": 0,
            }
        ]
    )
    store = GraphStore(client=client)

    rows = await store.failed_decode_chunks("s1")

    cypher = client.query.await_args.args[1]
    assert "c.decode_state = 'failed'" in cypher
    assert "c.session_id = $sid" in cypher
    assert client.query.await_args.kwargs["params"]["sid"] == "s1"
    assert rows == [
        {
            "chunk_id": "c1",
            "session_id": "s1",
            "text": "t1",
            "speaker": "",
            "ts": "",
            "seq": 0,
        }
    ]


@pytest.mark.asyncio
async def test_failed_decode_chunks_without_session_does_not_query() -> None:
    client = _mock_client()
    store = GraphStore(client=client)

    assert await store.failed_decode_chunks("") == []
    client.query.assert_not_awaited()

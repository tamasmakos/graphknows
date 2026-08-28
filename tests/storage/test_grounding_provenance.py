"""``GraphStore.write_groundings`` is the single write choke point for
``ENTITY-[:INSTANCE_OF]->ONTOLOGY_CLASS`` edges, and it must refuse a row
whose provenance block is incomplete: a grounding edge that cannot be
attributed to the chunk, resource version, or matcher that produced it is
unauditable.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from graphknows.storage.arcadedb.graph_store import GraphStore


def _mock_client() -> MagicMock:
    client = MagicMock()
    client.command = AsyncMock(return_value=[])
    client.query = AsyncMock(return_value=[])
    return client


def _row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "name": "Ada",
        "uri": "u:Person",
        "channel": "ontology",
        "version": "cco-1.4",
        "matcher": "lexical",
        "score": 0.95,
        "rank": 1,
        "chunk_id": "c1",
        "span": [0, 3],
    }
    row.update(overrides)
    return row


@pytest.mark.asyncio
async def test_row_missing_chunk_id_raises() -> None:
    client = _mock_client()
    store = GraphStore(client=client)
    row = _row()
    del row["chunk_id"]

    with pytest.raises((AssertionError, ValueError)):
        await store.write_groundings([row])


@pytest.mark.asyncio
async def test_row_missing_matcher_raises() -> None:
    client = _mock_client()
    store = GraphStore(client=client)
    row = _row()
    del row["matcher"]

    with pytest.raises((AssertionError, ValueError)):
        await store.write_groundings([row])


@pytest.mark.asyncio
async def test_complete_row_is_written_with_chunk_id_in_params() -> None:
    client = _mock_client()
    store = GraphStore(client=client)

    await store.write_groundings([_row(chunk_id="c42")])

    client.command.assert_awaited_once()
    params = client.command.await_args.kwargs["params"]
    rows = params["rows"]
    assert any(r.get("chunk_id") == "c42" for r in rows), rows

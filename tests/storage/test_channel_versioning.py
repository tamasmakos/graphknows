"""Channel/version identity must survive an ontology or corpus upgrade.

Nothing records which (channel, version) a namespace holds at all — there is
no SOURCE vertex and no reader to enumerate one, so a namespace populated from
FrameNet 1.7 keeps that plane forever against any later corpus, silently.

This file pins the fix's public contract: ``write_source``/``sources`` let a
(channel, version) import register itself and be enumerated, keyed on
channel+version, not overwritten.
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


@pytest.mark.asyncio
async def test_write_source_keys_the_merge_on_channel_and_version() -> None:
    client = _mock_client()
    store = GraphStore(client=client)

    await store.write_source(
        {
            "channel": "framenet",
            "version": "a1b2c3",
            "source_hash": "a1b2c3",
            "imported": 11428,
            "dropped_expressions": 0,
            "dropped_fe_mappings": 7,
        }
    )
    await store.write_source(
        {
            "channel": "framenet",
            "version": "d4e5f6",
            "source_hash": "d4e5f6",
            "imported": 12000,
            "dropped_expressions": 3,
            "dropped_fe_mappings": 9,
        }
    )

    assert client.command.await_count == 2
    calls = client.command.await_args_list
    sql_first = calls[0].args[1]
    sql_second = calls[1].args[1]
    # Both calls MERGE on a key that names both channel and version, so the
    # two calls address two distinct vertices rather than the same one.
    for sql in (sql_first, sql_second):
        assert "MERGE" in sql
        assert "SOURCE" in sql
        assert "channel" in sql
        assert "version" in sql

    params_first = calls[0].kwargs["params"]
    params_second = calls[1].kwargs["params"]
    assert params_first["channel"] == "framenet"
    assert params_first["version"] == "a1b2c3"
    assert params_first["source_hash"] == "a1b2c3"
    assert params_first["imported"] == 11428
    assert params_first["dropped_expressions"] == 0
    assert params_first["dropped_fe_mappings"] == 7

    assert params_second["version"] == "d4e5f6"
    assert params_second["source_hash"] == "d4e5f6"
    assert params_second["imported"] == 12000
    assert params_second["dropped_expressions"] == 3
    assert params_second["dropped_fe_mappings"] == 9

    # The two writes must not collide on the same merge key.
    assert (params_first["channel"], params_first["version"]) != (
        params_second["channel"],
        params_second["version"],
    )


@pytest.mark.asyncio
async def test_sources_enumerates_every_version_present() -> None:
    client = _mock_client()
    store = GraphStore(client=client)
    client.query = AsyncMock(
        return_value=[
            {
                "channel": "framenet",
                "version": "a1b2c3",
                "source_hash": "a1b2c3",
                "imported": 11428,
                "dropped_expressions": 0,
                "dropped_fe_mappings": 7,
            },
            {
                "channel": "framenet",
                "version": "d4e5f6",
                "source_hash": "d4e5f6",
                "imported": 12000,
                "dropped_expressions": 3,
                "dropped_fe_mappings": 9,
            },
        ]
    )

    rows = await store.sources("framenet")

    assert {r["version"] for r in rows} == {"a1b2c3", "d4e5f6"}
    by_version = {r["version"]: r for r in rows}
    assert by_version["a1b2c3"]["source_hash"] == "a1b2c3"
    assert by_version["a1b2c3"]["imported"] == 11428
    assert by_version["a1b2c3"]["dropped_fe_mappings"] == 7
    assert by_version["d4e5f6"]["dropped_expressions"] == 3

    filtered_sql = client.query.await_args.args[1]
    assert "channel" in filtered_sql

    client.query.reset_mock()
    await store.sources()
    unfiltered_sql = client.query.await_args.args[1]
    assert "channel" not in unfiltered_sql

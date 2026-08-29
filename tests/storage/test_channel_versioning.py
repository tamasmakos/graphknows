"""Channel/version identity must survive an ontology or corpus upgrade.

Today ``write_groundings`` MERGEs ``INSTANCE_OF`` on its endpoints alone and
assigns ``version`` in the SET clause, so a second version of the same
(entity, class) pair overwrites the first: two versions cannot coexist.
Nothing records which (channel, version) a namespace holds at all — there is
no SOURCE vertex and no reader to enumerate one, so a namespace populated from
FrameNet 1.7 keeps that plane forever against any later corpus, silently.

This file pins the fix's public contract:
- ``write_source``/``sources`` let a (channel, version) import register
  itself and be enumerated, keyed on channel+version, not overwritten.
- ``write_groundings`` MERGEs ``INSTANCE_OF`` on a pattern that includes
  ``version``, so two versions of one grounding coexist as two edges.
- ``supersede_groundings`` marks the superseded channel's other-version edges
  without deleting them, so a rebind is a dual-write, not data loss.
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


def _grounding_row(**overrides: object) -> dict[str, object]:
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


@pytest.mark.asyncio
async def test_two_versions_of_a_grounding_coexist() -> None:
    """The INSTANCE_OF merge pattern itself must be keyed on version.

    Fails today: the current SQL merges ``(e)-[i:INSTANCE_OF]->(o)`` on the
    endpoints alone and only assigns ``i.version`` in the SET clause, so the
    second write overwrites the first instead of creating a second edge.
    """
    client = _mock_client()
    store = GraphStore(client=client)

    await store.write_groundings([_grounding_row(version="cco-1.3")])
    await store.write_groundings([_grounding_row(version="cco-1.4")])

    for call in client.command.await_args_list:
        sql = call.args[1]
        if "INSTANCE_OF" not in sql:
            continue
        merge_idx = sql.rindex("MERGE", 0, sql.index("INSTANCE_OF"))
        set_idx = sql.index(" SET ", merge_idx)
        merge_clause = sql[merge_idx:set_idx]
        assert "INSTANCE_OF" in merge_clause
        assert "version" in merge_clause, (
            "INSTANCE_OF must be MERGEd on a pattern that includes version, "
            f"not assigned only in SET: {sql!r}"
        )


@pytest.mark.asyncio
async def test_supersede_groundings_dual_writes_without_deleting() -> None:
    client = _mock_client()
    store = GraphStore(client=client)

    await store.supersede_groundings("ontology", keep_version="NEW")

    client.command.assert_awaited_once()
    sql = client.command.await_args.args[1]
    params = client.command.await_args.kwargs["params"]

    assert "DELETE" not in sql.upper()
    assert "SUPERSEDED" in sql.upper() or "superseded" in sql
    assert "SET" in sql
    assert params.get("channel") == "ontology"
    assert params.get("keep_version") == "NEW"
    # The kept version must be excluded from whatever gets marked, not touched.
    assert "<>" in sql or "!=" in sql or "NOT" in sql.upper()

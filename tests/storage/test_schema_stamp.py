"""The SCHEMA_STAMP singleton, the version digest that fills it, and the refusal.

The decision table under test is contracts/schema-stamp.md.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from graphknows.exceptions import SchemaVersionMismatchError
from graphknows.storage.arcadedb import _schema
from graphknows.storage.arcadedb.graph_store import GraphStore


def test_schema_version_changes_with_ddl_text_and_dims(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline = _schema.schema_version(768)

    assert baseline == _schema.schema_version(768), "digest must be deterministic"
    assert baseline != _schema.schema_version(1024), "dims must reach the digest"

    monkeypatch.setattr(
        _schema,
        "_CORE_DDL",
        [*_schema._CORE_DDL, ("sql", "CREATE VERTEX TYPE LATER IF NOT EXISTS")],
    )
    assert baseline != _schema.schema_version(768), "DDL text must reach the digest"


def _client(*, pre_existing: bool, stamp: str | None) -> MagicMock:
    """A GraphStore client whose database already exists (or not) and carries a stamp (or not)."""
    client = MagicMock()
    client.command = AsyncMock(return_value=[])
    client.query = AsyncMock(return_value=[{"version": stamp}] if stamp else [])
    client.create_database = AsyncMock()
    client.database_exists = AsyncMock(return_value=pre_existing)
    return client


def _stamp_writes(client: MagicMock) -> list[dict]:
    return [
        call.kwargs["params"]
        for call in client.command.await_args_list
        if call.args[1].startswith("CREATE (s:SCHEMA_STAMP")
    ]


@pytest.mark.asyncio
async def test_fresh_namespace_stamps_and_proceeds() -> None:
    client = _client(pre_existing=False, stamp=None)
    await GraphStore(client=client, db="mem_acme").ensure_schema(dims=8)

    assert _stamp_writes(client) == [{"id": "singleton", "v": _schema.schema_version(8)}]


@pytest.mark.asyncio
async def test_matching_stamp_proceeds() -> None:
    client = _client(pre_existing=True, stamp=_schema.schema_version(8))
    await GraphStore(client=client, db="mem_acme").ensure_schema(dims=8)  # must not raise

    assert not _stamp_writes(client), "an already-stamped database must not be re-stamped"


@pytest.mark.asyncio
async def test_pre_existing_unstamped_database_refuses() -> None:
    """A graph that predates stamping is not assumed compatible (FR-014)."""
    client = _client(pre_existing=True, stamp=None)
    with pytest.raises(SchemaVersionMismatchError) as excinfo:
        await GraphStore(client=client, db="mem_acme").ensure_schema(dims=8)

    assert excinfo.value.recorded is None
    assert not _stamp_writes(client), "a pre-existing database must not be stamped silently"


@pytest.mark.asyncio
async def test_mismatched_stamp_refuses_naming_both_versions_and_the_remedy() -> None:
    client = _client(pre_existing=True, stamp="0123456789abcdef")
    with pytest.raises(SchemaVersionMismatchError) as excinfo:
        await GraphStore(client=client, db="mem_acme").ensure_schema(dims=8)

    message = str(excinfo.value)
    assert "acme" in message and "mem_acme" in message  # namespace and database
    assert "0123456789abcdef" in message  # recorded version
    assert _schema.schema_version(8) in message  # running version
    assert "drop_namespace" in message and "re-ingest" in message  # the remedy


class _RacingClient:
    """One shared database whose UNIQUE index on stamp_id admits a single stamp.

    Both openers are held at the stamp read until each has seen the database
    unstamped, so the second CREATE really is the loser of the race.
    """

    def __init__(self) -> None:
        self.stamps: list[str] = []
        self._unstamped_read = asyncio.Barrier(2)

    async def database_exists(self, db: str) -> bool:
        return False

    async def create_database(self, db: str) -> None:
        return None

    async def query(self, db: str, cypher: str, params: dict | None = None) -> list[dict]:
        if not cypher.startswith("MATCH (s:SCHEMA_STAMP"):
            return []
        if not self.stamps:
            await self._unstamped_read.wait()
        return [{"version": self.stamps[0]}] if self.stamps else []

    async def command(
        self, db: str, cypher: str, params: dict | None = None, language: str = "sql"
    ) -> list[dict]:
        if cypher.startswith("CREATE (s:SCHEMA_STAMP"):
            if self.stamps:
                raise RuntimeError("Duplicate key found on index SCHEMA_STAMP[stamp_id]")
            self.stamps.append((params or {})["v"])
        return []


@pytest.mark.asyncio
async def test_concurrent_first_open_leaves_exactly_one_stamp() -> None:
    """The rejected writer re-reads the winner's stamp instead of failing (FR-016)."""
    client = _RacingClient()
    await asyncio.gather(
        *(GraphStore(client=client, db="mem_acme").ensure_schema(dims=8) for _ in range(2))
    )

    assert client.stamps == [_schema.schema_version(8)]

"""Runtime stats/doctor must target the namespace's physical database pair."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from graphknows.memory import Memory


def _client_recording(query_dbs: list[str], exists_dbs: list[str]) -> MagicMock:
    """A mock ArcadeDBClient that records which database each call targets."""
    client = MagicMock()
    client.connect = AsyncMock()
    client.close = AsyncMock()

    async def _query(db: str, *_args: object, **_kwargs: object) -> list[dict[str, int]]:
        query_dbs.append(db)
        return [{"c": 0}]

    async def _exists(db: str) -> bool:
        exists_dbs.append(db)
        return True

    client.query = AsyncMock(side_effect=_query)
    client.database_exists = AsyncMock(side_effect=_exists)
    return client


@pytest.mark.asyncio
async def test_stats_targets_namespaced_database() -> None:
    query_dbs: list[str] = []
    client = _client_recording(query_dbs, [])

    with patch("graphknows.storage.build_arcadedb_client", return_value=client):
        await Memory(namespace="acme").stats(session_id="s1")

    # Every query goes to the namespace's single golden database.
    assert set(query_dbs) == {"mem_acme"}


@pytest.mark.asyncio
async def test_doctor_checks_namespaced_database() -> None:
    exists_dbs: list[str] = []
    client = _client_recording([], exists_dbs)

    with patch("graphknows.storage.build_arcadedb_client", return_value=client):
        result = await Memory(namespace="acme").doctor()

    assert exists_dbs == ["mem_acme"]
    assert result["arcadedb"] is True

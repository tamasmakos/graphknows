"""Lifecycle contract for ArcadeDBClient and the stores that share it.

Phase-1 hardening replaced private ``store._c`` poking with public
``connect()``/``close()`` and made ``ArcadeDBClient.connect()`` idempotent so
stores sharing one client cannot leak a second httpx pool (the latent
double-connect bug).
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from graphknows.memory.stores.arcadedb.client import ArcadeDBClient
from graphknows.memory.stores.arcadedb.ltm_store import ArcadeDBLTMStore
from graphknows.memory.stores.arcadedb.stm_store import ArcadeDBSTMStore


@pytest.mark.asyncio
async def test_connect_is_idempotent_single_pool() -> None:
    """connect() twice opens exactly one httpx pool; close() tears it down."""
    client = ArcadeDBClient("http://arcadedb:2480")
    fake_pool = AsyncMock()
    with (
        patch("httpx.AsyncClient", return_value=fake_pool) as pool_cls,
        patch("httpx.AsyncHTTPTransport"),
    ):
        assert client.connected is False
        await client.connect()
        await client.connect()  # second call must be a no-op
        assert pool_cls.call_count == 1
        assert client.connected is True

    await client.close()
    fake_pool.aclose.assert_awaited_once()
    assert client.connected is False
    # close() is idempotent — a second call does not re-invoke aclose.
    await client.close()
    fake_pool.aclose.assert_awaited_once()


@pytest.mark.asyncio
async def test_shared_client_connected_once_across_stores() -> None:
    """STM and LTM stores sharing one client connect the pool exactly once."""
    client = ArcadeDBClient("http://arcadedb:2480")
    stm = ArcadeDBSTMStore(client=client)
    ltm = ArcadeDBLTMStore(client=client)
    assert stm.client is client is ltm.client

    with (
        patch("httpx.AsyncClient", return_value=AsyncMock()) as pool_cls,
        patch("httpx.AsyncHTTPTransport"),
    ):
        await stm.connect()
        await ltm.connect()
        assert pool_cls.call_count == 1


@pytest.mark.asyncio
async def test_stm_drop_database_uses_store_database() -> None:
    """drop_database() targets this store's own database name."""
    client = MagicMock()
    client.drop_database = AsyncMock()
    stm = ArcadeDBSTMStore(client=client)
    await stm.drop_database()
    client.drop_database.assert_awaited_once_with(stm.database)
    assert stm.database == "stm"


def test_store_database_names() -> None:
    client = MagicMock()
    assert ArcadeDBSTMStore(client=client).database == "stm"
    assert ArcadeDBLTMStore(client=client).database == "ltm"

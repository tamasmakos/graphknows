"""Tests for MCP admin tools."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.mark.asyncio
async def test_memory_purge_awaits_pg_store_for_session_cleanup() -> None:
    from graphknows.mcp.tools.admin import memory_purge

    mg_store = MagicMock()
    mg_store.delete_session = AsyncMock()

    neo_session = MagicMock()
    neo_session.run.return_value = None
    neo_store = MagicMock()
    neo_store._db = "neo4j"
    neo_store._driver.session.return_value.__enter__ = MagicMock(return_value=neo_session)
    neo_store._driver.session.return_value.__exit__ = MagicMock(return_value=False)

    conn = AsyncMock()
    acquire_cm = MagicMock()
    acquire_cm.__aenter__ = AsyncMock(return_value=conn)
    acquire_cm.__aexit__ = AsyncMock(return_value=False)
    pool = MagicMock()
    pool.acquire.return_value = acquire_cm

    pg_store = MagicMock()
    pg_store.clear_stm = AsyncMock()
    pg_store._pool_or_raise.return_value = pool

    async_get_pg_store = AsyncMock(return_value=pg_store)

    with (
        patch("graphknows.mcp.tools.admin.get_mg_store", return_value=mg_store),
        patch("graphknows.mcp.tools.admin.get_neo_store", return_value=neo_store),
        patch("graphknows.mcp.tools.admin.get_pg_store", async_get_pg_store),
    ):
        result = await memory_purge(scope="both", session_id="sess-1")

    assert result["errors"] == []
    assert result["stm_deleted"] is True
    assert result["ltm_deleted"] is True
    assert async_get_pg_store.await_count == 2
    pg_store.clear_stm.assert_awaited_once_with("sess-1")
    assert conn.execute.await_count >= 4


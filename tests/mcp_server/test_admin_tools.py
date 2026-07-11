"""Tests for MCP admin tools."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.mark.asyncio
async def test_memory_purge_delegates_to_runtime() -> None:
    """memory_purge keeps the MCP adapter thin and calls the runtime facade."""
    from graphknows.mcp.tools.admin import memory_purge

    runtime = MagicMock()
    runtime.purge_memory = AsyncMock(
        return_value={
            "scope": "both",
            "session_id": "sess-1",
            "stm_deleted": True,
            "ltm_deleted": True,
            "errors": [],
        }
    )

    with patch(
        "graphknows.mcp.tools.admin.get_runtime", new=AsyncMock(return_value=runtime)
    ) as get_runtime:
        result = await memory_purge(scope="both", session_id="sess-1")

    get_runtime.assert_awaited_once_with("")
    runtime.purge_memory.assert_awaited_once_with(scope="both", session_id="sess-1")
    assert result["errors"] == []
    assert result["stm_deleted"] is True
    assert result["ltm_deleted"] is True


@pytest.mark.asyncio
async def test_memory_purge_routes_by_namespace() -> None:
    """An explicit namespace is routed to the matching runtime."""
    from graphknows.mcp.tools.admin import memory_purge

    runtime = MagicMock()
    runtime.purge_memory = AsyncMock(return_value={"errors": []})

    with patch(
        "graphknows.mcp.tools.admin.get_runtime", new=AsyncMock(return_value=runtime)
    ) as get_runtime:
        await memory_purge(scope="stm", session_id="s", namespace="acme")

    get_runtime.assert_awaited_once_with("acme")


@pytest.mark.asyncio
async def test_memory_drop_namespace_delegates() -> None:
    """memory_drop_namespace routes to the namespace's runtime and drops it."""
    from graphknows.mcp.tools.admin import memory_drop_namespace

    runtime = MagicMock()
    runtime.drop_namespace = AsyncMock(
        return_value={"dropped": True, "namespace": "acme", "databases": ["stm_acme", "ltm_acme"]}
    )

    with patch(
        "graphknows.mcp.tools.admin.get_runtime", new=AsyncMock(return_value=runtime)
    ) as get_runtime:
        result = await memory_drop_namespace(namespace="acme")

    get_runtime.assert_awaited_once_with("acme")
    runtime.drop_namespace.assert_awaited_once_with()
    assert result["dropped"] is True

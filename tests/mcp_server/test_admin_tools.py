"""Tests for MCP admin tools."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.mark.asyncio
async def test_memory_purge_delegates_to_runtime() -> None:
    """memory_purge keeps the MCP adapter thin and calls the runtime facade."""
    from graphknows.server.mcp.tools.admin import memory_purge

    runtime = MagicMock()
    runtime.purge_memory = AsyncMock(
        return_value={"session_id": "sess-1", "deleted": True, "errors": []}
    )

    with patch(
        "graphknows.server.mcp.tools.admin.get_runtime", new=AsyncMock(return_value=runtime)
    ) as get_runtime:
        result = await memory_purge(session_id="sess-1")

    get_runtime.assert_awaited_once_with()
    runtime.purge_memory.assert_awaited_once_with(session_id="sess-1")
    assert result["errors"] == []
    assert result["deleted"] is True


@pytest.mark.asyncio
async def test_memory_drop_namespace_delegates() -> None:
    """memory_drop_namespace drops the process runtime's own namespace."""
    from graphknows.server.mcp.tools.admin import memory_drop_namespace

    runtime = MagicMock()
    runtime.drop_namespace = AsyncMock(
        return_value={"dropped": True, "namespace": "acme", "databases": ["stm_acme", "ltm_acme"]}
    )

    with patch(
        "graphknows.server.mcp.tools.admin.get_runtime", new=AsyncMock(return_value=runtime)
    ) as get_runtime:
        result = await memory_drop_namespace()

    get_runtime.assert_awaited_once_with()
    runtime.drop_namespace.assert_awaited_once_with()
    assert result["dropped"] is True

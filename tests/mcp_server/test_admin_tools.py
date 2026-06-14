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

    with patch("graphknows.mcp.tools.admin.get_runtime", return_value=runtime):
        result = await memory_purge(scope="both", session_id="sess-1")

    runtime.purge_memory.assert_awaited_once_with(scope="both", session_id="sess-1")
    assert result["errors"] == []
    assert result["stm_deleted"] is True
    assert result["ltm_deleted"] is True

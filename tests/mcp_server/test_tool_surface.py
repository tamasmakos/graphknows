"""The advertised MCP tool surface: no tool lets a caller name a namespace."""

from __future__ import annotations

import pytest


@pytest.mark.asyncio
async def test_no_tool_accepts_namespace() -> None:
    """The namespace is bound by the process, so no inputSchema exposes it."""
    from graphknows.server.mcp._app import app

    offenders = [
        tool.name
        for tool in await app.list_tools()
        if "namespace" in tool.inputSchema.get("properties", {})
    ]
    assert offenders == []

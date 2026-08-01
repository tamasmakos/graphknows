"""Tests for the single tool-registration point and the unified recall verb."""

from __future__ import annotations

import pytest


def test_stdio_server_reexports_every_tool() -> None:
    """Every tool in the package inventory is importable from stdio_server."""
    from graphknows.server.mcp import stdio_server, tools

    missing = [name for name in tools.__all__ if not hasattr(stdio_server, name)]
    assert missing == []


def test_admin_tools_are_all_reexported() -> None:
    """Destructive admin tools are visible on the module surface, not just over MCP."""
    from graphknows.server.mcp.stdio_server import (  # noqa: F401
        memory_doctor,
        memory_drop_namespace,
        memory_purge,
        memory_stats,
    )


@pytest.mark.asyncio
async def test_registered_tools_match_package_inventory() -> None:
    """The FastMCP registry and tools.__all__ describe the same set of tools."""
    from graphknows.server.mcp import tools
    from graphknows.server.mcp._app import app

    registered = {tool.name for tool in await app.list_tools()}
    assert registered == set(tools.__all__)

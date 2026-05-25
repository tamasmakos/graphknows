"""Backward-compat re-export. Use graphknows.mcp.stdio_server directly."""

from __future__ import annotations

import asyncio
import logging

from graphknows.mcp.stdio_server import app, main

server = app  # old code expected ``server`` to be the MCP server instance

__all__ = [
    "app",
    "main",
    "server",
]


async def mcp_main() -> None:
    """Legacy entry point — delegates to stdio_server FastMCP server."""
    await app.run_stdio_async()


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    asyncio.run(mcp_main())

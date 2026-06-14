"""MCP admin tools."""

from __future__ import annotations

from typing import Any

from graphknows.mcp._app import app
from graphknows.mcp._state import get_runtime


@app.tool()
async def memory_doctor() -> dict[str, Any]:
    """Check graphknows backend connectivity."""
    return await get_runtime().doctor()


@app.tool()
async def memory_stats(session_id: str = "") -> dict[str, Any]:
    """Return graphknows memory statistics."""
    return await get_runtime().stats(session_id=session_id)


@app.tool()
async def memory_purge(
    scope: str = "both",
    session_id: str = "",
) -> dict[str, Any]:
    """Purge graphknows memory from STM, LTM, or both."""
    return await get_runtime().purge_memory(scope=scope, session_id=session_id)

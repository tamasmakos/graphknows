"""MCP admin tools."""

from __future__ import annotations

from typing import Any

from graphknows.server.mcp._app import app
from graphknows.server.mcp._state import get_runtime


@app.tool()
async def memory_doctor() -> dict[str, Any]:
    """Check graphknows backend connectivity."""
    runtime = await get_runtime()
    return await runtime.doctor()


@app.tool()
async def memory_stats(session_id: str = "") -> dict[str, Any]:
    """Return graphknows memory statistics."""
    runtime = await get_runtime()
    return await runtime.stats(session_id=session_id)


@app.tool()
async def memory_purge(session_id: str = "") -> dict[str, Any]:
    """Purge graphknows memory within one namespace (all, or one session)."""
    runtime = await get_runtime()
    return await runtime.purge_memory(session_id=session_id)


@app.tool()
async def memory_drop_namespace() -> dict[str, Any]:
    """Drop the server's namespace — its physical database — in one shot.

    An O(1) reset that removes the ``mem_<ns>`` database. Refuses the default
    ("") namespace; use ``memory_purge`` to clear the shared ``mem``.
    """
    runtime = await get_runtime()
    return await runtime.drop_namespace()

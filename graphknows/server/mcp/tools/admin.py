"""MCP admin tools.

The two destructive tools are defined unconditionally — they stay importable,
mockable and testable — but they are only *registered* on the app when
``GRAPHKNOWS_ENABLE_ADMIN_TOOLS`` is set. FastMCP builds ``list_tools`` from the
decorator, so a guard inside the body would still advertise a tool it then
refuses to run.
"""

from __future__ import annotations

from typing import Any

from graphknows.server.mcp._app import app
from graphknows.server.mcp._state import get_runtime, get_settings


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


async def memory_purge(session_id: str = "") -> dict[str, Any]:
    """Purge graphknows memory within one namespace (all, or one session)."""
    runtime = await get_runtime()
    return await runtime.purge_memory(session_id=session_id)


async def memory_drop_namespace() -> dict[str, Any]:
    """Drop the server's namespace — its physical database — in one shot.

    An O(1) reset that removes the ``mem_<ns>`` database. Refuses the default
    ("") namespace; use ``memory_purge`` to clear the shared ``mem``.
    """
    runtime = await get_runtime()
    return await runtime.drop_namespace()


_DESTRUCTIVE_TOOLS = (memory_drop_namespace, memory_purge)

#: The destructive tools actually advertised over MCP — empty unless the flag
#: is on. The single source of truth for the tool inventory and the startup
#: banner, so the gate cannot drift out of sync with what it gates.
ENABLED_ADMIN_TOOLS: tuple[str, ...] = ()

if get_settings().enable_admin_tools:
    for _tool in _DESTRUCTIVE_TOOLS:
        app.tool()(_tool)
    ENABLED_ADMIN_TOOLS = tuple(tool.__name__ for tool in _DESTRUCTIVE_TOOLS)

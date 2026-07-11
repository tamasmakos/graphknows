"""MCP Memory Mirror link helper."""

from __future__ import annotations

from typing import Any

from graphknows.mcp._app import app
from graphknows.mcp._state import get_runtime


@app.tool()
async def memory_mirror(
    session_id: str = "", scope: str = "ltm", namespace: str = ""
) -> dict[str, Any]:
    """Return a local Memory Mirror web UI link."""
    runtime = await get_runtime(namespace)
    return runtime.mirror(session_id=session_id, scope=scope)

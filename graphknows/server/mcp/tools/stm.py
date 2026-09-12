"""MCP tools for memory ingest and the run report."""

from __future__ import annotations

from typing import Any

from graphknows.server.mcp._app import app
from graphknows.server.mcp._bounded import bounded
from graphknows.server.mcp._state import get_runtime


@app.tool()
@bounded
async def memory_ingest(
    messages: str | list[dict] = "",
    session_id: str = "",
    user_id: str = "",
    agent_id: str = "",
    run_id: str = "",
    title: str = "",
    infer: bool = True,
    metadata: dict[str, Any] | None = None,
    text: str = "",
) -> dict[str, Any]:
    """Ingest text or a conversation thread into memory."""
    runtime = await get_runtime()
    return await runtime.ingest_memory(
        messages=messages,
        session_id=session_id,
        user_id=user_id,
        agent_id=agent_id,
        run_id=run_id,
        title=title,
        infer=infer,
        metadata=metadata,
        text=text,
    )


@app.tool()
async def memory_flush() -> dict[str, Any]:
    """Report how identity resolved and which written types nothing read back."""
    runtime = await get_runtime()
    return await runtime.flush()

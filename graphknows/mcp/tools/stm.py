"""MCP tools for STM ingest, recall, and flush."""

from __future__ import annotations

from typing import Any

from graphknows.mcp._app import app
from graphknows.mcp._state import get_runtime


@app.tool()
async def memory_ingest(
    messages: str | list[dict] = "",
    session_id: str = "",
    user_id: str = "",
    agent_id: str = "",
    run_id: str = "",
    title: str = "",
    infer: bool = True,
    memory_class: str = "raw_evidence",
    metadata: dict[str, Any] | None = None,
    text: str = "",
) -> dict[str, Any]:
    """Ingest text or a conversation thread into short-term memory."""
    return await get_runtime().ingest_memory(
        messages=messages,
        session_id=session_id,
        user_id=user_id,
        agent_id=agent_id,
        run_id=run_id,
        title=title,
        infer=infer,
        memory_class=memory_class,
        metadata=metadata,
        text=text,
    )


@app.tool()
async def memory_recall(
    query: str,
    session_id: str = "",
    top_k: int = 5,
    scope: str = "stm",
) -> dict[str, Any]:
    """Recall ranked memory hits from STM, LTM, or both."""
    return await get_runtime().recall_memory(
        query,
        session_id=session_id,
        top_k=top_k,
        scope=scope,
    )


@app.tool()
async def memory_flush(session_id: str) -> dict[str, Any]:
    """Flush a session from short-term memory into long-term memory."""
    return await get_runtime().flush_memory(session_id)

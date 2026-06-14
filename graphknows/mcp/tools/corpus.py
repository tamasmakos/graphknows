"""MCP corpus ingestion tool."""

from __future__ import annotations

from typing import Any

from graphknows.mcp._app import app
from graphknows.mcp._state import get_runtime


@app.tool()
async def corpus_ingest(
    documents: list[dict[str, Any]],
    session_id: str | None = None,
) -> dict[str, Any]:
    """Ingest a corpus of documents into the long-term knowledge graph."""
    return await get_runtime().corpus_ingest(documents, session_id=session_id)

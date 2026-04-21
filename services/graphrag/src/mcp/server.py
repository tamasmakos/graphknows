"""
MCP server that wraps the GraphRAG retrieval logic exposing it as MCP tools.

Internal dev tool — not intended for public exposure.

Tools:
- ``kg_chat``   — run the LlamaIndex ReActAgent for a query
- ``kg_schema`` — inspect the live Neo4j graph schema
- ``kg_health`` — lightweight ping
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel

from src.agent.workflow import run_agent
from src.infrastructure.neo4j_driver import get_driver

logger = logging.getLogger(__name__)
mcp = FastMCP("kg-chat-mcp")


class Message(BaseModel):
    role: str
    content: str


@mcp.tool()
async def kg_chat(
    query: str,
    messages: Optional[List[Dict[str, str]]] = None,
    database: str = "neo4j",
) -> Dict[str, Any]:
    """
    Run the GraphRAG ReActAgent for *query*.

    Parameters:
        query: User query text.
        messages: Optional prior conversation as a list of
            ``{"role": "user"|"assistant", "content": str}`` dicts.
        database: Target Neo4j database name (default ``"neo4j"``).

    Returns:
        A JSON-compatible dict with ``answer``, ``execution_time``, and optional ``error``.
    """
    start = time.time()

    # Convert plain dicts to LlamaIndex ChatMessage objects
    from llama_index.core.llms import ChatMessage, MessageRole  # type: ignore[import]

    chat_history: List[ChatMessage] = []
    if messages:
        for msg in messages:
            role_str = msg.get("role", "user")
            content = msg.get("content", "")
            role = MessageRole.USER if role_str == "user" else MessageRole.ASSISTANT
            chat_history.append(ChatMessage(role=role, content=content))

    try:
        async with get_driver() as driver:
            result = await run_agent(
                query=query,
                driver=driver,
                database=database,
                chat_history=chat_history,
            )
        return {
            "answer": result.get("answer", ""),
            "execution_time": time.time() - start,
            "sources": result.get("sources", []),
        }
    except Exception as exc:
        logger.error("kg_chat error: %s", exc, exc_info=True)
        return {
            "answer": f"Error: {exc}",
            "execution_time": time.time() - start,
            "error": str(exc),
        }


@mcp.tool()
async def kg_schema(database: str = "neo4j") -> Dict[str, Any]:
    """
    Dynamically inspect the live Neo4j graph schema.

    Returns node labels, relationship types, and property keys.
    """
    try:
        async with get_driver() as driver:
            async with driver.session(database=database) as session:
                labels_result = await (await session.run("CALL db.labels()")).data()
                rels_result = await (await session.run("CALL db.relationshipTypes()")).data()
                props_result = await (await session.run("CALL db.propertyKeys()")).data()

            return {
                "database": database,
                "node_labels": sorted({r.get("label", "") for r in labels_result}),
                "relationship_types": sorted({r.get("relationshipType", "") for r in rels_result}),
                "property_keys": sorted({r.get("propertyKey", "") for r in props_result}),
            }
    except Exception as exc:
        logger.error("kg_schema error: %s", exc, exc_info=True)
        return {"database": database, "error": str(exc)}


@mcp.tool()
async def kg_health() -> Dict[str, str]:
    """Lightweight health check."""
    return {"status": "ok"}


def main() -> None:
    """Entry point for MCP server."""
    mcp.run()


if __name__ == "__main__":
    main()
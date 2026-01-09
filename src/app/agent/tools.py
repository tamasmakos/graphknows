"""
Tool functions for the Pydantic-AI agent.

These are registered in `src.app.agent.core` as:
    agent.tool(query_memory_graph)
    agent.tool(expand_knowledge_graph)

Both tools ultimately call the MCP-backed KG chat (`kg_chat`) so we can confirm
the agent understands and can use the knowledge graph tool.
"""

from __future__ import annotations

from typing import Any, Dict

from pydantic_ai import RunContext

from .schema import ConversationContext
from src.app.mcp.server import kg_chat


async def query_memory_graph(
    ctx: RunContext[ConversationContext],
    query: str,
) -> str:
    """
    Query the knowledge graph, preferring any accumulated memory graph.

    This is intended for follow-up questions: it passes the current
    `ctx.deps.memory_graph` as `accumulated_graph_data` so the MCP tool can
    reuse and extend the existing subgraph.
    """
    accumulated: Dict[str, Any] | None = ctx.deps.memory_graph

    result = await kg_chat(
        query=query,
        messages=[],
        database="falkordb",
        accumulated_graph_data=accumulated,
    )

    # Update conversation context with the returned graph for later turns
    ctx.deps.memory_graph = result.get("graph_data")
    ctx.deps.last_query = query

    return result.get("answer", "")


async def expand_knowledge_graph(
    ctx: RunContext[ConversationContext],
    query: str,
) -> str:
    """
    Query the full knowledge graph without relying on prior memory.

    This is intended for new topics: it ignores any existing `memory_graph`
    when calling `kg_chat`, then replaces it with the latest result.
    """
    result = await kg_chat(
        query=query,
        messages=[],
        database="falkordb",
        accumulated_graph_data=None,
    )

    ctx.deps.memory_graph = result.get("graph_data")
    ctx.deps.last_query = query

    return result.get("answer", "")




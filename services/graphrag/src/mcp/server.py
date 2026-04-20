"""
MCP server that wraps the GraphRAG retrieval logic
exposing it as MCP tools.

The server exposes MCP tools that mirror the behavior of the FastAPI endpoints:
- `kg_chat`  -> POST /chat
- `kg_schema` -> GET /schema
- `kg_health` -> GET /health
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel

from src.infrastructure.config import get_app_config
from src.infrastructure.graph_db import get_database_client
from src.workflow.graph_workflow import GraphWorkflow


mcp = FastMCP("kg-chat-mcp")

class Message(BaseModel):
    role: str
    content: str


@mcp.tool()
async def kg_chat(
    query: str,
    messages: Optional[List[Dict[str, str]]] = None,
    database: str = "falkordb",
    accumulated_graph_data: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Run the focused retrieval pipeline, mirroring the /chat REST endpoint.

    Parameters:
        query: User query text.
        messages: Optional list of conversation messages, each with
            {"role": "user" | "assistant", "content": str}.
        database: Target database type ("falkordb").
        accumulated_graph_data: Optional graph data from previous calls.

    Returns:
        A JSON-compatible dict with the same shape as the /chat endpoint.
    """
    start_time = time.time()
    
    from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
    
    lc_messages = []
    if messages:
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "user":
                lc_messages.append(HumanMessage(content=content))
            elif role == "assistant":
                lc_messages.append(AIMessage(content=content))
            else:
                lc_messages.append(SystemMessage(content=content))

    try:
        workflow = GraphWorkflow(timeout=60, verbose=True)
        result = await workflow.run(query=query, messages=lc_messages)
        
        execution_time = time.time() - start_time
        
        return {
            "answer": result.get("answer"),
            "context": result.get("context"),
            "execution_time": execution_time,
            "graph_data": result.get("graph_data", {}),
            "reasoning_chain": result.get("trace", []),
            "cypher_query": "Workflow Execution",
            "confidence_score": 1.0,
        }
    except Exception as e:
        return {
            "answer": f"Error: {e}",
            "context": "",
            "execution_time": time.time() - start_time,
            "error": str(e)
        }


def _extract_single_column(results: List[Dict[str, Any]], preferred_key: str) -> List[str]:
    """Utility to pull a single column out of a list of row dicts."""
    values: List[str] = []
    for row in results:
        if preferred_key in row:
            val = row[preferred_key]
        else:
            val = next(iter(row.values()), None)
        if isinstance(val, str):
            values.append(val)
    return values


@mcp.tool()
async def kg_schema(database: str = "falkordb") -> Dict[str, Any]:
    """
    Dynamically inspect the graph schema for the requested database.
    """
    config = get_app_config()

    try:
        db = get_database_client(config, database)
        try:
            node_labels_raw = db.query("CALL db.labels()")
            rel_types_raw = db.query("CALL db.relationshipTypes()")
            prop_keys_raw = db.query("CALL db.propertyKeys()")

            node_labels = _extract_single_column(node_labels_raw, "label")
            relationship_types = _extract_single_column(rel_types_raw, "relationshipType")
            property_keys = _extract_single_column(prop_keys_raw, "propertyKey")

            return {
                "database": database,
                "node_labels": sorted(set(node_labels)),
                "relationship_types": sorted(set(relationship_types)),
                "property_keys": sorted(set(property_keys)),
            }
        finally:
            db.close()
    except Exception:
        return {}


@mcp.tool()
async def kg_health() -> Dict[str, str]:
    """Lightweight health check."""
    return {"status": "ok"}


def main() -> None:
    """Entry point."""
    mcp.run()


if __name__ == "__main__":
    main()
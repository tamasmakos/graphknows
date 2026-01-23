"""
Graph Context Management.

This module provides a request-scoped context for accumulating graph data
(nodes and edges) during agent execution. It allows tools to transparently
contribute to the final graph visualization without passing state manually.
"""

from __future__ import annotations

import logging
from contextvars import ContextVar
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)

# Type definitions
GraphData = Dict[str, Any]  # Structure with "nodes" and "edges" keys

# Context variable to hold the graph data
_graph_context: ContextVar[Optional[GraphData]] = ContextVar("graph_context", default=None)
# Context variable to hold the text context (tool outputs)
_text_context: ContextVar[Optional[List[str]]] = ContextVar("text_context", default=None)


def init_graph_context(initial_data: Optional[Dict[str, Any]] = None) -> None:
    """
    Initialize a new graph context for the current request.
    
    Args:
        initial_data: Optional existing graph data to start with (accumulation)
    """
    data = {
        "nodes": {},  # Dict[id, node_data]
        "edges": [],  # List[edge_data]
    }
    
    # Merge initial data if provided
    if initial_data:
        if "nodes" in initial_data:
            # Handle both list and dict formats for nodes (API sometimes returns lists)
            if isinstance(initial_data["nodes"], list):
                for node in initial_data["nodes"]:
                    node_id = node.get("element_id") or node.get("id")
                    if node_id:
                        data["nodes"][str(node_id)] = node
            elif isinstance(initial_data["nodes"], dict):
                data["nodes"].update(initial_data["nodes"])
                
        if "edges" in initial_data and isinstance(initial_data["edges"], list):
            data["edges"].extend(initial_data["edges"])
            
    _graph_context.set(data)
    _text_context.set([]) # Initialize empty list for text snippets
    logger.debug("Graph context initialized with %d nodes, %d edges", 
                len(data["nodes"]), len(data["edges"]))


def get_graph_context() -> Optional[GraphData]:
    """Get the current graph context data."""
    return _graph_context.get()


def get_accumulated_data() -> Dict[str, Any]:
    """
    Get the final accumulated graph data in the standard format.
    
    Returns:
        Dict with "nodes" (list) and "edges" (list)
    """
    ctx = _graph_context.get()
    if not ctx:
        return {"nodes": [], "edges": []}
        
    return {
        "nodes": list(ctx["nodes"].values()),
        "edges": ctx["edges"]
    }


def get_text_context() -> str:
    """
    Get the accumulated text context as a single string.
    """
    ctx = _text_context.get()
    if not ctx:
        return ""
    return "\n\n".join(ctx)


def capture_text_context(text: str) -> None:
    """
    Add text retrieval result to the context.
    
    Args:
        text: String content returned by a tool
    """
    ctx = _text_context.get()
    if ctx is not None:
        ctx.append(text)


def capture_nodes(nodes: List[Dict[str, Any]]) -> None:
    """
    Add nodes to the accumulated context.
    
    Args:
        nodes: List of node data dictionaries
    """
    ctx = _graph_context.get()
    if ctx is None:
        return # No context active, ignore
        
    count = 0
    for node in nodes:
        # Determine strict ID for deduplication
        node_id = node.get("element_id")
        if not node_id:
            node_id = node.get("id")
            
        if node_id:
            # Only add if not present or maybe update? For now just overwrite
            ctx["nodes"][str(node_id)] = node
            count += 1
            
    if count > 0:
        logger.debug("Captured %d nodes to graph context", count)


def capture_relationships(relationships: List[Dict[str, Any]]) -> None:
    """
    Add relationships (edges) to the accumulated context.
    
    Args:
        relationships: List of relationship data dictionaries
    """
    ctx = _graph_context.get()
    if ctx is None:
        return
    
    # Simple deduplication could be expensive, just append for now
    # The frontend usually handles deduping, or we can do it at end
    ctx["edges"].extend(relationships)
    logger.debug("Captured %d edges to graph context", len(relationships))


def capture_graph_data(nodes_dict: Dict[str, Any], edges_list: List[Dict[str, Any]]) -> None:
    """
    Merge full graph data (nodes dict + edges list) into context.
    This matches the format returned by expand_subgraph.
    """
    ctx = _graph_context.get()
    if ctx is None:
        return

    # Merge nodes
    ctx["nodes"].update(nodes_dict)
    
    # Merge edges
    ctx["edges"].extend(edges_list)
    
    logger.debug("Captured bulk graph data: %d nodes, %d edges", 
                len(nodes_dict), len(edges_list))

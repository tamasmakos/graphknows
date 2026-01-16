"""
LlamaIndex tool wrappers for agent operations.

This module provides pre-configured tools for the LlamaIndex agent.
The tools are imported from graph_tools.py for consistency.
"""

from __future__ import annotations

from typing import List

from llama_index.core.tools import FunctionTool

from src.app.services.graph_tools import (
    GRAPH_TOOLS,
    search_entities_by_keywords,
    get_entity_connections,
    get_timeline_events,
    get_topics_overview,
    semantic_search,
    expand_full_context,
    get_entity_details,
    create_graph_tools,
)


def get_all_tools() -> List[FunctionTool]:
    """
    Get all available agent tools.
    
    Returns:
        List of configured FunctionTool instances
    """
    return GRAPH_TOOLS


# Re-export for backward compatibility
__all__ = [
    "GRAPH_TOOLS",
    "get_all_tools",
    "create_graph_tools",
    "search_entities_by_keywords",
    "get_entity_connections",
    "get_timeline_events",
    "get_topics_overview",
    "semantic_search",
    "expand_full_context",
    "get_entity_details",
]

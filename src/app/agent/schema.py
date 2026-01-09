"""
Schema definitions for the Pydantic-AI agent.

`ConversationContext` is used as the `deps_type` for the agent.
It can hold session-scoped state such as an accumulated memory graph.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from pydantic import BaseModel


class ConversationContext(BaseModel):
    """
    Minimal conversation-scoped context for the agent.

    - `memory_graph`: accumulated graph data returned from previous KG queries,
      shaped like the `graph_data` in the backend/MCP response.
    - `last_query`: last user query string, useful for follow-up reasoning.
    """

    memory_graph: Optional[Dict[str, Any]] = None
    last_query: Optional[str] = None




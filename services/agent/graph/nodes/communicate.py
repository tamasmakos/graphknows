"""Communicate node: hybrid retrieval and brief generation."""
from __future__ import annotations

from typing import Any

from services.agent.graph.events import brief_done_event
from services.agent.graph.state import InvestigativeState


async def communicate_node(state: InvestigativeState) -> dict[str, Any]:
    """Generate the final briefing from gathered evidence.

    In production this runs hybrid retrieval (pgvector + Memgraph 2-hop + RRF)
    followed by an LLM call to render an EPRS-style briefing.

    Args:
        state: Current investigative state with all gathered evidence.

    Returns:
        Partial state update with ``briefing`` and ``ws_events``.
    """
    briefing = (
        f"# Investigation Brief\n\n"
        f"**Topic:** {state['topic']}\n\n"
        f"**Iterations:** {state['iteration_count']}\n\n"
        f"This is a placeholder briefing. In production, the communicate node "
        f"will perform hybrid retrieval and LLM-based brief rendering."
    )

    ws_events = list(state.get("ws_events", []))
    ws_events.append(brief_done_event(state["session_id"], briefing))

    return {
        "briefing": briefing,
        "ws_events": ws_events,
    }

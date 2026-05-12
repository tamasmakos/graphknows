"""Scope node: decompose topic into sub-questions and build the research plan."""
from __future__ import annotations

import uuid
from typing import Any

from services.agent.graph.events import plan_ready_event
from services.agent.graph.state import InvestigativeState, SubQuestion


def scope_node(state: InvestigativeState) -> dict[str, Any]:
    """Decompose the research topic into sub-questions for investigation.

    For the initial implementation, a simple heuristic splits the topic into
    a single sub-question. Future iterations will use an LLM call.

    Args:
        state: Current investigative state containing the topic.

    Returns:
        Partial state update with ``plan``, ``queries``, and ``ws_events``.
    """
    sq_id = f"sq-{uuid.uuid4().hex[:8]}"
    plan = [
        SubQuestion(
            id=sq_id,
            text=state["topic"],
            source_priorities=["eprs_briefings", "ep_debates", "web"],
        ),
    ]

    queries = [state["topic"]]

    ws_events = list(state.get("ws_events", []))
    ws_events.append(plan_ready_event(state["session_id"], plan))

    return {
        "plan": plan,
        "queries": queries,
        "ws_events": ws_events,
    }

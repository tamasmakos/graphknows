"""Scope node: decompose topic into sub-questions and build the research plan."""
from __future__ import annotations

import json
import os
import uuid
from typing import Any

from langchain_groq import ChatGroq

from services.agent.graph.events import plan_ready_event
from services.agent.graph.state import InvestigativeState, SubQuestion


def scope_node(state: InvestigativeState) -> dict[str, Any]:
    """Decompose the research topic into sub-questions for investigation.

    Uses ChatGroq to generate 3-5 sub-questions. Falls back to a simple
    heuristic (single sub-question = topic) when the LLM is unavailable.

    Args:
        state: Current investigative state containing the topic.

    Returns:
        Partial state update with ``plan``, ``queries``, and ``ws_events``.
    """
    plan: list[SubQuestion] = []
    groq_key = os.environ.get("GROQ_API_KEY")
    if groq_key:
        try:
            llm = ChatGroq(api_key=groq_key, model="llama-3.3-70b-versatile", temperature=0.2)
            prompt = (
                f"Given the research topic '{state['topic']}', decompose it into "
                "3-5 sub-questions for investigation. Return as JSON list of strings."
            )
            response = llm.invoke(prompt)
            content = str(response.content).strip()
            if content.startswith("```"):
                lines = content.splitlines()
                content = "\n".join(lines[1:-1]) if len(lines) > 2 else ""
            sub_questions = json.loads(content)
            if isinstance(sub_questions, list):
                plan = [
                    SubQuestion(
                        id=f"sq-{uuid.uuid4().hex[:8]}",
                        text=str(q),
                        source_priorities=["eprs_briefings", "ep_debates", "web"],
                    )
                    for q in sub_questions
                ]
        except Exception:
            plan = []

    if not plan:
        sq_id = f"sq-{uuid.uuid4().hex[:8]}"
        plan = [
            SubQuestion(
                id=sq_id,
                text=state["topic"],
                source_priorities=["eprs_briefings", "ep_debates", "web"],
            ),
        ]

    queries = [sq.text for sq in plan]
    ws_events = list(state.get("ws_events", []))
    ws_events.append(plan_ready_event(state["session_id"], plan))

    return {
        "plan": plan,
        "queries": queries,
        "ws_events": ws_events,
    }

"""Communicate node: hybrid retrieval and EPRS-style brief generation."""
from __future__ import annotations

import os
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq

from services.agent.graph.events import brief_done_event, retrieval_done_event
from services.agent.graph.state import InvestigativeState
from services.agent.retrieval.hybrid_retriever import HybridRetriever

EPRS_SYSTEM_PROMPT = """You are an EPRS (European Parliamentary Research Service) analyst.
Synthesize the provided evidence chunks into a structured investigative briefing.

Format:
# [Title]
**EPRS Intelligence Brief** | Session [ID] | [Date]

## Executive Summary
## Key Findings (numbered, with inline [Doc-N] citations)
## Legislative Record (CELEX/ELI numbers if present)
## Context & Analysis
## Sources

Rules:
- Cite sources inline with [Doc-N] referencing the chunk doc_id
- Be factual, concise, and authoritative
- Highlight tensions or contradictions if evidence conflicts
"""


async def communicate_node(
    state: InvestigativeState,
    retriever: HybridRetriever | None = None,
) -> dict[str, Any]:
    """Generate an EPRS-style briefing from gathered evidence.

    In production this runs hybrid retrieval (pgvector + Memgraph 2-hop + RRF)
    followed by an LLM call to render an EPRS-style briefing.

    Args:
        state: Current investigative state with all gathered evidence.
        retriever: Optional HybridRetriever instance. When None falls back to
            the first 8 chunks from state.

    Returns:
        Partial state update with ``briefing``, ``retrieval_hits``, and
        ``ws_events``.
    """
    ws_events = list(state.get("ws_events", []))
    session_id = state["session_id"]

    # 1. Hybrid retrieval
    if retriever is not None:
        hits = await retriever.retrieve(state["topic"], k_dense=20, k_final=8)
    else:
        hits = state.get("chunks", [])[:8]
    ws_events.append(retrieval_done_event(session_id, len(hits)))

    # 2. Build evidence context string from hits
    evidence_lines = []
    for i, hit in enumerate(hits, 1):
        evidence_lines.append(f"[Doc-{i}] {hit.doc_id}: {hit.text[:400]}")
    evidence_text = "\n\n".join(evidence_lines)

    # 3. LLM synthesis
    llm = ChatGroq(
        api_key=os.environ.get("GROQ_API_KEY"),
        model="llama-3.3-70b-versatile",
        temperature=0.2,
    )
    messages = [
        SystemMessage(content=EPRS_SYSTEM_PROMPT),
        HumanMessage(
            content=f"Topic: {state['topic']}\n\nEvidence:\n{evidence_text}\n\nWrite the briefing."
        ),
    ]
    response = await llm.ainvoke(messages)
    briefing = str(response.content)

    ws_events.append(brief_done_event(session_id, briefing))

    return {
        "briefing": briefing,
        "retrieval_hits": hits,
        "ws_events": ws_events,
    }


def make_communicate_node(
    retriever: HybridRetriever | None = None,
):
    """Factory: return a communicate_node closure with a bound retriever.

    Args:
        retriever: Optional HybridRetriever instance to bind.

    Returns:
        Async callable compatible with LangGraph node signature.
    """

    async def _communicate_node(state: InvestigativeState) -> dict[str, Any]:
        return await communicate_node(state, retriever=retriever)

    return _communicate_node

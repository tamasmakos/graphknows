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


def _format_node_features(nf: dict, top_n: int = 8) -> str:
    if not nf:
        return ""
    top = sorted(nf.items(), key=lambda x: x[1].get("pagerank", 0), reverse=True)[:top_n]
    lines = ["## Graph Feature Analysis (session KG)", ""]
    for eid, feat in top:
        lines.append(
            f"- **{eid}**: PageRank={feat['pagerank']:.4f}, "
            f"Betweenness={feat['betweenness']:.4f}, "
            f"Community={feat['community']}"
        )
    lines += ["", "Use this to highlight structurally central entities in the EPRS brief.", ""]
    return "\n".join(lines)


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

    # 2b. Graph feature analysis section (if available)
    node_features = state.get("node_features", {})
    graph_features_section = _format_node_features(node_features)

    # 3. LLM synthesis
    llm = ChatGroq(
        api_key=os.environ.get("GROQ_API_KEY"),
        model="llama-3.3-70b-versatile",
        temperature=0.2,
    )

    system_prompt_content = EPRS_SYSTEM_PROMPT
    if graph_features_section:
        system_prompt_content = EPRS_SYSTEM_PROMPT + "\n\n" + graph_features_section

    messages = [
        SystemMessage(content=system_prompt_content),
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
    pg_client: Any = None,
    mg_driver: Any = None,
):
    """Factory: return a communicate_node closure with bound dependencies.

    If ``retriever`` is None but ``pg_client`` and ``mg_driver`` are provided,
    a HybridRetriever is lazily constructed at invocation time (when the
    session_id is available from state).

    Args:
        retriever: Optional pre-built HybridRetriever instance.
        pg_client: Optional PgVectorClient for lazy retriever construction.
        mg_driver: Optional Memgraph driver for lazy retriever construction.

    Returns:
        Async callable compatible with LangGraph node signature.
    """

    async def _communicate_node(state: InvestigativeState) -> dict[str, Any]:
        actual_retriever = retriever
        if actual_retriever is None and pg_client is not None and mg_driver is not None:
            actual_retriever = HybridRetriever(state["session_id"], pg_client, mg_driver)
        return await communicate_node(state, retriever=actual_retriever)

    return _communicate_node

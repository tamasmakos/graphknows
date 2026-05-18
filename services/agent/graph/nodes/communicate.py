"""Communicate node: hybrid retrieval and EPRS-style brief generation."""
from __future__ import annotations

import os
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq

from services.agent.graph.events import brief_done_event, retrieval_done_event
from services.agent.graph.state import InvestigativeState

EPRS_SYSTEM_PROMPT = """You are a senior analyst at the European Parliamentary Research Service (EPRS), specialising in hybrid threats, information warfare, and democratic resilience. You produce long-form, academically rigorous intelligence reports for Members of the European Parliament.

Your reports must:
- Be 1800–2500 words minimum. Never truncate or summarise when you can elaborate.
- Follow the exact section structure below, using Markdown headings.
- Cite every factual claim inline as [Doc-N] referencing the evidence chunks provided.
- Translate structural graph insights (centrality, community clusters, bridge nodes) into fluent analytical prose — NEVER expose raw numbers or mention a 'knowledge graph'. Instead write things like: 'X emerges as a pivotal actor connecting Y and Z networks', 'the structural analysis reveals three distinct operational clusters', 'organisations at the intersection of multiple influence channels'.
- Maintain formal academic register throughout: third person, passive voice where appropriate, hedged language for uncertain claims ('evidence suggests', 'available sources indicate').
- Include specific named entities, dates, legislative references, and operational details drawn from evidence.
- Highlight contradictions or gaps in the evidence with appropriate epistemic humility.

REPORT STRUCTURE (all sections mandatory):

# [Specific Descriptive Title — not generic]
**EPRS Intelligence Analysis** | [Session ID] | [Date] | Restricted — LIBE Committee

## Executive Summary
3–4 paragraphs. Summarise the principal findings, the scale of the threat, key actors identified, and the main policy implications. Do not merely list — synthesise.

## 1. Background and Analytical Framework
Contextualise the investigation within the broader landscape of foreign information manipulation and interference (FIMI). Reference relevant EU policy frameworks (DSA, FIMI framework, strategic compass). 2–3 paragraphs.

## 2. Operational Architecture of the Campaigns
Detailed breakdown of the operational infrastructure identified. Describe campaign mechanics, technical vectors, content themes, target audiences, and geographic spread. Use specific examples from evidence. 4–6 paragraphs.

## 3. Actor Network Analysis
Analysis of the principal actors, their roles, and inter-relationships. Identify coordinating hubs, amplifying nodes, and peripheral actors. Describe the structural organisation of the network in analytical prose — which actors are central connectors, which operate as bridges between communities, which are isolated amplifiers. 4–5 paragraphs.

## 4. Funding and Organisational Structures
Where evidence permits, trace financial flows, shell organisations, and institutional proxies. Note sanctioned entities and their operational relevance. 2–3 paragraphs.

## 5. Impact Assessment: The 2024 European Parliament Elections
Evidence-based assessment of the measurable or inferred impact on electoral discourse, voter sentiment, and political outcomes. Reference specific electoral results, polling data, or documented narrative penetration. 3–4 paragraphs.

## 6. Legislative and Institutional Response
Document the EU institutional response: relevant legislation (with CELEX numbers where available), Commission decisions, EP resolutions, and their adequacy. 2–3 paragraphs.

## 7. Analytical Conclusions and Intelligence Gaps
Synthesize the principal findings. Identify what the evidence does not cover — operational gaps, attribution uncertainties, unknown funding sources. Formulate 3–5 specific recommendations for the LIBE Committee. 3–4 paragraphs.

## Sources and Methodology
List all cited sources with [Doc-N] reference numbers, titles, and URLs. Note the evidence retrieval methodology in one sentence.
"""


def _format_node_features(nf: dict, top_n: int = 12) -> str:
    if not nf:
        return ""
    top = sorted(nf.items(), key=lambda x: x[1].get("pagerank", 0), reverse=True)[:top_n]
    # Group into tiers
    central = top[:3]
    connectors = top[3:7]
    peripheral = top[7:]

    lines = ["## Structural Network Intelligence (for analytical use — translate into prose, do not quote)", ""]
    lines.append("TIER 1 — CENTRAL HUBS (highest network centrality, key to dismantling the network):")
    for eid, feat in central:
        lines.append(f"  {eid}: centrality_score={feat['pagerank']:.4f}, bridge_coefficient={feat['betweenness']:.4f}, cluster={feat['community']}")
    lines.append("")
    lines.append("TIER 2 — CONNECTOR NODES (bridge between clusters):")
    for eid, feat in connectors:
        lines.append(f"  {eid}: centrality_score={feat['pagerank']:.4f}, bridge_coefficient={feat['betweenness']:.4f}, cluster={feat['community']}")
    lines.append("")
    lines.append("TIER 3 — PERIPHERAL AMPLIFIERS:")
    for eid, feat in peripheral:
        lines.append(f"  {eid}: centrality_score={feat['pagerank']:.4f}, cluster={feat['community']}")
    lines.append("")
    lines.append("INSTRUCTION: Use this structural data to write analytically rich prose about the network architecture. Do NOT copy these numbers into the report. Instead interpret them: high pagerank = central/influential actor; high betweenness = bridge between communities; same cluster = operationally linked.")
    return "\n".join(lines)


async def communicate_node(
    state: InvestigativeState,
) -> dict[str, Any]:
    """Generate an EPRS-style briefing from gathered evidence.

    Performs hybrid retrieval via the hybridrag service, then passes the
    resulting XML context block to the LLM as a <retrieved_context> element.
    Falls back to in-state chunks when the service is unavailable.

    Args:
        state: Current investigative state with all gathered evidence.

    Returns:
        Partial state update with ``briefing``, ``retrieval_hits``, and
        ``ws_events``.
    """
    ws_events = list(state.get("ws_events", []))
    session_id = state["session_id"]

    # 1. Hybrid retrieval via GraphKnowsClient → XML context string
    import os as _os
    graphknows_url = _os.environ.get("GRAPHKNOWS_URL", "http://localhost:8025")
    retrieved_xml: str = ""
    hits: list[Any] = []

    try:
        from services.agent.tools.graph.client import GraphKnowsClient
        client = GraphKnowsClient(graphknows_url, timeout=30.0)
        retrieved_xml = await client.retrieve(
            state["topic"],
            session_id=session_id,
            k_entity=10,
            k_semantic=20,
            k_final=10,
        )
        # Count chunks from XML for event emission
        import re as _re
        hit_count = len(_re.findall(r"<chunk ", retrieved_xml))
        hits = [None] * hit_count  # placeholder list for retrieval_hits count
    except Exception as exc:
        import logging as _logging
        _logging.getLogger(__name__).warning(
            "graphknows retrieval failed, falling back to state chunks: %s", exc
        )
        state_chunks = state.get("chunks", [])[:10]
        hits = state_chunks
        # Build minimal XML fallback from state chunks
        fallback_lines = ["<retrieved_context source='state_fallback'>"]
        for i, chunk in enumerate(state_chunks, 1):
            text = getattr(chunk, "text", str(chunk))[:1200]
            doc_id = getattr(chunk, "doc_id", f"chunk-{i}")
            fallback_lines.append(f"  <chunk rank='{i}' doc_id='{doc_id}'><text>{text}</text></chunk>")
        fallback_lines.append("</retrieved_context>")
        retrieved_xml = "\n".join(fallback_lines)

    ws_events.append(retrieval_done_event(session_id, len(hits)))

    # 2. Graph feature analysis section (if available)
    node_features = state.get("node_features", {})
    graph_features_section = _format_node_features(node_features)

    # 3. LLM synthesis — pass XML as <retrieved_context> block
    llm = ChatGroq(
        api_key=os.environ.get("GROQ_API_KEY"),
        model="llama-3.3-70b-versatile",
        temperature=0.15,
        max_tokens=8192,
    )

    system_prompt_content = EPRS_SYSTEM_PROMPT
    if graph_features_section:
        system_prompt_content = EPRS_SYSTEM_PROMPT + "\n\n" + graph_features_section

    messages = [
        SystemMessage(content=system_prompt_content),
        HumanMessage(
            content=f"""Topic of Investigation: {state['topic']}

<retrieved_context>
{retrieved_xml}
</retrieved_context>

Write a complete, academically rigorous EPRS Intelligence Analysis report following all structural requirements in your system instructions. Minimum 1800 words. All sections mandatory. Cite claims with [Doc-N] referencing chunk doc_id values from the retrieved context."""
        ),
    ]
    response = await llm.ainvoke(messages)
    briefing = str(response.content)

    ws_events.append(brief_done_event(session_id, briefing))

    # Export intelligence brief HTML for observability
    try:
        from services.agent.graph.html_report import export_brief_html
        export_state = dict(state)
        export_state["briefing"] = briefing
        export_state["ws_events"] = ws_events
        export_brief_html(export_state)  # type: ignore[arg-type]
    except Exception:
        pass

    return {
        "briefing": briefing,
        "retrieval_hits": hits,
        "ws_events": ws_events,
    }


def make_communicate_node():
    """Factory: return a communicate_node closure.

    All retrieval is performed via the hybridrag HTTP service (HYBRIDRAG_URL env var).
    No dependency injection required.

    Returns:
        Async callable compatible with LangGraph node signature.
    """

    async def _communicate_node(state: InvestigativeState) -> dict[str, Any]:
        return await communicate_node(state)

    return _communicate_node

"""LangGraph StateGraph for the ADR-002 3-phase macro graph."""
from __future__ import annotations

from typing import Any

from langgraph.graph import END, StateGraph

from services.agent.graph.nodes.communicate import make_communicate_node
from services.agent.graph.nodes.investigate import make_investigate_node
from services.agent.graph.nodes.scope import scope_node
from services.agent.graph.state import InvestigativeState


def should_continue(state: InvestigativeState) -> str:
    """Determine whether the investigate loop should continue or exit.

    The loop stops when either:
    - iteration_count reaches the hard cap of 12, or
    - dup_ratio exceeds 0.6 (majority of new chunks are near-duplicates).

    Args:
        state: Current investigative state.

    Returns:
        ``"done"`` to transition to communicate, ``"continue"`` to loop.
    """
    if state["iteration_count"] >= 12:
        return "done"
    if state["dup_ratio"] > 0.6:
        return "done"
    return "continue"


def build_workflow(
    pg_client: Any | None = None,
    mg_driver: Any | None = None,
    search_client: Any | None = None,
    backend: Any | None = None,
    ontology_router: Any | None = None,
    rel_extractor: Any | None = None,
    tavily: Any | None = None,
    cellar: Any | None = None,
    extractor: Any | None = None,
) -> StateGraph:
    """Build the 3-phase macro StateGraph.

    Args:
        pg_client: Optional PgVectorClient for hybrid retrieval wiring.
        mg_driver: Optional Memgraph driver for hybrid retrieval wiring.
        search_client: Optional SearchServiceClient for real document fetching.
        backend: Optional GraphBackend for swappable session graph implementation.
        ontology_router: Optional OntologyRouter for domain-adaptive extraction.
        rel_extractor: Optional OntologyRelExtractor for typed ONTOLOGY_REL edges.

    Returns:
        A compiled StateGraph with scope -> investigate -> communicate flow.
    """
    inv_node = make_investigate_node(
        pg_client, mg_driver, search_client, backend,
        ontology_router=ontology_router,
        rel_extractor=rel_extractor,
        tavily=tavily,
        cellar=cellar,
        extractor=extractor,
    )
    comm_node = make_communicate_node(pg_client=pg_client, mg_driver=mg_driver)

    workflow = StateGraph(InvestigativeState)
    workflow.add_node("scope", scope_node)
    workflow.add_node("investigate", inv_node)
    workflow.add_node("communicate", comm_node)
    workflow.set_entry_point("scope")
    workflow.add_edge("scope", "investigate")
    workflow.add_conditional_edges(
        "investigate",
        should_continue,
        {"continue": "investigate", "done": "communicate"},
    )
    workflow.add_edge("communicate", END)
    return workflow.compile()

# Events to stream from the LangGraph agent
REQUIRED_STREAM_EVENTS = ("on_chat_model_stream", "on_tool_start", "on_tool_end", "on_chain_end")

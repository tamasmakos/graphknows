"""Dict-based WSEvent factories for the ADR-002 investigative workflow."""
from __future__ import annotations

import time

from services.agent.graph.state import SubQuestion, WSEvent


def _ts() -> float:
    return time.time()


def plan_ready_event(session_id: str, plan: list[SubQuestion]) -> WSEvent:
    """Emit when the scope phase produces a research plan."""
    return WSEvent(
        session_id=session_id,
        event_type="plan_ready",
        payload={"plan": [{"id": sq.id, "text": sq.text, "source_priorities": sq.source_priorities} for sq in plan]},
        ts=_ts(),
    )


def search_start_event(session_id: str, iteration: int, queries: list[str]) -> WSEvent:
    """Emit when a new search iteration begins."""
    return WSEvent(
        session_id=session_id,
        event_type="search_start",
        payload={"iteration": iteration, "queries": queries},
        ts=_ts(),
    )


def docs_fetched_event(session_id: str, iteration: int, count: int) -> WSEvent:
    """Emit after raw documents are fetched."""
    return WSEvent(
        session_id=session_id,
        event_type="docs_fetched",
        payload={"iteration": iteration, "count": count},
        ts=_ts(),
    )


def chunks_written_event(session_id: str, iteration: int, count: int) -> WSEvent:
    """Emit after chunks are written to the vector store."""
    return WSEvent(
        session_id=session_id,
        event_type="chunks_written",
        payload={"iteration": iteration, "count": count},
        ts=_ts(),
    )


def entities_written_event(session_id: str, iteration: int, entity_count: int, relation_count: int) -> WSEvent:
    """Emit after entities and relations are written to the graph store."""
    return WSEvent(
        session_id=session_id,
        event_type="entities_written",
        payload={"iteration": iteration, "entity_count": entity_count, "relation_count": relation_count},
        ts=_ts(),
    )


def novelty_check_event(session_id: str, iteration: int, dup_ratio: float) -> WSEvent:
    """Emit after a novelty/duplication check completes."""
    return WSEvent(
        session_id=session_id,
        event_type="novelty_check",
        payload={"iteration": iteration, "dup_ratio": dup_ratio},
        ts=_ts(),
    )


def search_done_event(session_id: str, reason: str, iterations: int) -> WSEvent:
    """Emit when the search loop exits."""
    return WSEvent(
        session_id=session_id,
        event_type="search_done",
        payload={"reason": reason, "iterations": iterations},
        ts=_ts(),
    )


def retrieval_done_event(session_id: str, count: int) -> WSEvent:
    """Emit after hybrid retrieval completes."""
    return WSEvent(
        session_id=session_id,
        event_type="retrieval_done",
        payload={"count": count},
        ts=_ts(),
    )


def brief_chunk_event(session_id: str, token: str) -> WSEvent:
    """Emit for each token streamed during brief generation."""
    return WSEvent(
        session_id=session_id,
        event_type="brief_chunk",
        payload={"token": token},
        ts=_ts(),
    )


def brief_done_event(session_id: str, briefing: str) -> WSEvent:
    """Emit when the briefing is complete."""
    return WSEvent(
        session_id=session_id,
        event_type="brief_done",
        payload={"briefing": briefing},
        ts=_ts(),
    )


def graph_update_event(
    session_id: str,
    iteration: int,
    entities: list[dict],
    relations: list[dict],
) -> WSEvent:
    """Emit live entity/relation data so the frontend can update the graph in real time.

    Args:
        session_id: Active session identifier.
        iteration: Current investigation iteration index.
        entities: List of dicts with keys ``id``, ``name``, ``label``.
        relations: List of dicts with keys ``source``, ``target``, ``type``.
    """
    return WSEvent(
        session_id=session_id,
        event_type="graph_update",
        payload={
            "iteration": iteration,
            "nodes": entities,
            "edges": relations,
        },
        ts=_ts(),
    )


def relations_extracted_event(
    session_id: str,
    iteration: int,
    count: int,
    ontology_id: str,
    ontology_name: str,
) -> WSEvent:
    """Emit after ontology-guided relation extraction completes."""
    return WSEvent(
        session_id=session_id,
        event_type="relations_extracted",
        payload={
            "iteration": iteration,
            "count": count,
            "ontology_id": ontology_id,
            "ontology_name": ontology_name,
        },
        ts=_ts(),
    )


def error_event(session_id: str, message: str) -> WSEvent:
    """Emit when an error occurs during workflow execution."""
    return WSEvent(
        session_id=session_id,
        event_type="error",
        payload={"message": message},
        ts=_ts(),
    )


# ---------------------------------------------------------------------------
# Pydantic SSE event models for the /investigate endpoint
# ---------------------------------------------------------------------------
from typing import Literal, Optional  # noqa: E402
from pydantic import BaseModel  # noqa: E402


class PhaseChangeEvent(BaseModel):
    """Emitted when the investigation transitions between macro phases."""
    event: Literal["phase_change"] = "phase_change"
    session_id: str
    phase: str
    node: str


class ThreadStartedEvent(BaseModel):
    """Emitted when a new investigation thread starts."""
    event: Literal["thread_started"] = "thread_started"
    session_id: str
    thread_id: str
    title: str
    sub_question_id: str


class EvidenceAddedEvent(BaseModel):
    """Emitted when a new evidence chunk is added."""
    event: Literal["evidence_added"] = "evidence_added"
    session_id: str
    evidence_id: str
    thread_id: str
    summary: str
    source_authority: int = 3
    document_url: Optional[str] = None


class GraphBuildingEvent(BaseModel):
    """Emitted when graph construction starts."""
    event: Literal["graph_building"] = "graph_building"
    session_id: str
    namespace: str


class GraphReadyEvent(BaseModel):
    """Emitted when the session graph is ready."""
    event: Literal["graph_ready"] = "graph_ready"
    session_id: str
    namespace: str
    node_count: int
    edge_count: int


class ClaimVerifiedEvent(BaseModel):
    """Emitted when a claim is verified by the report."""
    event: Literal["claim_verified"] = "claim_verified"
    session_id: str
    claim_id: str
    text: str
    status: str = "supported"
    importance: int = 5


class ContradictionDetectedEvent(BaseModel):
    """Emitted when a contradiction is detected."""
    event: Literal["contradiction_detected"] = "contradiction_detected"
    session_id: str
    contradiction_id: str
    claim_id: str
    summary: str
    resolution_status: str = "open"


class ReportReadyEvent(BaseModel):
    """Emitted when the final report is ready."""
    event: Literal["report_ready"] = "report_ready"
    session_id: str
    findings_count: int
    report_excerpt: str


def graph_export_event(session_id: str, path: str, n_nodes: int, n_edges: int) -> WSEvent:
    """Emit after a session graph HTML visualisation is written to disk."""
    return WSEvent(
        session_id=session_id,
        event_type="graph_export",
        payload={"path": path, "n_nodes": n_nodes, "n_edges": n_edges},
        ts=_ts(),
    )


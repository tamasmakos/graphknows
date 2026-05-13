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


def error_event(session_id: str, message: str) -> WSEvent:
    """Emit when an error occurs during workflow execution."""
    return WSEvent(
        session_id=session_id,
        event_type="error",
        payload={"message": message},
        ts=_ts(),
    )

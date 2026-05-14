"""Typed state contracts for the ADR-002 3-phase macro graph."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, TypedDict


@dataclass
class SubQuestion:
    """A decomposed sub-question within the research plan."""

    id: str
    text: str
    source_priorities: list[str] = field(default_factory=lambda: ["eprs_briefings", "ep_debates", "web"])


@dataclass
class RawDoc:
    """A raw document fetched during a search iteration."""

    id: str
    title: str
    text: str
    url: str
    source: str
    date: str | None


@dataclass
class Chunk:
    """A chunked and embedded passage from a raw document."""

    id: str
    doc_id: str
    text: str
    embedding: list[float] | None = None
    metadata: dict = field(default_factory=dict)


@dataclass
class Entity:
    """An extracted entity from investigated documents."""

    id: str
    name: str
    label: str
    confidence: float
    doc_ids: list[str] = field(default_factory=list)


@dataclass
class Relation:
    """An extracted relation between two entities."""

    source: str
    target: str
    type: str
    confidence: float


class WSEvent(TypedDict):
    """WebSocket event emitted during workflow execution."""

    session_id: str
    event_type: Literal[
        "plan_ready", "search_start", "docs_fetched", "chunks_written",
        "entities_written", "novelty_check", "search_done", "retrieval_done",
        "brief_chunk", "brief_done", "error",
    ]
    payload: dict
    ts: float


class InvestigativeState(TypedDict):
    """Macro-phase state for the 3-phase scope/investigate/communicate workflow."""

    session_id: str
    topic: str
    plan: list[SubQuestion]
    queries: list[str]
    raw_docs: list[RawDoc]
    chunks: list[Chunk]
    entities: list[Entity]
    relations: list[Relation]
    iteration_count: int
    dup_ratio: float
    retrieval_hits: list[Chunk]
    briefing: str | None
    ws_events: list[WSEvent]
    node_features: dict
    current_macro_phase: str


def build_initial_investigative_state(session_id: str, topic: str) -> InvestigativeState:
    """Create the initial investigative state for a new session.

    Args:
        session_id: Unique session identifier.
        topic: The research topic or question to investigate.

    Returns:
        A minimally populated InvestigativeState ready for the scope phase.
    """
    return InvestigativeState(
        session_id=session_id,
        topic=topic,
        plan=[],
        queries=[],
        raw_docs=[],
        chunks=[],
        entities=[],
        relations=[],
        iteration_count=0,
        dup_ratio=0.0,
        retrieval_hits=[],
        briefing=None,
        ws_events=[],
        node_features={},
        current_macro_phase="scope",
    )

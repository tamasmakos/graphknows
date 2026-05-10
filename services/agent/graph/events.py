"""Typed stream-event payloads for the investigative agent workflow."""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel


class PhaseChangeEvent(BaseModel):
    event: Literal["phase_change"] = "phase_change"
    session_id: str
    phase: str
    node: str


class ThreadStartedEvent(BaseModel):
    event: Literal["thread_started"] = "thread_started"
    session_id: str
    thread_id: str
    title: str
    sub_question_id: str


class EvidenceAddedEvent(BaseModel):
    event: Literal["evidence_added"] = "evidence_added"
    session_id: str
    evidence_id: str
    thread_id: str
    summary: str
    source_authority: int
    document_url: Optional[str] = None


class GraphBuildingEvent(BaseModel):
    event: Literal["graph_building"] = "graph_building"
    session_id: str
    namespace: str


class GraphReadyEvent(BaseModel):
    event: Literal["graph_ready"] = "graph_ready"
    session_id: str
    namespace: str
    node_count: int
    edge_count: int


class ClaimVerifiedEvent(BaseModel):
    event: Literal["claim_verified"] = "claim_verified"
    session_id: str
    claim_id: str
    text: str
    status: str
    importance: int


class ContradictionDetectedEvent(BaseModel):
    event: Literal["contradiction_detected"] = "contradiction_detected"
    session_id: str
    contradiction_id: str
    claim_id: str
    summary: str
    resolution_status: str


class ReportReadyEvent(BaseModel):
    event: Literal["report_ready"] = "report_ready"
    session_id: str
    findings_count: int
    report_excerpt: str

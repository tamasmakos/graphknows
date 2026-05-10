"""Tests for investigative state contracts and stream-event payloads."""
from __future__ import annotations

import json
import sys
import types


if "services" not in sys.modules:
    import agent as _agent

    services_mod = types.ModuleType("services")
    services_mod.agent = _agent
    sys.modules["services"] = services_mod
    sys.modules["services.agent"] = _agent


from agent.graph.events import (
    ClaimVerifiedEvent,
    ContradictionDetectedEvent,
    EvidenceAddedEvent,
    GraphBuildingEvent,
    GraphReadyEvent,
    PhaseChangeEvent,
    ReportReadyEvent,
    ThreadStartedEvent,
)
from agent.graph.state import build_initial_investigative_state
from agent.graph.workflow import REQUIRED_STREAM_EVENTS, build_workflow_spec


def test_all_required_events_have_model():
    event_classes = {
        "phase_change": PhaseChangeEvent,
        "thread_started": ThreadStartedEvent,
        "evidence_added": EvidenceAddedEvent,
        "graph_building": GraphBuildingEvent,
        "graph_ready": GraphReadyEvent,
        "claim_verified": ClaimVerifiedEvent,
        "contradiction_detected": ContradictionDetectedEvent,
        "report_ready": ReportReadyEvent,
    }
    for name in REQUIRED_STREAM_EVENTS:
        assert name in event_classes, f"Missing model for event {name!r}"


def test_phase_change_event_serialises():
    ev = PhaseChangeEvent(session_id="s1", phase="scope", node="scope_frame")
    payload = json.loads(ev.model_dump_json())
    assert payload["event"] == "phase_change"
    assert payload["phase"] == "scope"


def test_thread_started_event_serialises():
    ev = ThreadStartedEvent(
        session_id="s1",
        thread_id="t1",
        title="Test thread",
        sub_question_id="sq_0",
    )
    payload = json.loads(ev.model_dump_json())
    assert payload["event"] == "thread_started"
    assert payload["thread_id"] == "t1"


def test_evidence_added_event_optional_url():
    ev = EvidenceAddedEvent(
        session_id="s1",
        evidence_id="ev_0",
        thread_id="t1",
        summary="Some evidence text",
        source_authority=3,
    )
    payload = json.loads(ev.model_dump_json())
    assert payload["event"] == "evidence_added"
    assert payload["document_url"] is None


def test_graph_building_event_serialises():
    ev = GraphBuildingEvent(session_id="s1", namespace="s1")
    payload = json.loads(ev.model_dump_json())
    assert payload["event"] == "graph_building"


def test_graph_ready_event_serialises():
    ev = GraphReadyEvent(session_id="s1", namespace="s1", node_count=5, edge_count=4)
    payload = json.loads(ev.model_dump_json())
    assert payload["node_count"] == 5


def test_claim_verified_event_serialises():
    ev = ClaimVerifiedEvent(
        session_id="s1",
        claim_id="cl_0",
        text="The Commission proposed X.",
        status="supported",
        importance=5,
    )
    payload = json.loads(ev.model_dump_json())
    assert payload["status"] == "supported"


def test_contradiction_detected_event_serialises():
    ev = ContradictionDetectedEvent(
        session_id="s1",
        contradiction_id="con_0",
        claim_id="cl_0",
        summary="Evidence conflicts on date.",
        resolution_status="open",
    )
    payload = json.loads(ev.model_dump_json())
    assert payload["resolution_status"] == "open"


def test_report_ready_event_serialises():
    ev = ReportReadyEvent(session_id="s1", findings_count=3, report_excerpt="Summary here.")
    payload = json.loads(ev.model_dump_json())
    assert payload["findings_count"] == 3


def test_investigative_state_phases_covered_by_workflow():
    spec = build_workflow_spec()
    phases_in_spec = {node.phase for node in spec}
    assert phases_in_spec == {"scope", "investigate", "communicate"}


def test_initial_state_fields():
    state = build_initial_investigative_state("sid1", "What is X?")
    assert state["current_macro_phase"] == "scope"
    assert state["evidence_ledger"] == []
    assert state["claims"] == []
    assert state["contradictions"] == []
    assert state["hypothesis_graph"]["status"] == "unseeded"

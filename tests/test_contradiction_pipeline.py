"""Tests for the contradiction tracking pipeline (Slice 4)."""
from __future__ import annotations

from services.agent.graph.nodes import (
    communicate_report,
    investigate_surface_contradictions,
    render_contradictions_section,
    route_after_contradictions,
)
from services.agent.graph.state import (
    ClaimRecord,
    ContradictionRecord,
    build_initial_investigative_state,
)


def _make_state_with_claims(claims: list[ClaimRecord], contradictions: list[ContradictionRecord] | None = None):
    state = build_initial_investigative_state("ses-1", "test query")
    state["claims"] = claims
    state["contradictions"] = contradictions or []
    return state


def _contradicted_claim(claim_id: str = "claim-1") -> ClaimRecord:
    return ClaimRecord(
        claim_id=claim_id,
        text="The EU adopted EDIS.",
        sub_question_id="sq-1",
        status="contradicted",
        importance=4,
        supporting_evidence_ids=[],
        contradicting_evidence_ids=["ev-2"],
        graph_path_ids=[],
        requires_document_support=True,
        rationale="Contradicted by evidence.",
    )


def _supported_claim(claim_id: str = "claim-2") -> ClaimRecord:
    return ClaimRecord(
        claim_id=claim_id,
        text="The Commission published a White Paper.",
        sub_question_id="sq-1",
        status="supported",
        importance=3,
        supporting_evidence_ids=["ev-1"],
        contradicting_evidence_ids=[],
        graph_path_ids=[],
        requires_document_support=True,
        rationale="Passage-backed.",
    )


def test_surface_contradictions_creates_records_for_contradicted_claims():
    state = _make_state_with_claims([_contradicted_claim()])
    result = investigate_surface_contradictions(state)

    assert len(result["contradictions"]) == 1
    record = result["contradictions"][0]
    assert record["claim_id"] == "claim-1"
    assert record["contradiction_id"] == "contradiction-claim-1"
    assert record["resolution_status"] == "needs_more_evidence"
    assert record["evidence_against_ids"] == ["ev-2"]
    assert record["evidence_for_ids"] == []


def test_surface_contradictions_skips_non_contradicted_claims():
    state = _make_state_with_claims([_supported_claim()])
    result = investigate_surface_contradictions(state)

    assert result["contradictions"] == []


def test_surface_contradictions_does_not_duplicate_existing_records():
    existing: ContradictionRecord = ContradictionRecord(
        contradiction_id="contradiction-claim-1",
        claim_id="claim-1",
        summary="already tracked",
        evidence_for_ids=[],
        evidence_against_ids=["ev-2"],
        resolution_status="open",
    )
    state = _make_state_with_claims([_contradicted_claim()], contradictions=[existing])
    result = investigate_surface_contradictions(state)

    assert len(result["contradictions"]) == 1
    assert result["contradictions"][0]["summary"] == "already tracked"


def test_route_after_contradictions_needs_more_evidence():
    contradiction: ContradictionRecord = ContradictionRecord(
        contradiction_id="c-1",
        claim_id="claim-1",
        summary="conflict",
        evidence_for_ids=[],
        evidence_against_ids=["ev-2"],
        resolution_status="needs_more_evidence",
    )
    state = _make_state_with_claims([], contradictions=[contradiction])
    assert route_after_contradictions(state) == "investigate_collect_thread_evidence"


def test_route_after_contradictions_open_routes_to_report():
    contradiction: ContradictionRecord = ContradictionRecord(
        contradiction_id="c-1",
        claim_id="claim-1",
        summary="conflict",
        evidence_for_ids=["ev-1"],
        evidence_against_ids=["ev-2"],
        resolution_status="open",
    )
    state = _make_state_with_claims([], contradictions=[contradiction])
    assert route_after_contradictions(state) == "communicate_report"


def test_route_after_contradictions_no_contradictions_routes_to_report():
    state = _make_state_with_claims([])
    assert route_after_contradictions(state) == "communicate_report"


def test_communicate_report_includes_contradictions_section():
    contradiction: ContradictionRecord = ContradictionRecord(
        contradiction_id="c-1",
        claim_id="claim-1",
        summary="Claim is contested.",
        evidence_for_ids=["ev-1"],
        evidence_against_ids=["ev-2"],
        resolution_status="open",
    )
    state = _make_state_with_claims([_supported_claim()], contradictions=[contradiction])
    result = communicate_report(state)

    assert "# Investigation Report" in result["report"]
    assert "## Verified Findings" in result["report"]
    assert "## Contradictions" in result["report"]
    assert "Claim is contested." in result["report"]
    assert result["findings"] == ["The Commission published a White Paper."]


def test_communicate_report_no_contradictions_section_when_clean():
    state = _make_state_with_claims([_supported_claim()])
    result = communicate_report(state)

    assert "# Investigation Report" in result["report"]
    assert "## Contradictions" not in result["report"]


def test_render_contradictions_section_empty():
    assert render_contradictions_section([]) == ""


def test_render_contradictions_section_formats_correctly():
    contradiction: ContradictionRecord = ContradictionRecord(
        contradiction_id="c-1",
        claim_id="claim-1",
        summary="EU and Council disagree on scope.",
        evidence_for_ids=["ev-1", "ev-2"],
        evidence_against_ids=["ev-3"],
        resolution_status="open",
    )
    section = render_contradictions_section([contradiction])

    assert "## Contradictions" in section
    assert "1 unresolved contradiction(s)" in section
    assert "EU and Council disagree on scope." in section
    assert "Supporting evidence: 2 item(s)" in section
    assert "Contradicting evidence: 1 item(s)" in section
    assert "Status: open" in section

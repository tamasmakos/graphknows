from __future__ import annotations

from services.agent.graph.state import (
    ClaimRecord,
    ContradictionRecord,
    EvidenceRecord,
    build_initial_investigative_state,
    claim_has_document_support,
    claim_has_graph_support,
    claim_is_publishable,
    format_contradiction_section,
    has_unresolved_contradictions,
    index_evidence_by_id,
    promote_writeback_outputs,
    select_publishable_claims,
    surface_contradictions_from_claims,
)
from services.agent.graph.workflow import REQUIRED_STREAM_EVENTS, node_names_by_phase


def _build_evidence_record(
    evidence_id: str,
    *,
    include_passage: bool,
    include_graph_path: bool,
) -> EvidenceRecord:
    return {
        "evidence_id": evidence_id,
        "thread_id": "thread-policy-text",
        "sub_question_id": "sq-1",
        "summary": "Relevant EPRS passage about the policy.",
        "source_authority": 5,
        "independence": 4,
        "directness": 5,
        "recency": 4,
        "claim_ids": ["claim-1"],
        "tags": ["official-record"],
        "provenance": {
            "retrieval": {
                "retrieval_id": "ret-1",
                "query": "EDIS readiness",
                "tool_name": "search_eu",
                "retrieved_at": "2026-05-10T18:00:00+00:00",
                "rank": 1,
                "score": 0.91,
                "source_type": "eprs_briefings",
            },
            "document": {
                "document_id": "doc-1",
                "title": "European defence industrial strategy",
                "url": "https://example.test/edis",
                "publisher": "European Parliament Think Tank",
                "published_at": "2024-03-05",
            },
            "passage": {
                "chunk_id": "chunk-1",
                "text": "The strategy links readiness to procurement and industrial capacity.",
                "start_char": 120,
                "end_char": 196,
            }
            if include_passage
            else None,
            "relation": {
                "relation_id": "rel-1",
                "source_entity": "European Union",
                "target_entity": "EDIS",
                "relation_type": "ADOPTED",
                "extraction_method": "graphgen-session-job",
                "confidence": 0.86,
            },
            "graph_path": {
                "path_id": "path-1",
                "entity_names": ["European Union", "EDIS"],
                "edge_types": ["ADOPTED"],
                "supporting_chunk_ids": ["chunk-1"],
                "hypothesis_only": False,
            }
            if include_graph_path
            else None,
        },
    }


def test_build_initial_investigative_state_starts_with_bounded_hypothesis_graph():
    state = build_initial_investigative_state(
        session_id="session-123",
        user_query="How did the Commission frame EDIS?",
    )

    assert state["current_macro_phase"] == "scope"
    assert state["evidence_ledger"] == []
    assert state["claims"] == []
    assert state["contradictions"] == []
    assert state["hypothesis_graph"] == {
        "status": "unseeded",
        "graph_task_id": None,
        "graph_namespace": "session-123",
        "seeded_from_corpus": False,
        "non_authoritative": True,
        "last_updated_at": None,
    }


def test_claim_publishability_requires_document_support_for_important_claims():
    graph_only = _build_evidence_record(
        "evidence-graph-only",
        include_passage=False,
        include_graph_path=True,
    )
    hybrid = _build_evidence_record(
        "evidence-hybrid",
        include_passage=True,
        include_graph_path=True,
    )
    evidence_index = index_evidence_by_id([graph_only, hybrid])

    graph_only_claim: ClaimRecord = {
        "claim_id": "claim-graph-only",
        "text": "The EU adopted EDIS.",
        "sub_question_id": "sq-1",
        "status": "supported",
        "importance": 5,
        "supporting_evidence_ids": ["evidence-graph-only"],
        "contradicting_evidence_ids": [],
        "graph_path_ids": ["path-1"],
        "requires_document_support": True,
        "rationale": "Important claims require passages, not just graph structure.",
    }
    hybrid_claim: ClaimRecord = {
        "claim_id": "claim-hybrid",
        "text": "The EU adopted EDIS.",
        "sub_question_id": "sq-1",
        "status": "supported",
        "importance": 5,
        "supporting_evidence_ids": ["evidence-hybrid"],
        "contradicting_evidence_ids": [],
        "graph_path_ids": ["path-1"],
        "requires_document_support": True,
        "rationale": "Important claims require passages, not just graph structure.",
    }

    assert claim_has_graph_support(graph_only_claim, evidence_index) is True
    assert claim_has_document_support(graph_only_claim, evidence_index) is False
    assert claim_is_publishable(graph_only_claim, evidence_index) is False
    assert claim_is_publishable(hybrid_claim, evidence_index) is True


def test_workflow_prototype_exposes_parallel_threads_and_verification_nodes():
    nodes = node_names_by_phase()

    assert nodes["scope"] == ("scope_frame", "scope_map", "scope_plan")
    assert "investigate_dispatch_threads" in nodes["investigate"]
    assert "investigate_collect_thread_evidence" in nodes["investigate"]
    assert "investigate_verify_claims" in nodes["investigate"]
    assert "investigate_surface_contradictions" in nodes["investigate"]
    assert nodes["communicate"] == ("communicate_report", "communicate_writeback")
    assert REQUIRED_STREAM_EVENTS == (
        "phase_change",
        "thread_started",
        "evidence_added",
        "graph_building",
        "graph_ready",
        "claim_verified",
        "contradiction_detected",
        "report_ready",
    )


def test_provenance_envelope_carries_all_required_fields():
    """ProvenanceEnvelope keys exist and have expected types on a full evidence record."""
    record = _build_evidence_record(
        "evidence-full",
        include_passage=True,
        include_graph_path=True,
    )
    prov = record["provenance"]

    # retrieval fields
    assert prov["retrieval"]["retrieval_id"] == "ret-1"
    assert prov["retrieval"]["tool_name"] == "search_eu"
    assert isinstance(prov["retrieval"]["score"], float)

    # document fields
    assert prov["document"]["document_id"] == "doc-1"
    assert prov["document"]["publisher"] == "European Parliament Think Tank"

    # passage fields
    assert prov["passage"] is not None
    assert prov["passage"]["chunk_id"] == "chunk-1"
    assert prov["passage"]["start_char"] < prov["passage"]["end_char"]

    # graph path fields
    assert prov["graph_path"] is not None
    assert prov["graph_path"]["hypothesis_only"] is False
    assert "European Union" in prov["graph_path"]["entity_names"]


def test_provenance_envelope_nullable_fields_are_none_when_excluded():
    record = _build_evidence_record(
        "evidence-sparse",
        include_passage=False,
        include_graph_path=False,
    )
    prov = record["provenance"]

    assert prov["passage"] is None
    assert prov["graph_path"] is None
    # relation is always set in helper; document is always set
    assert prov["document"] is not None
    assert prov["retrieval"] is not None


def test_hypothesis_graph_non_authoritative_flag_is_preserved_in_writeback():
    state = build_initial_investigative_state(
        session_id="session-wb",
        user_query="Test write-back semantics",
    )
    state["report"] = "Draft report."

    result = promote_writeback_outputs(state)

    meta = result["hypothesis_graph_metadata"]
    assert meta["non_authoritative"] is True
    assert meta["namespace"] == "session-wb"
    assert meta["seeded_from_corpus"] is False
    assert result["validated_claims"] == []
    assert result["report"] == "Draft report."


def test_promote_writeback_filters_unpublishable_claims():
    graph_only = _build_evidence_record(
        "evidence-graph-only",
        include_passage=False,
        include_graph_path=True,
    )
    hybrid = _build_evidence_record(
        "evidence-hybrid",
        include_passage=True,
        include_graph_path=True,
    )

    unpublishable: ClaimRecord = {
        "claim_id": "claim-unpublishable",
        "text": "Graph-only claim.",
        "sub_question_id": "sq-1",
        "status": "supported",
        "importance": 5,
        "supporting_evidence_ids": ["evidence-graph-only"],
        "contradicting_evidence_ids": [],
        "graph_path_ids": ["path-1"],
        "requires_document_support": True,
        "rationale": "Needs passage.",
    }
    publishable: ClaimRecord = {
        "claim_id": "claim-publishable",
        "text": "Hybrid claim.",
        "sub_question_id": "sq-1",
        "status": "supported",
        "importance": 5,
        "supporting_evidence_ids": ["evidence-hybrid"],
        "contradicting_evidence_ids": [],
        "graph_path_ids": [],
        "requires_document_support": True,
        "rationale": "Has passage.",
    }

    state = build_initial_investigative_state("session-filt", "filter test")
    state["evidence_ledger"] = [graph_only, hybrid]
    state["claims"] = [unpublishable, publishable]
    state["report"] = "Final report."

    result = promote_writeback_outputs(state)

    ids = [c["claim_id"] for c in result["validated_claims"]]
    assert "claim-publishable" in ids
    assert "claim-unpublishable" not in ids


def test_surface_contradictions_produces_open_records():
    evidence = _build_evidence_record(
        "e1", include_passage=True, include_graph_path=False
    )
    contested_claim: ClaimRecord = {
        "claim_id": "claim-contested",
        "text": "The EU adopted EDIS.",
        "sub_question_id": "sq-1",
        "status": "contradicted",
        "importance": 4,
        "supporting_evidence_ids": ["e1"],
        "contradicting_evidence_ids": ["e1"],  # same id for test simplicity
        "graph_path_ids": [],
        "requires_document_support": False,
        "rationale": "Both sides cite same record in this test.",
    }

    contradictions = surface_contradictions_from_claims([contested_claim], [evidence])

    assert len(contradictions) == 1
    assert contradictions[0]["claim_id"] == "claim-contested"
    assert contradictions[0]["resolution_status"] == "open"
    assert "e1" in contradictions[0]["evidence_for_ids"]


# ---------------------------------------------------------------------------
# Contradiction tracking helpers
# ---------------------------------------------------------------------------


def _build_claim(
    claim_id: str,
    *,
    supporting: list[str],
    contradicting: list[str],
    status: str = "contradicted",
) -> ClaimRecord:
    return {
        "claim_id": claim_id,
        "text": f"Claim text for {claim_id}.",
        "sub_question_id": "sq-1",
        "status": status,
        "importance": 3,
        "supporting_evidence_ids": supporting,
        "contradicting_evidence_ids": contradicting,
        "graph_path_ids": [],
        "requires_document_support": False,
        "rationale": "",
    }


def test_surface_contradictions_returns_contradiction_per_conflicted_claim():
    e1 = _build_evidence_record("ev-for", include_passage=True, include_graph_path=False)
    e2 = _build_evidence_record("ev-against", include_passage=False, include_graph_path=False)
    claim_with_conflict = _build_claim("cl-1", supporting=["ev-for"], contradicting=["ev-against"])
    claim_clean = _build_claim("cl-2", supporting=["ev-for"], contradicting=[])

    result = surface_contradictions_from_claims([claim_with_conflict, claim_clean], [e1, e2])

    assert len(result) == 1
    c = result[0]
    assert c["claim_id"] == "cl-1"
    assert c["resolution_status"] == "open"
    assert "ev-for" in c["evidence_for_ids"]
    assert "ev-against" in c["evidence_against_ids"]


def test_surface_contradictions_skips_dangling_evidence_ids():
    e1 = _build_evidence_record("ev-real", include_passage=True, include_graph_path=False)
    claim = _build_claim("cl-dangle", supporting=["ev-real"], contradicting=["ev-nonexistent"])

    result = surface_contradictions_from_claims([claim], [e1])

    # "ev-nonexistent" is not in the ledger -> no valid against_ids -> no contradiction
    assert result == []


def test_has_unresolved_contradictions_routing_predicate():
    open_c: ContradictionRecord = {
        "contradiction_id": "c-1",
        "claim_id": "cl-1",
        "summary": "...",
        "evidence_for_ids": ["a"],
        "evidence_against_ids": ["b"],
        "resolution_status": "open",
    }
    explained_c: ContradictionRecord = {
        "contradiction_id": "c-2",
        "claim_id": "cl-1",
        "summary": "...",
        "evidence_for_ids": ["a"],
        "evidence_against_ids": ["b"],
        "resolution_status": "explained",
    }
    needs_more_c: ContradictionRecord = {
        "contradiction_id": "c-3",
        "claim_id": "cl-1",
        "summary": "...",
        "evidence_for_ids": ["a"],
        "evidence_against_ids": ["b"],
        "resolution_status": "needs_more_evidence",
    }

    assert has_unresolved_contradictions([open_c]) is True
    assert has_unresolved_contradictions([needs_more_c]) is True
    assert has_unresolved_contradictions([explained_c]) is False
    assert has_unresolved_contradictions([]) is False
    assert has_unresolved_contradictions([explained_c, open_c]) is True


def test_format_contradiction_section_omits_explained_and_includes_open():
    e1 = _build_evidence_record("ev-for", include_passage=True, include_graph_path=False)
    e2 = _build_evidence_record("ev-against", include_passage=False, include_graph_path=False)
    evidence_index = index_evidence_by_id([e1, e2])

    open_c: ContradictionRecord = {
        "contradiction_id": "c-open",
        "claim_id": "cl-1",
        "summary": "Conflicting signals on policy adoption.",
        "evidence_for_ids": ["ev-for"],
        "evidence_against_ids": ["ev-against"],
        "resolution_status": "open",
    }
    explained_c: ContradictionRecord = {
        "contradiction_id": "c-explained",
        "claim_id": "cl-2",
        "summary": "Already resolved.",
        "evidence_for_ids": [],
        "evidence_against_ids": [],
        "resolution_status": "explained",
    }

    section = format_contradiction_section([open_c, explained_c], evidence_index)

    assert "## Unresolved Contradictions" in section
    assert "Conflicting signals on policy adoption." in section
    assert "Already resolved." not in section
    assert "Relevant EPRS passage" in section


def test_format_contradiction_section_returns_empty_when_all_resolved():
    explained_c: ContradictionRecord = {
        "contradiction_id": "c-exp",
        "claim_id": "cl-1",
        "summary": "...",
        "evidence_for_ids": [],
        "evidence_against_ids": [],
        "resolution_status": "explained",
    }
    result = format_contradiction_section([explained_c], {})
    assert result == ""

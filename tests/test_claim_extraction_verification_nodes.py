"""Tests for investigate_extract_claims and investigate_verify_claims nodes (Slice 3)."""
from __future__ import annotations

from services.agent.graph.nodes import investigate_extract_claims, investigate_verify_claims
from services.agent.graph.state import (
    ClaimRecord,
    EvidenceRecord,
    build_initial_investigative_state,
    index_evidence_by_id,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_evidence(
    evidence_id: str,
    claim_ids: list[str],
    *,
    include_passage: bool,
    include_graph_path: bool,
    source_authority: int = 3,
) -> EvidenceRecord:
    return {
        "evidence_id": evidence_id,
        "thread_id": "thread-1",
        "sub_question_id": "sq-1",
        "summary": f"Evidence {evidence_id}.",
        "source_authority": source_authority,
        "independence": 3,
        "directness": 3,
        "recency": 3,
        "claim_ids": claim_ids,
        "tags": [],
        "provenance": {
            "retrieval": {
                "retrieval_id": f"ret-{evidence_id}",
                "query": "test query",
                "tool_name": "search_eu",
                "retrieved_at": "2026-05-10T18:00:00+00:00",
                "rank": 1,
                "score": 0.9,
                "source_type": "eprs_briefings",
            },
            "document": {
                "document_id": f"doc-{evidence_id}",
                "title": "Test doc",
                "url": "https://example.test",
                "publisher": "EP",
                "published_at": "2024-01-01",
            },
            "passage": {
                "chunk_id": f"chunk-{evidence_id}",
                "text": "Passage text.",
                "start_char": 0,
                "end_char": 12,
            }
            if include_passage
            else None,
            "relation": None,
            "graph_path": {
                "path_id": f"path-{evidence_id}",
                "entity_names": ["A", "B"],
                "edge_types": ["REL"],
                "supporting_chunk_ids": [],
                "hypothesis_only": False,
            }
            if include_graph_path
            else None,
        },
    }


# ---------------------------------------------------------------------------
# investigate_extract_claims
# ---------------------------------------------------------------------------


def test_extract_claims_empty_ledger_returns_empty_claims():
    state = build_initial_investigative_state("s1", "query")
    result = investigate_extract_claims(state)
    assert result["claims"] == []


def test_extract_claims_produces_one_claim_per_unique_claim_id():
    e1 = _make_evidence("ev-1", ["claim-a", "claim-b"], include_passage=True, include_graph_path=False)
    e2 = _make_evidence("ev-2", ["claim-a"], include_passage=False, include_graph_path=True)

    state = build_initial_investigative_state("s2", "query")
    state["evidence_ledger"] = [e1, e2]

    result = investigate_extract_claims(state)
    claims = result["claims"]

    claim_ids = {c["claim_id"] for c in claims}
    assert claim_ids == {"claim-a", "claim-b"}


def test_extract_claims_aggregates_evidence_ids_per_claim():
    e1 = _make_evidence("ev-1", ["claim-a"], include_passage=True, include_graph_path=False)
    e2 = _make_evidence("ev-2", ["claim-a"], include_passage=False, include_graph_path=False)

    state = build_initial_investigative_state("s3", "query")
    state["evidence_ledger"] = [e1, e2]

    result = investigate_extract_claims(state)
    claims = result["claims"]

    assert len(claims) == 1
    claim = claims[0]
    assert set(claim["supporting_evidence_ids"]) == {"ev-1", "ev-2"}


def test_extract_claims_requires_document_support_when_authority_high():
    high_auth = _make_evidence(
        "ev-high", ["claim-x"], include_passage=True, include_graph_path=False, source_authority=5
    )
    low_auth = _make_evidence(
        "ev-low", ["claim-y"], include_passage=True, include_graph_path=False, source_authority=2
    )

    state = build_initial_investigative_state("s4", "query")
    state["evidence_ledger"] = [high_auth, low_auth]

    result = investigate_extract_claims(state)
    by_id = {c["claim_id"]: c for c in result["claims"]}

    assert by_id["claim-x"]["requires_document_support"] is True
    assert by_id["claim-y"]["requires_document_support"] is False


def test_extract_claims_initial_status_is_unverified():
    e = _make_evidence("ev-1", ["claim-1"], include_passage=True, include_graph_path=False)
    state = build_initial_investigative_state("s5", "query")
    state["evidence_ledger"] = [e]

    result = investigate_extract_claims(state)
    assert result["claims"][0]["status"] == "unverified"


# ---------------------------------------------------------------------------
# investigate_verify_claims
# ---------------------------------------------------------------------------


def test_verify_claims_promotes_to_supported_when_passage_present():
    e = _make_evidence("ev-1", ["claim-1"], include_passage=True, include_graph_path=False)
    state = build_initial_investigative_state("s6", "query")
    state["evidence_ledger"] = [e]

    # Build a claim that requires document support and has passage-backed evidence.
    claim: ClaimRecord = {
        "claim_id": "claim-1",
        "text": "Claim 1.",
        "sub_question_id": "sq-1",
        "status": "unverified",
        "importance": 5,
        "supporting_evidence_ids": ["ev-1"],
        "contradicting_evidence_ids": [],
        "graph_path_ids": [],
        "requires_document_support": True,
        "rationale": "",
    }
    state["claims"] = [claim]

    result = investigate_verify_claims(state)
    assert result["claims"][0]["status"] == "supported"


def test_verify_claims_demotes_to_uncertain_when_graph_only_and_requires_doc():
    """Graph-only support must not satisfy high-importance claims."""
    e = _make_evidence(
        "ev-graph",
        ["claim-important"],
        include_passage=False,
        include_graph_path=True,
        source_authority=5,
    )
    state = build_initial_investigative_state("s7", "query")
    state["evidence_ledger"] = [e]

    claim: ClaimRecord = {
        "claim_id": "claim-important",
        "text": "Important claim backed only by graph.",
        "sub_question_id": "sq-1",
        "status": "unverified",
        "importance": 5,
        "supporting_evidence_ids": ["ev-graph"],
        "contradicting_evidence_ids": [],
        "graph_path_ids": ["path-ev-graph"],
        "requires_document_support": True,
        "rationale": "",
    }
    state["claims"] = [claim]

    result = investigate_verify_claims(state)
    assert result["claims"][0]["status"] == "uncertain"


def test_verify_claims_low_importance_supported_without_passage():
    """Low-importance claims (no document support required) pass with any evidence."""
    e = _make_evidence(
        "ev-any", ["claim-low"], include_passage=False, include_graph_path=False, source_authority=2
    )
    state = build_initial_investigative_state("s8", "query")
    state["evidence_ledger"] = [e]

    claim: ClaimRecord = {
        "claim_id": "claim-low",
        "text": "Low-importance claim.",
        "sub_question_id": "sq-1",
        "status": "unverified",
        "importance": 2,
        "supporting_evidence_ids": ["ev-any"],
        "contradicting_evidence_ids": [],
        "graph_path_ids": [],
        "requires_document_support": False,
        "rationale": "",
    }
    state["claims"] = [claim]

    result = investigate_verify_claims(state)
    assert result["claims"][0]["status"] == "supported"


def test_verify_claims_no_evidence_stays_unverified():
    state = build_initial_investigative_state("s9", "query")
    state["evidence_ledger"] = []

    claim: ClaimRecord = {
        "claim_id": "claim-empty",
        "text": "Claim with no evidence.",
        "sub_question_id": "sq-1",
        "status": "unverified",
        "importance": 3,
        "supporting_evidence_ids": [],
        "contradicting_evidence_ids": [],
        "graph_path_ids": [],
        "requires_document_support": False,
        "rationale": "",
    }
    state["claims"] = [claim]

    result = investigate_verify_claims(state)
    assert result["claims"][0]["status"] == "unverified"


def test_verify_claims_publishability_gate_rejects_graph_only():
    """End-to-end: extract then verify should mark graph-only claim uncertain."""
    e = _make_evidence(
        "ev-g",
        ["cl-g"],
        include_passage=False,
        include_graph_path=True,
        source_authority=4,
    )
    state = build_initial_investigative_state("s10", "query")
    state["evidence_ledger"] = [e]

    # Step 1: extract
    extracted = investigate_extract_claims(state)
    state["claims"] = extracted["claims"]

    # Step 2: verify
    verified = investigate_verify_claims(state)

    claim = verified["claims"][0]
    assert claim["claim_id"] == "cl-g"
    assert claim["requires_document_support"] is True
    assert claim["status"] == "uncertain"

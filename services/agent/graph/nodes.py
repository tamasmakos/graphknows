"""LangGraph node implementations for claim extraction, verification, and contradiction tracking.

Slices 3–4 of the evidence-led investigative workflow redesign.
Nodes are pure functions: they receive InvestigativeState and return a partial
state dict that LangGraph merges back into the shared state.
"""
from __future__ import annotations

import uuid
from typing import Any

from services.agent.graph.state import (
    ClaimRecord,
    ContradictionRecord,
    EvidenceRecord,
    InvestigativeState,
    claim_has_document_support,
    claim_has_graph_support,
    index_evidence_by_id,
)


def _new_claim_id() -> str:
    """Return a short deterministic-looking claim id."""
    return f"claim-{uuid.uuid4().hex[:8]}"


def investigate_extract_claims(state: InvestigativeState) -> dict[str, Any]:
    """Extract atomic claims from the gathered evidence ledger.

    Each evidence record carries a ``claim_ids`` field listing the claim
    identifiers that the evidence speaks to.  This node aggregates all
    evidence records per claim id and constructs a ``ClaimRecord`` stub for
    each unique claim.  The initial status is ``"unverified"`` — the
    ``investigate_verify_claims`` node promotes statuses downstream.

    High-importance evidence (source_authority >= 4) causes the derived claim
    to require document support before it can be published.

    Args:
        state: Current investigation state.

    Returns:
        Partial state dict with ``claims`` populated from evidence.
    """
    ledger: list[EvidenceRecord] = state["evidence_ledger"]
    if not ledger:
        return {"claims": []}

    # Group evidence ids by claim id declared in each evidence record.
    claim_to_evidence: dict[str, list[str]] = {}
    claim_to_sub_q: dict[str, str] = {}
    claim_to_max_authority: dict[str, int] = {}

    for record in ledger:
        for cid in record["claim_ids"]:
            claim_to_evidence.setdefault(cid, []).append(record["evidence_id"])
            claim_to_sub_q.setdefault(cid, record["sub_question_id"])
            prev = claim_to_max_authority.get(cid, 0)
            claim_to_max_authority[cid] = max(prev, record["source_authority"])

    claims: list[ClaimRecord] = []
    for cid, evidence_ids in claim_to_evidence.items():
        importance = claim_to_max_authority.get(cid, 3)
        requires_doc = importance >= 4
        claims.append(
            ClaimRecord(
                claim_id=cid,
                text=f"Claim extracted from evidence: {cid}",
                sub_question_id=claim_to_sub_q.get(cid, ""),
                status="unverified",
                importance=importance,
                supporting_evidence_ids=evidence_ids,
                contradicting_evidence_ids=[],
                graph_path_ids=[],
                requires_document_support=requires_doc,
                rationale="",
            )
        )

    return {"claims": claims}


def investigate_verify_claims(state: InvestigativeState) -> dict[str, Any]:
    """Verify each claim against the evidence ledger and hypothesis graph.

    Applies the publication grounding rule from ``claim_is_publishable``:
    - Claims that ``require_document_support`` must be backed by at least one
      evidence record carrying both a document reference and an exact passage.
      If the only support is a graph path or relation, the claim is demoted to
      ``"uncertain"`` rather than asserted as a fact.
    - Claims that do not require document support are marked ``"supported"``
      as long as they have at least one evidence record.
    - Claims with no evidence remain ``"unverified"``.

    Args:
        state: Current investigation state (claims and evidence_ledger must be set).

    Returns:
        Partial state dict with updated ``claims`` list.
    """
    claims: list[ClaimRecord] = state["claims"]
    ledger: list[EvidenceRecord] = state["evidence_ledger"]
    evidence_index = index_evidence_by_id(ledger)

    updated: list[ClaimRecord] = []
    for claim in claims:
        if not claim["supporting_evidence_ids"]:
            updated.append({**claim, "status": "unverified"})
            continue

        has_doc = claim_has_document_support(claim, evidence_index)
        has_graph = claim_has_graph_support(claim, evidence_index)

        if claim["requires_document_support"]:
            if has_doc:
                new_status: str = "supported"
            elif has_graph:
                # Graph-only support is not sufficient for high-importance claims.
                new_status = "uncertain"
            else:
                # Has evidence ids but none with document or graph provenance.
                new_status = "uncertain"
        else:
            # Low-importance claims: any evidence is sufficient.
            new_status = "supported"

        updated.append({**claim, "status": new_status})

    return {"claims": updated}

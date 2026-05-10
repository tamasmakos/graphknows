"""Typed state contracts for the evidence-led investigative workflow prototype."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal, TypedDict


MacroPhase = Literal["scope", "investigate", "communicate"]
ThreadStatus = Literal["pending", "running", "complete", "blocked"]
ClaimStatus = Literal["supported", "contradicted", "uncertain", "unverified"]
HypothesisGraphStatus = Literal["unseeded", "seeded", "building", "ready", "stale"]


class RetrievalEventRef(TypedDict):
    """Reference to the retrieval action that introduced evidence."""

    retrieval_id: str
    query: str
    tool_name: str
    retrieved_at: str
    rank: int
    score: float | None
    source_type: str


class DocumentRef(TypedDict):
    """Reference to the source document behind an evidence item."""

    document_id: str
    title: str
    url: str
    publisher: str
    published_at: str


class PassageRef(TypedDict):
    """Reference to the exact passage grounding an evidence item."""

    chunk_id: str
    text: str
    start_char: int
    end_char: int


class RelationRef(TypedDict):
    """Reference to an extracted relation and its extraction metadata."""

    relation_id: str
    source_entity: str
    target_entity: str
    relation_type: str
    extraction_method: str
    confidence: float | None


class GraphPathRef(TypedDict):
    """Reference to a graph path used during hybrid reasoning."""

    path_id: str
    entity_names: list[str]
    edge_types: list[str]
    supporting_chunk_ids: list[str]
    hypothesis_only: bool


class ProvenanceEnvelope(TypedDict):
    """Full provenance for an evidence item or verified claim."""

    retrieval: RetrievalEventRef
    document: DocumentRef
    passage: PassageRef | None
    relation: RelationRef | None
    graph_path: GraphPathRef | None


class EvidenceRecord(TypedDict):
    """An evidence-led ledger entry collected during the investigation."""

    evidence_id: str
    thread_id: str
    sub_question_id: str
    summary: str
    source_authority: int
    independence: int
    directness: int
    recency: int
    claim_ids: list[str]
    tags: list[str]
    provenance: ProvenanceEnvelope


class EvidenceThread(TypedDict):
    """A parallel evidence collection lane within the investigation."""

    thread_id: str
    title: str
    sub_question_id: str
    source_classes: list[str]
    status: ThreadStatus
    evidence_ids: list[str]
    notes: list[str]


class ClaimRecord(TypedDict):
    """A claim extracted from draft findings and sent through verification."""

    claim_id: str
    text: str
    sub_question_id: str
    status: ClaimStatus
    importance: int
    supporting_evidence_ids: list[str]
    contradicting_evidence_ids: list[str]
    graph_path_ids: list[str]
    requires_document_support: bool
    rationale: str


class ContradictionRecord(TypedDict):
    """A surfaced contradiction that must survive into final outputs."""

    contradiction_id: str
    claim_id: str
    summary: str
    evidence_for_ids: list[str]
    evidence_against_ids: list[str]
    resolution_status: Literal["open", "explained", "needs_more_evidence"]


class HypothesisGraph(TypedDict):
    """Session-scoped graph state used as a bounded hypothesis graph."""

    status: HypothesisGraphStatus
    graph_task_id: str | None
    graph_namespace: str
    seeded_from_corpus: bool
    non_authoritative: bool
    last_updated_at: str | None


class InvestigativeState(TypedDict):
    """Prototype LangGraph state for an evidence-led investigation."""

    session_id: str
    user_query: str
    current_macro_phase: MacroPhase
    messages: list[Any]
    hypothesis: str
    unit_of_analysis: str
    key_actors: list[str]
    evidence_plan: list[dict[str, Any]]
    evidence_threads: list[EvidenceThread]
    evidence_ledger: list[EvidenceRecord]
    claims: list[ClaimRecord]
    contradictions: list[ContradictionRecord]
    findings: list[str]
    report: str
    hypothesis_graph: HypothesisGraph


def utc_now_iso() -> str:
    """Return the current UTC timestamp in ISO-8601 format."""

    return datetime.now(timezone.utc).isoformat()


def build_initial_investigative_state(session_id: str, user_query: str) -> InvestigativeState:
    """Create the initial evidence-led investigation state.

    Args:
        session_id: Session identifier for the investigation.
        user_query: Original user question or briefing request.

    Returns:
        A minimally populated state ready for the scope phase.
    """

    return InvestigativeState(
        session_id=session_id,
        user_query=user_query,
        current_macro_phase="scope",
        messages=[],
        hypothesis="",
        unit_of_analysis="",
        key_actors=[],
        evidence_plan=[],
        evidence_threads=[],
        evidence_ledger=[],
        claims=[],
        contradictions=[],
        findings=[],
        report="",
        hypothesis_graph=HypothesisGraph(
            status="unseeded",
            graph_task_id=None,
            graph_namespace=session_id,
            seeded_from_corpus=False,
            non_authoritative=True,
            last_updated_at=None,
        ),
    )


def index_evidence_by_id(evidence_ledger: list[EvidenceRecord]) -> dict[str, EvidenceRecord]:
    """Index evidence items by id for claim verification helpers."""

    return {record["evidence_id"]: record for record in evidence_ledger}


def claim_has_document_support(
    claim: ClaimRecord,
    evidence_index: dict[str, EvidenceRecord],
) -> bool:
    """Return whether the claim is backed by at least one document passage.

    Args:
        claim: Claim under evaluation.
        evidence_index: Evidence lookup keyed by evidence id.

    Returns:
        True if any supporting evidence includes both a document and a passage.
    """

    for evidence_id in claim["supporting_evidence_ids"]:
        record = evidence_index.get(evidence_id)
        if record is None:
            continue
        provenance = record["provenance"]
        if provenance["document"] and provenance["passage"]:
            return True
    return False


def claim_has_graph_support(
    claim: ClaimRecord,
    evidence_index: dict[str, EvidenceRecord],
) -> bool:
    """Return whether the claim is backed by at least one graph path or relation."""

    for evidence_id in claim["supporting_evidence_ids"]:
        record = evidence_index.get(evidence_id)
        if record is None:
            continue
        provenance = record["provenance"]
        if provenance["graph_path"] or provenance["relation"]:
            return True
    return False


def surface_contradictions_from_claims(
    claims: list[ClaimRecord],
    evidence_ledger: list[EvidenceRecord],
) -> list[ContradictionRecord]:
    """Promote claims with both supporting and contradicting evidence into ContradictionRecords.

    A contradiction is surfaced whenever a claim has at least one supporting evidence id
    AND at least one contradicting evidence id.  Contradictions are given an initial
    resolution_status of ``"open"``.  Only claims that appear in the provided
    evidence_ledger are considered (dangling ids are silently skipped).

    Args:
        claims: All claim records produced during the investigation.
        evidence_ledger: The current evidence ledger (used to validate id references).

    Returns:
        A list of new ContradictionRecord objects, one per contradicted claim.
    """

    valid_ids: set[str] = {record["evidence_id"] for record in evidence_ledger}
    contradictions: list[ContradictionRecord] = []
    for claim in claims:
        for_ids = [eid for eid in claim["supporting_evidence_ids"] if eid in valid_ids]
        against_ids = [eid for eid in claim["contradicting_evidence_ids"] if eid in valid_ids]
        if not for_ids or not against_ids:
            continue
        contradictions.append(
            ContradictionRecord(
                contradiction_id=f"contradiction-{claim['claim_id']}",
                claim_id=claim["claim_id"],
                summary=(
                    f"Claim '{claim['text'][:80]}' has {len(for_ids)} supporting and "
                    f"{len(against_ids)} contradicting evidence item(s)."
                ),
                evidence_for_ids=for_ids,
                evidence_against_ids=against_ids,
                resolution_status="open",
            )
        )
    return contradictions


def has_unresolved_contradictions(contradictions: list[ContradictionRecord]) -> bool:
    """Return True if any contradiction is still open or needs more evidence.

    This is the routing predicate used by the ``investigate_surface_contradictions``
    node to decide whether to loop back for additional evidence collection or to
    proceed to the communicate phase.

    Args:
        contradictions: The contradiction records currently in state.

    Returns:
        True when at least one contradiction has resolution_status ``"open"`` or
        ``"needs_more_evidence"``.
    """

    return any(
        c["resolution_status"] in {"open", "needs_more_evidence"}
        for c in contradictions
    )


def format_contradiction_section(
    contradictions: list[ContradictionRecord],
    evidence_index: dict[str, EvidenceRecord],
) -> str:
    """Format unresolved contradictions into a plain-text report section.

    Only contradictions with resolution_status ``"open"`` or ``"needs_more_evidence"``
    are included.  Resolved/explained contradictions are omitted.

    Args:
        contradictions: All contradiction records in state.
        evidence_index: Evidence lookup keyed by evidence id.

    Returns:
        A formatted string suitable for appending to a report, or an empty string
        when there are no unresolved contradictions.
    """

    unresolved = [
        c for c in contradictions
        if c["resolution_status"] in {"open", "needs_more_evidence"}
    ]
    if not unresolved:
        return ""

    lines: list[str] = ["## Unresolved Contradictions\n"]
    for i, contradiction in enumerate(unresolved, start=1):
        lines.append(f"### {i}. {contradiction['summary']}")
        lines.append(f"**Status:** {contradiction['resolution_status']}")

        for_summaries = [
            evidence_index[eid]["summary"]
            for eid in contradiction["evidence_for_ids"]
            if eid in evidence_index
        ]
        against_summaries = [
            evidence_index[eid]["summary"]
            for eid in contradiction["evidence_against_ids"]
            if eid in evidence_index
        ]

        if for_summaries:
            lines.append("**Evidence for:**")
            for s in for_summaries:
                lines.append(f"  - {s}")
        if against_summaries:
            lines.append("**Evidence against:**")
            for s in against_summaries:
                lines.append(f"  - {s}")
        lines.append("")

    return "\n".join(lines)


def claim_is_publishable(
    claim: ClaimRecord,
    evidence_index: dict[str, EvidenceRecord],
) -> bool:
    """Return whether a claim meets the publication grounding rule.

    Important claims must never rely on graph structure alone. If the claim requires
    document support, at least one supporting evidence item must carry both a document
    reference and an exact supporting passage.

    Args:
        claim: Claim under evaluation.
        evidence_index: Evidence lookup keyed by evidence id.

    Returns:
        True when the claim is sufficiently grounded for report generation.
    """

    if claim["status"] != "supported":
        return False
    if not claim["requires_document_support"]:
        return bool(claim["supporting_evidence_ids"])
    return claim_has_document_support(claim, evidence_index)


def select_publishable_claims(
    claims: list[ClaimRecord],
    evidence_index: dict[str, EvidenceRecord],
) -> list[ClaimRecord]:
    """Return only claims that pass the publication grounding gate.

    Args:
        claims: All extracted claims from the investigation.
        evidence_index: Evidence lookup keyed by evidence id.

    Returns:
        Subset of claims cleared for inclusion in the final report write-back.
    """

    return [c for c in claims if claim_is_publishable(c, evidence_index)]


def promote_writeback_outputs(
    state: "InvestigativeState",
) -> dict[str, object]:
    """Extract validated outputs ready for durable write-back.

    Only claims that pass the publishability gate are included.  The hypothesis
    graph is always returned as bounded metadata — callers must not persist it
    as authoritative knowledge; it is non-authoritative by design.

    Args:
        state: Completed investigation state.

    Returns:
        Dict with validated_claims, report, and hypothesis_graph_metadata.
    """

    evidence_index = index_evidence_by_id(state["evidence_ledger"])
    validated_claims = select_publishable_claims(state["claims"], evidence_index)
    hg = state["hypothesis_graph"]
    graph_meta: dict[str, object] = {
        "namespace": hg["graph_namespace"],
        "task_id": hg["graph_task_id"],
        "status": hg["status"],
        "seeded_from_corpus": hg["seeded_from_corpus"],
        "non_authoritative": hg["non_authoritative"],
        "last_updated_at": hg["last_updated_at"],
    }
    return {
        "validated_claims": validated_claims,
        "report": state["report"],
        "hypothesis_graph_metadata": graph_meta,
    }

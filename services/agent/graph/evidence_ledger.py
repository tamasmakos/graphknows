"""Evidence ledger normalization: convert search and graphrag outputs into EvidenceRecord entries.

This module is the single integration point between retrieval tools (SearchTool, RetrieveTool)
and the investigative workflow's evidence ledger. Call ``normalize_search_doc`` or
``normalize_graphrag_chunk`` to build fully-typed provenance envelopes for each item, then
append the resulting ``EvidenceRecord`` to ``InvestigativeState["evidence_ledger"]``.

Scoring defaults (1–5 scale):
    source_authority — how authoritative the publishing institution is considered:
        5  EP/Council/Commission official documents, EUR-Lex primary legislation
        4  ECB, ECA, EPRS briefings, EP resolutions
        3  council conclusions, press releases, tavily web
        2  Unknown / unclassified source
    independence — default 3 (caller may override for corroborating sources)
    directness — 5 if a passage/chunk is present, 3 otherwise (summary-level evidence)
    recency — derived from document date (5 ≤1 year old … 1 ≥5 years old)
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from services.agent.graph.state import (
    DocumentRef,
    EvidenceRecord,
    GraphPathRef,
    PassageRef,
    ProvenanceEnvelope,
    RelationRef,
    RetrievalEventRef,
)


# ---------------------------------------------------------------------------
# Source authority table
# ---------------------------------------------------------------------------

_AUTHORITY_BY_SOURCE: dict[str, int] = {
    # EP official
    "ep_debates": 5,
    "ep_resolutions": 5,
    # Primary legislation
    "eurlex": 5,
    "cellar_sparql": 5,
    # Institutional research / secondary official
    "eprs_briefings": 4,
    "ecb_publications": 4,
    "eca_reports": 4,
    # Policy-adjacent
    "council_conclusions": 3,
    "ec_press": 3,
    # Web / news
    "tavily": 3,
    # graphrag internal corpus
    "graphrag": 4,
}

_DEFAULT_AUTHORITY = 2


def _source_authority(source: str) -> int:
    """Return authority score (1–5) for a named source."""
    return _AUTHORITY_BY_SOURCE.get(source.lower().strip(), _DEFAULT_AUTHORITY)


def _recency_score(date_str: str | None) -> int:
    """Convert an ISO date string to a recency score (1–5).

    5 → document is ≤1 year old
    4 → 1–2 years
    3 → 2–3 years
    2 → 3–5 years
    1 → ≥5 years or unknown date
    """
    if not date_str:
        return 1
    try:
        # Accept YYYY, YYYY-MM, YYYY-MM-DD, and full ISO-8601
        raw = date_str.strip()
        if len(raw) == 4:
            raw = raw + "-01-01"
        elif len(raw) == 7:
            raw = raw + "-01"
        # Strip time zone if full datetime
        pub = datetime.fromisoformat(raw[:10])
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        age_years = (now - pub).days / 365.25
        if age_years <= 1:
            return 5
        if age_years <= 2:
            return 4
        if age_years <= 3:
            return 3
        if age_years <= 5:
            return 2
        return 1
    except (ValueError, TypeError):
        return 1


def _new_evidence_id() -> str:
    return f"ev-{uuid.uuid4().hex[:12]}"


def _new_retrieval_id() -> str:
    return f"ret-{uuid.uuid4().hex[:12]}"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Search doc normalizer
# ---------------------------------------------------------------------------


def normalize_search_doc(
    doc: dict[str, Any],
    *,
    query: str,
    thread_id: str,
    sub_question_id: str,
    rank: int = 0,
    retrieval_id: str | None = None,
    evidence_id: str | None = None,
    retrieved_at: str | None = None,
    independence: int = 3,
) -> EvidenceRecord:
    """Convert a single search-tool document dict into an EvidenceRecord.

    Search docs have the shape::

        {
            "id": str,
            "title": str,
            "text": str,
            "url": str,
            "source": str,          # e.g. "ep_debates", "eurlex", "tavily"
            "date": str | None,
        }

    Args:
        doc: Raw document dict from SearchTool.run().
        query: The query string that produced this result.
        thread_id: Evidence thread this record belongs to.
        sub_question_id: Sub-question being investigated.
        rank: Result rank (0-indexed) within the batch.
        retrieval_id: Override retrieval id (generated if None).
        evidence_id: Override evidence id (generated if None).
        retrieved_at: Override ISO timestamp (current UTC if None).
        independence: Source independence score (1–5). Caller should lower this
            when the same institution appears in multiple results.

    Returns:
        A fully-populated EvidenceRecord with default quality scores.
    """
    source = str(doc.get("source") or "").strip()
    authority = _source_authority(source)
    date_str = str(doc.get("date") or "").strip() or None
    recency = _recency_score(date_str)

    text = str(doc.get("text") or "").strip()
    has_passage = bool(text)
    directness = 5 if has_passage else 3

    _retrieval_id = retrieval_id or _new_retrieval_id()
    _evidence_id = evidence_id or _new_evidence_id()
    _retrieved_at = retrieved_at or _utc_now_iso()

    retrieval_ref: RetrievalEventRef = {
        "retrieval_id": _retrieval_id,
        "query": query,
        "tool_name": "search",
        "retrieved_at": _retrieved_at,
        "rank": rank,
        "score": None,
        "source_type": source,
    }

    document_ref: DocumentRef = {
        "document_id": str(doc.get("id") or ""),
        "title": str(doc.get("title") or ""),
        "url": str(doc.get("url") or ""),
        "publisher": source,
        "published_at": date_str or "",
    }

    passage_ref: PassageRef | None = None
    if has_passage:
        passage_ref = {
            "chunk_id": str(doc.get("id") or _evidence_id),
            "text": text[:2000],
            "start_char": 0,
            "end_char": min(len(text), 2000),
        }

    provenance: ProvenanceEnvelope = {
        "retrieval": retrieval_ref,
        "document": document_ref,
        "passage": passage_ref,
        "relation": None,
        "graph_path": None,
    }

    title = str(doc.get("title") or "")
    summary = title if title else (text[:120] + "…" if len(text) > 120 else text)

    return EvidenceRecord(
        evidence_id=_evidence_id,
        thread_id=thread_id,
        sub_question_id=sub_question_id,
        summary=summary,
        source_authority=authority,
        independence=independence,
        directness=directness,
        recency=recency,
        claim_ids=[],
        tags=[source] if source else [],
        provenance=provenance,
    )


# ---------------------------------------------------------------------------
# GraphRAG chunk normalizer
# ---------------------------------------------------------------------------


def normalize_graphrag_chunk(
    chunk: dict[str, Any],
    *,
    query: str,
    thread_id: str,
    sub_question_id: str,
    rank: int = 0,
    retrieval_id: str | None = None,
    evidence_id: str | None = None,
    retrieved_at: str | None = None,
    independence: int = 3,
    relation: dict[str, Any] | None = None,
    graph_path: dict[str, Any] | None = None,
) -> EvidenceRecord:
    """Convert a single graphrag retrieve chunk dict into an EvidenceRecord.

    GraphRAG chunks have the shape returned by the ``/retrieve`` or ``/chunks``
    graphrag service endpoint::

        {
            "chunk_id": str,
            "text": str,
            "score": float | None,
            "entity_id": str | None,
            "metadata": {
                "document_id": str,
                "title": str,
                "url": str,
                "source": str,
                "published": str | None,
            }
        }

    The ``relation`` and ``graph_path`` overrides let callers pass pre-built
    ``RelationRef`` / ``GraphPathRef``-compatible dicts when the caller has
    graph context (e.g. from ``get_path`` or ``get_community`` tools).

    Args:
        chunk: Raw chunk dict from RetrieveTool.run() or GetChunksTool.run().
        query: The query string that produced this chunk.
        thread_id: Evidence thread this record belongs to.
        sub_question_id: Sub-question being investigated.
        rank: Result rank (0-indexed) within the batch.
        retrieval_id: Override retrieval id (generated if None).
        evidence_id: Override evidence id (generated if None).
        retrieved_at: Override ISO timestamp (current UTC if None).
        independence: Source independence score (1–5).
        relation: Optional RelationRef-compatible dict to attach.
        graph_path: Optional GraphPathRef-compatible dict to attach.

    Returns:
        A fully-populated EvidenceRecord with default quality scores.
    """
    meta: dict[str, Any] = chunk.get("metadata") or {}
    source = str(meta.get("source") or "graphrag").strip()
    authority = _source_authority(source)

    date_str = str(meta.get("published") or meta.get("date") or "").strip() or None
    recency = _recency_score(date_str)

    text = str(chunk.get("text") or "").strip()
    has_passage = bool(text)
    directness = 5 if has_passage else 3

    _retrieval_id = retrieval_id or _new_retrieval_id()
    _evidence_id = evidence_id or _new_evidence_id()
    _retrieved_at = retrieved_at or _utc_now_iso()
    score = chunk.get("score")

    retrieval_ref: RetrievalEventRef = {
        "retrieval_id": _retrieval_id,
        "query": query,
        "tool_name": "retrieve",
        "retrieved_at": _retrieved_at,
        "rank": rank,
        "score": float(score) if score is not None else None,
        "source_type": source,
    }

    document_ref: DocumentRef = {
        "document_id": str(meta.get("document_id") or chunk.get("chunk_id") or ""),
        "title": str(meta.get("title") or ""),
        "url": str(meta.get("url") or ""),
        "publisher": source,
        "published_at": date_str or "",
    }

    chunk_id = str(chunk.get("chunk_id") or _evidence_id)
    passage_ref: PassageRef | None = None
    if has_passage:
        passage_ref = {
            "chunk_id": chunk_id,
            "text": text[:2000],
            "start_char": 0,
            "end_char": min(len(text), 2000),
        }

    relation_ref: RelationRef | None = None
    if relation:
        relation_ref = RelationRef(
            relation_id=str(relation.get("relation_id") or f"rel-{_evidence_id}"),
            source_entity=str(relation.get("source_entity") or ""),
            target_entity=str(relation.get("target_entity") or ""),
            relation_type=str(relation.get("relation_type") or "RELATED_TO"),
            extraction_method=str(relation.get("extraction_method") or "graphrag"),
            confidence=float(relation["confidence"]) if relation.get("confidence") is not None else None,
        )

    graph_path_ref: GraphPathRef | None = None
    if graph_path:
        graph_path_ref = GraphPathRef(
            path_id=str(graph_path.get("path_id") or f"path-{_evidence_id}"),
            entity_names=list(graph_path.get("entity_names") or []),
            edge_types=list(graph_path.get("edge_types") or []),
            supporting_chunk_ids=list(graph_path.get("supporting_chunk_ids") or [chunk_id]),
            hypothesis_only=bool(graph_path.get("hypothesis_only", False)),
        )

    provenance: ProvenanceEnvelope = {
        "retrieval": retrieval_ref,
        "document": document_ref,
        "passage": passage_ref,
        "relation": relation_ref,
        "graph_path": graph_path_ref,
    }

    title = str(meta.get("title") or "")
    summary = title if title else (text[:120] + "…" if len(text) > 120 else text)

    tags = [source]
    if graph_path:
        tags.append("graph-path")
    if relation:
        tags.append("relation")

    return EvidenceRecord(
        evidence_id=_evidence_id,
        thread_id=thread_id,
        sub_question_id=sub_question_id,
        summary=summary,
        source_authority=authority,
        independence=independence,
        directness=directness,
        recency=recency,
        claim_ids=[],
        tags=tags,
        provenance=provenance,
    )


# ---------------------------------------------------------------------------
# Batch helpers
# ---------------------------------------------------------------------------


def normalize_search_results(
    documents: list[dict[str, Any]],
    *,
    query: str,
    thread_id: str,
    sub_question_id: str,
    retrieved_at: str | None = None,
) -> list[EvidenceRecord]:
    """Normalize a full ranked document list from SearchTool into EvidenceRecords.

    Args:
        documents: List of document dicts from SearchTool.run()["documents"].
        query: The search query string.
        thread_id: Evidence thread for all resulting records.
        sub_question_id: Sub-question being investigated.
        retrieved_at: Override ISO timestamp (current UTC if None).

    Returns:
        One EvidenceRecord per document, preserving rank order.
    """
    _retrieved_at = retrieved_at or _utc_now_iso()
    retrieval_id = _new_retrieval_id()
    return [
        normalize_search_doc(
            doc,
            query=query,
            thread_id=thread_id,
            sub_question_id=sub_question_id,
            rank=i,
            retrieval_id=retrieval_id,
            retrieved_at=_retrieved_at,
        )
        for i, doc in enumerate(documents)
    ]


def normalize_graphrag_results(
    chunks: list[dict[str, Any]],
    *,
    query: str,
    thread_id: str,
    sub_question_id: str,
    retrieved_at: str | None = None,
) -> list[EvidenceRecord]:
    """Normalize a full chunk list from RetrieveTool into EvidenceRecords.

    Args:
        chunks: List of chunk dicts from RetrieveTool.run() or GetChunksTool.run().
        query: The query string.
        thread_id: Evidence thread for all resulting records.
        sub_question_id: Sub-question being investigated.
        retrieved_at: Override ISO timestamp (current UTC if None).

    Returns:
        One EvidenceRecord per chunk, preserving rank order.
    """
    _retrieved_at = retrieved_at or _utc_now_iso()
    retrieval_id = _new_retrieval_id()
    return [
        normalize_graphrag_chunk(
            chunk,
            query=query,
            thread_id=thread_id,
            sub_question_id=sub_question_id,
            rank=i,
            retrieval_id=retrieval_id,
            retrieved_at=_retrieved_at,
        )
        for i, chunk in enumerate(chunks)
    ]

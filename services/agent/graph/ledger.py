"""Normalize retrieval outputs into EvidenceRecord ledger entries."""
from __future__ import annotations

from datetime import datetime, timezone
import uuid
from typing import Any

from services.agent.graph.state import (
    EvidenceRecord,
    GraphPathRef,
    PassageRef,
    ProvenanceEnvelope,
)


_AUTHORITY_OVERRIDES: dict[str, int] = {
    "ep_debates": 5,
    "eurlex": 5,
    "tavily": 2,
}


def _utc_now_iso() -> str:
    """Return current UTC timestamp in ISO-8601 format."""

    return datetime.now(timezone.utc).isoformat()


def _new_evidence_id() -> str:
    """Return a new unique evidence identifier."""

    return uuid.uuid4().hex


def _source_authority(source: str) -> int:
    """Return source authority score using neutral defaults and known overrides."""

    normalized = source.strip().lower()
    return _AUTHORITY_OVERRIDES.get(normalized, 3)


def _safe_int(value: Any, default: int) -> int:
    """Return integer value when coercion succeeds, otherwise default."""

    try:
        return int(value)
    except Exception:
        return default


def _safe_float(value: Any) -> float | None:
    """Return float value when coercion succeeds, otherwise None."""

    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _summary_from_text_or_title(text: str, title: str) -> str:
    """Build summary from text first, falling back to title, capped at 200 chars."""

    if text:
        return text[:200]
    return title[:200]


def _path_to_graph_ref(path: dict[str, Any]) -> GraphPathRef:
    """Convert a raw path object into GraphPathRef schema."""

    return {
        "path_id": str(path.get("path_id") or path.get("id") or uuid.uuid4().hex),
        "entity_names": [str(name) for name in (path.get("entity_names") or [])],
        "edge_types": [str(edge) for edge in (path.get("edge_types") or [])],
        "supporting_chunk_ids": [str(cid) for cid in (path.get("supporting_chunk_ids") or [])],
        "hypothesis_only": bool(path.get("hypothesis_only", False)),
    }


def _find_matching_path(
    chunk_id: str,
    paths: list[dict[str, Any]],
) -> GraphPathRef | None:
    """Find the first graph path supporting the given chunk id."""

    for path in paths:
        supporting_chunk_ids = path.get("supporting_chunk_ids") or []
        if any(str(value) == chunk_id for value in supporting_chunk_ids):
            return _path_to_graph_ref(path)
    if paths:
        return _path_to_graph_ref(paths[0])
    return None


def normalize_search_results(
    raw: dict,
    *,
    retrieval_id: str,
    query: str,
    thread_id: str,
    sub_question_id: str,
    tool_name: str = "search_eu",
) -> list[EvidenceRecord]:
    """Convert SearchTool output dict into EvidenceRecord entries.

    Args:
        raw: Search tool output payload.
        retrieval_id: Stable retrieval event identifier for this search batch.
        query: Query string sent to the search tool.
        thread_id: Evidence thread identifier in the investigative workflow.
        sub_question_id: Sub-question currently being investigated.
        tool_name: Name of retrieval tool that produced these results.

    Returns:
        List of normalized evidence records preserving input order.
    """

    documents = raw.get("documents")
    if not isinstance(documents, list):
        return []

    retrieved_at = _utc_now_iso()
    records: list[EvidenceRecord] = []
    for i, item in enumerate(documents):
        if not isinstance(item, dict):
            continue

        text = str(item.get("text") or "")
        title = str(item.get("title") or "")
        source = str(item.get("source") or "")
        rank = _safe_int(item.get("rank"), i)

        provenance: ProvenanceEnvelope = {
            "retrieval": {
                "retrieval_id": retrieval_id,
                "query": query,
                "tool_name": tool_name,
                "retrieved_at": retrieved_at,
                "rank": rank,
                "score": _safe_float(item.get("score")),
                "source_type": source,
            },
            "document": {
                "document_id": str(item.get("id") or ""),
                "title": title,
                "url": str(item.get("url") or ""),
                "publisher": source,
                "published_at": str(item.get("date") or ""),
            },
            "passage": None,
            "relation": None,
            "graph_path": None,
        }

        records.append(
            {
                "evidence_id": _new_evidence_id(),
                "thread_id": thread_id,
                "sub_question_id": sub_question_id,
                "summary": _summary_from_text_or_title(text, title),
                "source_authority": _source_authority(source),
                "independence": 3,
                "directness": 3,
                "recency": 3,
                "claim_ids": [],
                "tags": [],
                "provenance": provenance,
            }
        )

    return records


def normalize_graphrag_results(
    raw: dict,
    *,
    retrieval_id: str,
    query: str,
    thread_id: str,
    sub_question_id: str,
) -> list[EvidenceRecord]:
    """Convert RetrieveTool output dict into EvidenceRecord entries.

    Args:
        raw: GraphRAG retrieval payload, usually containing chunks and optional paths.
        retrieval_id: Stable retrieval event identifier for this retrieval batch.
        query: Query string sent to the retrieve endpoint.
        thread_id: Evidence thread identifier in the investigative workflow.
        sub_question_id: Sub-question currently being investigated.

    Returns:
        List of normalized evidence records preserving chunk order.
    """

    chunks = raw.get("chunks")
    if not isinstance(chunks, list):
        return []

    raw_paths = raw.get("paths")
    paths: list[dict[str, Any]] = []
    if isinstance(raw_paths, list):
        paths = [item for item in raw_paths if isinstance(item, dict)]

    retrieved_at = _utc_now_iso()
    records: list[EvidenceRecord] = []
    for i, chunk in enumerate(chunks):
        if not isinstance(chunk, dict):
            continue

        metadata = chunk.get("metadata")
        meta = metadata if isinstance(metadata, dict) else {}
        text = str(chunk.get("text") or "")
        title = str(meta.get("title") or "")
        source = str(meta.get("source") or "")
        chunk_id = str(chunk.get("chunk_id") or "")
        rank = _safe_int(chunk.get("rank"), i)

        passage: PassageRef | None = None
        if text:
            passage = {
                "chunk_id": chunk_id,
                "text": text,
                "start_char": 0,
                "end_char": len(text),
            }

        provenance: ProvenanceEnvelope = {
            "retrieval": {
                "retrieval_id": retrieval_id,
                "query": query,
                "tool_name": "retrieve",
                "retrieved_at": retrieved_at,
                "rank": rank,
                "score": _safe_float(chunk.get("score")),
                "source_type": source,
            },
            "document": {
                "document_id": str(meta.get("document_id") or ""),
                "title": title,
                "url": str(meta.get("url") or ""),
                "publisher": source,
                "published_at": str(meta.get("published_at") or ""),
            },
            "passage": passage,
            "relation": None,
            "graph_path": _find_matching_path(chunk_id, paths),
        }

        records.append(
            {
                "evidence_id": _new_evidence_id(),
                "thread_id": thread_id,
                "sub_question_id": sub_question_id,
                "summary": _summary_from_text_or_title(text, title),
                "source_authority": _source_authority(source),
                "independence": 3,
                "directness": 3,
                "recency": 3,
                "claim_ids": [],
                "tags": [],
                "provenance": provenance,
            }
        )

    return records


def extend_ledger(
    state: dict,
    new_records: list[EvidenceRecord],
) -> dict:
    """Return state dict with new_records appended to evidence_ledger.

    Args:
        state: Existing state dictionary.
        new_records: New records to append to ``evidence_ledger``.

    Returns:
        A new dictionary with extended ledger while preserving input state immutability.
    """

    existing = list(state.get("evidence_ledger", []))
    existing.extend(new_records)
    return {**state, "evidence_ledger": existing}

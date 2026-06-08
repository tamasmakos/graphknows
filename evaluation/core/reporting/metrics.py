"""Metrics and validation helpers for graphknows MCP evaluation."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from statistics import median
from typing import Any


@dataclass(frozen=True)
class ExpectedEvidence:
    """Expected evidence used to score one recall-like tool response."""

    terms: tuple[str, ...] = ()
    entities: tuple[str, ...] = ()


@dataclass
class ToolObservation:
    """One measured MCP tool call."""

    case_id: str
    tool_name: str
    success: bool
    latency_ms: float
    result: dict[str, Any] | list[Any] | None = None
    error: str = ""
    expected: ExpectedEvidence | None = None
    cold_start: bool = False

    @property
    def returned_error(self) -> bool:
        """Whether the tool returned an error-looking payload despite transport success."""
        if not isinstance(self.result, dict):
            return False
        errors = self.result.get("errors")
        if isinstance(errors, list) and errors:
            return True
        return bool(self.result.get("error"))


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = round((len(ordered) - 1) * percentile)
    return ordered[idx]


def _extract_texts(payload: Any) -> list[str]:
    if not isinstance(payload, dict):
        return []
    rows = payload.get("hits") or payload.get("results") or payload.get("topics") or []
    if not isinstance(rows, list):
        return []
    texts: list[str] = []
    for row in rows:
        if isinstance(row, dict):
            text = row.get("text") or row.get("summary") or row.get("topic_id")
            if text:
                texts.append(str(text))
    return texts


def _extract_entities(payload: Any) -> list[str]:
    if not isinstance(payload, dict):
        return []
    rows = payload.get("hits") or payload.get("results") or payload.get("topics") or []
    if not isinstance(rows, list):
        return []
    entities: list[str] = []
    for row in rows:
        if isinstance(row, dict):
            values = row.get("entities") or row.get("entity_names") or []
            if isinstance(values, list):
                entities.extend(str(v) for v in values)
    return entities


def validate_result_shape(tool_name: str, payload: Any) -> tuple[bool, str]:
    if not isinstance(payload, dict):
        return False, "payload is not an object"
    required_by_tool = {
        "memory_doctor": ("memgraph", "neo4j", "pgvector"),
        "memory_stats": ("stm", "ltm"),
        "memory_ingest": ("file_id", "chunks", "entities", "elapsed_s"),
        "memory_recall": ("hits",),
        "memory_flush": ("elapsed_s", "errors"),
        "memory_purge": ("scope", "session_id", "errors"),
        "ltm_query": ("results",),
        "ltm_entity": (),
        "ltm_topics": ("topics",),
        "recall_by_document": ("results",),
        "recall_by_dialog": ("results",),
        "recall_by_topic": ("results",),
        "corpus_ingest": ("documents_processed",),
    }
    missing = [key for key in required_by_tool.get(tool_name, ()) if key not in payload]
    if missing:
        return False, f"missing keys: {', '.join(missing)}"
    return True, ""


def score_observation(observation: ToolObservation, top_k: int = 5) -> dict[str, float | bool]:
    expected = observation.expected
    if expected is None or not observation.success:
        return {
            "expected_text_found": False,
            "expected_entity_found": False,
            "recall_at_k": 0.0,
            "precision_at_k": 0.0,
            "answer_support_hit": False,
        }

    texts = [text.lower() for text in _extract_texts(observation.result)]
    entities = [entity.lower() for entity in _extract_entities(observation.result)]
    terms = [term.lower() for term in expected.terms]
    expected_entities = [entity.lower() for entity in expected.entities]

    matched_text_terms = [term for term in terms if any(term in text for text in texts[:top_k])]
    matched_entities = [
        entity
        for entity in expected_entities
        if any(entity in candidate for candidate in entities[:top_k])
    ]
    relevant_hits = 0
    for text in texts[:top_k]:
        if any(term in text for term in terms):
            relevant_hits += 1

    total_expected = len(terms) + len(expected_entities)
    total_matched = len(matched_text_terms) + len(matched_entities)
    recall_at_k = total_matched / total_expected if total_expected else 0.0
    precision_at_k = relevant_hits / min(top_k, len(texts)) if texts else 0.0

    return {
        "expected_text_found": bool(matched_text_terms),
        "expected_entity_found": bool(matched_entities),
        "recall_at_k": recall_at_k,
        "precision_at_k": precision_at_k,
        "answer_support_hit": bool(matched_text_terms or matched_entities),
    }


def aggregate_observations(observations: list[ToolObservation], top_k: int = 5) -> dict[str, Any]:
    by_tool: dict[str, list[ToolObservation]] = defaultdict(list)
    for observation in observations:
        by_tool[observation.tool_name].append(observation)

    tool_rows = []
    for tool_name, rows in sorted(by_tool.items()):
        latencies = [row.latency_ms for row in rows]
        quality = [score_observation(row, top_k=top_k) for row in rows if row.expected]
        tool_rows.append(
            {
                "tool_name": tool_name,
                "calls": len(rows),
                "successes": sum(1 for row in rows if row.success),
                "failures": sum(1 for row in rows if not row.success),
                "returned_errors": sum(1 for row in rows if row.returned_error),
                "success_rate": sum(1 for row in rows if row.success) / len(rows),
                "p50_latency_ms": median(latencies),
                "p95_latency_ms": _percentile(latencies, 0.95),
                "cold_start_latency_ms": next(
                    (row.latency_ms for row in rows if row.cold_start), None
                ),
                "avg_recall_at_k": (
                    sum(float(q["recall_at_k"]) for q in quality) / len(quality) if quality else None
                ),
                "avg_precision_at_k": (
                    sum(float(q["precision_at_k"]) for q in quality) / len(quality)
                    if quality
                    else None
                ),
                "answer_support_rate": (
                    sum(1 for q in quality if q["answer_support_hit"]) / len(quality)
                    if quality
                    else None
                ),
            }
        )

    elapsed_ms = sum(row.latency_ms for row in observations)
    return {
        "total_calls": len(observations),
        "success_rate": (
            sum(1 for row in observations if row.success) / len(observations)
            if observations
            else 0.0
        ),
        "error_rate": (
            sum(1 for row in observations if not row.success or row.returned_error)
            / len(observations)
            if observations
            else 0.0
        ),
        "throughput_calls_per_s": (
            len(observations) / (elapsed_ms / 1000.0) if elapsed_ms > 0 else 0.0
        ),
        "tools": tool_rows,
    }


def observation_to_record(observation: ToolObservation, top_k: int = 5) -> dict[str, Any]:
    shape_ok, shape_error = validate_result_shape(observation.tool_name, observation.result)
    return {
        "case_id": observation.case_id,
        "tool_name": observation.tool_name,
        "success": observation.success,
        "latency_ms": observation.latency_ms,
        "returned_error": observation.returned_error,
        "error": observation.error,
        "shape_ok": shape_ok,
        "shape_error": shape_error,
        "quality": score_observation(observation, top_k=top_k),
        "result": observation.result,
    }

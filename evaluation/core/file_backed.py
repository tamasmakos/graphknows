"""File-backed memory benchmark runner for local benchmark exports."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from evaluation.core.client import GraphknowsMCPClient
from evaluation.core.models import BenchmarkCaseResult


@dataclass(frozen=True)
class FileBackedBenchmarkOptions:
    """Options for benchmarks represented as local JSON or JSONL cases."""

    benchmark: str
    data_root: Path
    questions_file: str
    run_id: str
    top_k: int = 5
    offset: int = 0
    limit: int = 1
    max_documents: int = 25
    max_chars_per_document: int = 4000
    recall_scope: str = "stm"
    promote_to_ltm: bool = False
    session_prefix: str = "eval"
    purge: bool = True


def _read_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} does not exist. Add a local benchmark export or pass the matching "
            "--*-data-root/--*-questions option."
        )

    if path.suffix == ".jsonl":
        return [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    if isinstance(payload, dict):
        for key in ("questions", "cases", "examples", "data"):
            rows = payload.get(key)
            if isinstance(rows, list):
                return [row for row in rows if isinstance(row, dict)]
    raise ValueError(f"{path} must contain a JSON array, JSONL rows, or a dict with cases/questions")


def _first_string(row: dict[str, Any], keys: tuple[str, ...], default: str = "") -> str:
    for key in keys:
        value = row.get(key)
        if value is not None:
            return str(value)
    return default


def _normalize_tokens(text: str) -> set[str]:
    return {token for token in re.findall(r"[a-z0-9][a-z0-9_-]{1,}", text.lower())}


def _clip(text: str, max_chars: int) -> str:
    clean = text.strip()
    if len(clean) <= max_chars:
        return clean
    return clean[: max(0, max_chars - 3)].rstrip() + "..."


def _message_text(message: Any) -> str:
    if isinstance(message, str):
        return message
    if not isinstance(message, dict):
        return ""
    role = _first_string(message, ("role", "speaker", "author"), "message")
    content = _first_string(message, ("content", "text", "utterance", "message"))
    timestamp = _first_string(message, ("timestamp", "created_at", "time"))
    prefix = f"{timestamp} | {role}" if timestamp else role
    return f"{prefix}: {content}" if content else ""


def _document_text(document: Any) -> tuple[str, str]:
    if isinstance(document, str):
        return "document", document
    if not isinstance(document, dict):
        return "document", ""
    title = _first_string(document, ("title", "id", "name", "source"), "document")
    text = _first_string(document, ("text", "content", "body", "document", "memory"))
    if not text and isinstance(document.get("messages"), list):
        text = "\n".join(filter(None, (_message_text(msg) for msg in document["messages"])))
    return title, text


def _case_documents(row: dict[str, Any], *, max_documents: int, max_chars: int) -> list[dict[str, str]]:
    sources: list[Any] = []
    for key in ("documents", "evidence", "memories", "haystack", "contexts"):
        value = row.get(key)
        if isinstance(value, list):
            sources.extend(value)
    for key in ("conversation", "messages", "turns"):
        value = row.get(key)
        if isinstance(value, list):
            sources.append({"title": key, "messages": value})
    sessions = row.get("sessions")
    if isinstance(sessions, list):
        for index, session in enumerate(sessions):
            if isinstance(session, dict):
                sources.append(
                    {
                        "title": _first_string(session, ("title", "id", "session_id"), f"session-{index}"),
                        "messages": session.get("messages") or session.get("turns") or [],
                        "text": _first_string(session, ("text", "content", "summary")),
                    }
                )
            else:
                sources.append(session)
    for key in ("context", "memory", "transcript"):
        value = row.get(key)
        if isinstance(value, str):
            sources.append({"title": key, "text": value})

    documents: list[dict[str, str]] = []
    for index, source in enumerate(sources[:max_documents]):
        title, text = _document_text(source)
        text = _clip(text, max_chars)
        if text:
            documents.append({"title": title or f"document-{index}", "text": text})
    return documents


def _answer_support_score(answer: Any, hits: list[dict[str, Any]]) -> float:
    expected = str(answer or "").strip().lower()
    if not expected:
        return 0.0
    haystack = "\n".join(str(hit.get("text", "")) for hit in hits).lower()
    if expected in haystack:
        return 1.0
    expected_tokens = _normalize_tokens(expected)
    if not expected_tokens:
        return 0.0
    overlap = expected_tokens & _normalize_tokens(haystack)
    return len(overlap) / len(expected_tokens)


async def run_file_backed_benchmark(
    client: GraphknowsMCPClient,
    options: FileBackedBenchmarkOptions,
) -> list[BenchmarkCaseResult]:
    """Run a file-backed memory benchmark against the MCP memory tools."""

    rows = _read_rows(options.data_root / options.questions_file)
    selected = rows[options.offset : options.offset + options.limit]
    cases: list[BenchmarkCaseResult] = []
    for ordinal, row in enumerate(selected, start=options.offset):
        case_id = _first_string(row, ("id", "question_id", "case_id"), f"{options.benchmark}-{ordinal}")
        question = _first_string(row, ("question", "query", "prompt"))
        if not question:
            raise ValueError(f"{options.benchmark} case {case_id} is missing a question/query/prompt")
        gold = row.get("answer", row.get("gold", row.get("target")))
        category = _first_string(row, ("category", "type", "task_type", "question_type"))
        session_id = f"{options.session_prefix}-{options.benchmark}-{options.run_id}-{case_id}"
        documents = _case_documents(
            row,
            max_documents=options.max_documents,
            max_chars=options.max_chars_per_document,
        )
        ingest_results: list[dict[str, Any]] = []
        flush_result: dict[str, Any] = {}
        recall: list[dict[str, Any]] = []
        try:
            for index, document in enumerate(documents):
                result = await client.call_tool(
                    "memory_ingest",
                    {
                        "text": document["text"],
                        "session_id": session_id,
                        "title": document["title"] or f"{case_id}-document-{index}",
                    },
                )
                ingest_results.append(result if isinstance(result, dict) else {"result": result})
            if options.promote_to_ltm:
                result = await client.call_tool("memory_flush", {"session_id": session_id})
                flush_result = result if isinstance(result, dict) else {"result": result}
            result = await client.call_tool(
                "memory_recall",
                {
                    "query": question,
                    "session_id": session_id,
                    "top_k": options.top_k,
                    "scope": options.recall_scope,
                },
            )
            recall = list(result.get("hits", [])) if isinstance(result, dict) else []
        finally:
            if options.purge:
                await client.call_tool("memory_purge", {"scope": "both", "session_id": session_id})

        answer_score = _answer_support_score(gold, recall)
        cases.append(
            BenchmarkCaseResult(
                case_id=case_id,
                question=question,
                gold=gold,
                memory_context=recall,
                metadata={
                    "dataset": options.benchmark,
                    "category": category or None,
                    "session_id": session_id,
                    "source_file": str(options.data_root / options.questions_file),
                    "ingest_documents": len(documents),
                    "ingest_results": ingest_results,
                    "flush_result": flush_result or None,
                },
                metrics={
                    "recall_hit_count": len(recall),
                    "answer_support_score": answer_score,
                    "answer_support_hit": answer_score >= 1.0,
                    "ingest_calls": len(ingest_results),
                    "ingested_chunks": sum(int(result.get("chunks", 0)) for result in ingest_results),
                    "ingested_entities": sum(int(result.get("entities", 0)) for result in ingest_results),
                    "flush_nodes": int(flush_result.get("nodes", 0)) if flush_result else 0,
                    "flush_edges": int(flush_result.get("edges", 0)) if flush_result else 0,
                    "flush_chunks": int(flush_result.get("chunks", 0)) if flush_result else 0,
                    "flush_topics": int(flush_result.get("topics_created", 0)) if flush_result else 0,
                    "flush_error_count": len(flush_result.get("errors", [])) if flush_result else 0,
                },
                score=answer_score,
            )
        )
    return cases

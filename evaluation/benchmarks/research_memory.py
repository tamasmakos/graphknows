"""Research-grade retrieval evaluation with fixed memory operations."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from evaluation.benchmarks.hotpotqa import (
    _context_documents,
    fetch_hotpotqa_rows,
)
from evaluation.benchmarks.longmemeval import (
    CapsuleOptions,
    capsule_documents,
    extract_lme_trajectories,
    load_lme_metadata,
    select_lme_questions,
)
from evaluation.benchmarks.memoryarena import (
    _compact_answer,
    _expected_prior_terms,
    fetch_memoryarena_rows,
)
from evaluation.benchmarks.types import BenchmarkCaseResult
from evaluation.core.client import GraphknowsMCPClient


@dataclass(frozen=True)
class ResearchMemoryOptions:
    """Options for deterministic, retrieval-only memory evaluation."""

    suite: str
    run_id: str
    top_k: int
    taxonomy_file: str | None
    offset: int = 0
    limit: int = 1
    domain: str = "bundled_shopping"
    subtask_index: int = 1
    data_root: Path = Path("evaluation/data/longmemeval-v2")
    question_id: str | None = None
    max_trajectories: int = 5
    max_states: int = 8
    max_chars_per_trajectory: int = 4000
    max_docs: int = 10
    max_chars_per_doc: int = 2500


@dataclass(frozen=True)
class ResearchCase:
    """One retrieval-only case ready for fixed memory execution."""

    case_id: str
    question: str
    gold: Any
    documents: list[dict[str, str]]
    expected_answer_terms: tuple[str, ...]
    expected_evidence_terms: tuple[str, ...]
    expected_entity_terms: tuple[str, ...]
    metadata: dict[str, Any]


def _tokens(value: Any) -> tuple[str, ...]:
    text = json.dumps(value, ensure_ascii=False) if not isinstance(value, str) else value
    return tuple(dict.fromkeys(re.findall(r"[A-Za-z0-9][A-Za-z0-9_-]{2,}", text)))


def _normalise(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip().lower()


def _contains_any(text: str, terms: tuple[str, ...]) -> bool:
    haystack = _normalise(text)
    return any(_normalise(term) in haystack for term in terms if _normalise(term))


def _fraction_found(text: str, terms: tuple[str, ...]) -> float:
    unique_terms = tuple(dict.fromkeys(term for term in terms if _normalise(term)))
    if not unique_terms:
        return 0.0
    return sum(1 for term in unique_terms if _contains_any(text, (term,))) / len(unique_terms)


def _result_text(payload: Any) -> str:
    if not isinstance(payload, dict):
        return ""
    rows = payload.get("hits") or payload.get("results") or payload.get("topics") or []
    if not isinstance(rows, list):
        return ""
    parts: list[str] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        for key in ("text", "summary", "topic_id", "label"):
            value = row.get(key)
            if value:
                parts.append(str(value))
        for key in ("entities", "entity_names"):
            values = row.get(key)
            if isinstance(values, list):
                parts.extend(str(value) for value in values)
    return "\n".join(parts)


def _score_case(
    case: ResearchCase,
    retrievals: dict[str, Any],
) -> dict[str, Any]:
    evidence_text = "\n".join(
        _result_text(retrievals.get(name))
        for name in ("memory_recall_ltm", "recall_by_document")
    )
    topic_entity_text = "\n".join(
        _result_text(retrievals.get(name))
        for name in ("recall_by_topic", "ltm_query", "ltm_topics")
    )
    all_text = "\n".join((evidence_text, topic_entity_text))

    answer_support_score = float(_contains_any(all_text, case.expected_answer_terms))
    evidence_recall_score = _fraction_found(evidence_text, case.expected_evidence_terms)
    topic_entity_score = max(
        _fraction_found(topic_entity_text, case.expected_entity_terms),
        _fraction_found(topic_entity_text, case.expected_evidence_terms),
    )
    overall = (answer_support_score + evidence_recall_score + topic_entity_score) / 3.0

    return {
        "answer_support_score": answer_support_score,
        "evidence_recall_score": evidence_recall_score,
        "topic_entity_score": topic_entity_score,
        "research_overall_score": overall,
    }


async def _build_hotpot_cases(options: ResearchMemoryOptions) -> list[ResearchCase]:
    rows, total_rows = await fetch_hotpotqa_rows(options.offset, options.limit)
    cases: list[ResearchCase] = []
    for row_index, row in enumerate(rows):
        supporting = row.get("supporting_facts", {})
        supporting_titles = tuple(str(title) for title in supporting.get("title", []) if title)
        answer = str(row.get("answer", ""))
        docs = _context_documents(
            row,
            max_docs=options.max_docs,
            max_chars_per_doc=options.max_chars_per_doc,
        )
        cases.append(
            ResearchCase(
                case_id=str(row.get("id", options.offset + row_index)),
                question=str(row.get("question", "")),
                gold=row.get("answer"),
                documents=docs,
                expected_answer_terms=(answer,) if answer else (),
                expected_evidence_terms=supporting_titles,
                expected_entity_terms=supporting_titles,
                metadata={
                    "dataset": "hotpotqa/hotpot_qa",
                    "offset": options.offset + row_index,
                    "official_rows": total_rows,
                    "level": row.get("level"),
                    "type": row.get("type"),
                    "context_docs_indexed": len(docs),
                },
            )
        )
    return cases


async def _build_memoryarena_cases(options: ResearchMemoryOptions) -> list[ResearchCase]:
    rows, total_rows = await fetch_memoryarena_rows(options.domain, options.offset, options.limit)
    cases: list[ResearchCase] = []
    for task_index, task in enumerate(rows):
        questions = list(task.get("questions", []))
        answers = list(task.get("answers", []))
        if options.subtask_index <= 0:
            raise ValueError("subtask_index must be > 0 so prior memory exists")
        if options.subtask_index >= len(questions):
            raise ValueError(
                f"subtask_index {options.subtask_index} outside task length {len(questions)}"
            )
        docs = [
            {
                "id": f"{task.get('id', task_index)}-{idx}",
                "title": f"memoryarena-{options.domain}-{task.get('id', task_index)}-{idx}",
                "source": "memoryarena",
                "text": f"Q: {questions[idx]}\nA: {_compact_answer(answers[idx])}",
            }
            for idx in range(options.subtask_index)
        ]
        prior_terms = _expected_prior_terms(answers[options.subtask_index - 1])
        cases.append(
            ResearchCase(
                case_id=str(task.get("id", options.offset + task_index)),
                question=str(questions[options.subtask_index]),
                gold=answers[options.subtask_index],
                documents=docs,
                expected_answer_terms=prior_terms,
                expected_evidence_terms=prior_terms,
                expected_entity_terms=prior_terms,
                metadata={
                    "dataset": "ZexueHe/memoryarena",
                    "domain": options.domain,
                    "offset": options.offset + task_index,
                    "official_rows": total_rows,
                    "task_id": task.get("id", options.offset + task_index),
                    "subtask_index": options.subtask_index,
                    "prior_turns_indexed": len(docs),
                },
            )
        )
    return cases


async def _build_longmemeval_cases(options: ResearchMemoryOptions) -> list[ResearchCase]:
    questions, haystack = load_lme_metadata(options.data_root)
    selected = select_lme_questions(
        questions,
        haystack,
        question_id=options.question_id,
        max_questions=options.limit,
    )
    cases: list[ResearchCase] = []
    capsule_options = CapsuleOptions(
        max_states=options.max_states,
        max_chars_per_state=max(500, options.max_chars_per_trajectory // max(options.max_states, 1)),
        max_chars_per_capsule=options.max_chars_per_trajectory,
    )
    for question in selected:
        qid = str(question["id"])
        trajectory_ids = list(haystack[qid])
        trajectories = extract_lme_trajectories(
            options.data_root,
            trajectory_ids,
            options.max_trajectories,
        )
        docs = capsule_documents(trajectories, str(question["question"]), capsule_options)
        indexed_ids = tuple(doc["id"] for doc in docs)
        answer = str(question.get("answer", ""))
        cases.append(
            ResearchCase(
                case_id=qid,
                question=str(question["question"]),
                gold=question.get("answer"),
                documents=docs,
                expected_answer_terms=(answer,) if answer else (),
                expected_evidence_terms=indexed_ids,
                expected_entity_terms=tuple(_tokens(question["question"])[:5]),
                metadata={
                    "dataset": "xiaowu0162/longmemeval-v2",
                    "question_id": qid,
                    "domain": question.get("domain"),
                    "question_type": question.get("question_type"),
                    "official_haystack_size": len(trajectory_ids),
                    "indexed_trajectories": len(docs),
                },
            )
        )
    return cases


async def build_research_cases(options: ResearchMemoryOptions) -> list[ResearchCase]:
    """Build benchmark cases without touching memory backends."""
    if options.suite == "hotpotqa":
        return await _build_hotpot_cases(options)
    if options.suite == "memoryarena":
        return await _build_memoryarena_cases(options)
    if options.suite == "longmemeval":
        return await _build_longmemeval_cases(options)
    raise ValueError(f"unsupported research suite: {options.suite}")


async def _call_tool_safe(
    client: GraphknowsMCPClient,
    name: str,
    arguments: dict[str, Any],
) -> dict[str, Any]:
    try:
        result = await client.call_tool(name, arguments)
    except Exception as exc:
        return {"error": str(exc)}
    return result if isinstance(result, dict) else {"result": result}


async def run_research_memory(
    client: GraphknowsMCPClient,
    options: ResearchMemoryOptions,
) -> list[BenchmarkCaseResult]:
    """Run isolated, retrieval-only research cases through fixed memory operations."""
    cases = await build_research_cases(options)
    results: list[BenchmarkCaseResult] = []
    for index, case in enumerate(cases):
        session_id = f"{options.run_id}-{options.suite}-{case.case_id}-{index}"
        pre_purge = await _call_tool_safe(
            client,
            "memory_purge",
            {"scope": "both", "session_id": session_id},
        )
        ingest = await _call_tool_safe(
            client,
            "corpus_ingest",
            {
                "documents": case.documents,
                "taxonomy_file": options.taxonomy_file,
                "session_id": session_id,
            },
        )
        post_flush_stats = await _call_tool_safe(
            client,
            "memory_stats",
            {"session_id": session_id},
        )
        retrievals = {
            "memory_recall_ltm": await _call_tool_safe(
                client,
                "memory_recall",
                {
                    "query": case.question,
                    "session_id": session_id,
                    "top_k": options.top_k,
                    "scope": "ltm",
                },
            ),
            "recall_by_document": await _call_tool_safe(
                client,
                "recall_by_document",
                {"query": case.question, "top_k": options.top_k},
            ),
            "recall_by_topic": await _call_tool_safe(
                client,
                "recall_by_topic",
                {"query": case.question, "top_k": options.top_k},
            ),
            "ltm_query": await _call_tool_safe(
                client,
                "ltm_query",
                {"query": case.question, "top_k": options.top_k},
            ),
            "ltm_topics": await _call_tool_safe(
                client,
                "ltm_topics",
                {"min_entities": 1},
            ),
        }
        metrics = _score_case(case, retrievals)
        metrics.update(
            {
                "documents_processed": int(ingest.get("documents_processed", 0)),
                "entities_extracted": int(ingest.get("entities_extracted", 0)),
                "chunks_written": int(ingest.get("chunks_written", 0)),
                "nodes_written": int(ingest.get("nodes_written", 0)),
                "edges_written": int(ingest.get("edges_written", 0)),
                "ingest_errors": len(ingest.get("errors", []) or []),
                "topics_available": len(
                    retrievals.get("ltm_topics", {}).get("topics", [])
                    if isinstance(retrievals.get("ltm_topics"), dict)
                    else []
                ),
            }
        )
        cleanup = await _call_tool_safe(
            client,
            "memory_purge",
            {"scope": "both", "session_id": session_id},
        )
        results.append(
            BenchmarkCaseResult(
                case_id=case.case_id,
                question=case.question,
                gold=case.gold,
                memory_context=retrievals["memory_recall_ltm"].get("hits", []),
                metadata={
                    **case.metadata,
                    "session_id": session_id,
                    "taxonomy_file": options.taxonomy_file or "taxonomy.generic.yaml",
                    "operation_sequence": [
                        "memory_purge:pre",
                        "corpus_ingest:taxonomy+stm+flush+topics+gds",
                        "memory_stats:post_flush",
                        "memory_recall:ltm",
                        "recall_by_document",
                        "recall_by_topic",
                        "ltm_query",
                        "ltm_topics",
                        "memory_purge:post",
                    ],
                    "pre_purge": pre_purge,
                    "ingest": ingest,
                    "post_flush_stats": post_flush_stats,
                    "retrievals": retrievals,
                    "cleanup": cleanup,
                    "expected_answer_terms": case.expected_answer_terms,
                    "expected_evidence_terms": case.expected_evidence_terms,
                    "expected_entity_terms": case.expected_entity_terms,
                },
                metrics=metrics,
                score=float(metrics["research_overall_score"]),
            )
        )
    return results

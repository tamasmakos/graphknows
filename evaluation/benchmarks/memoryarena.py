"""MemoryArena benchmark helpers for graphknows evaluation."""

from __future__ import annotations

import json
import re
from typing import Any

from evaluation.benchmarks.huggingface import fetch_dataset_rows
from evaluation.benchmarks.types import BenchmarkCaseResult
from evaluation.core.client import GraphknowsMCPClient

MEMORYARENA_DATASET = "ZexueHe/memoryarena"
DOMAIN_CONFIGS = {
    "bundled_shopping": "bundled_shopping",
    "progressive_search": "progressive_search",
    "group_travel_planner": "group_travel_planner",
    "formal_reasoning_math": "formal_reasoning_math",
    "formal_reasoning_phys": "formal_reasoning_phys",
}


def exact_match_score(pred: str, gold: Any) -> float:
    if isinstance(gold, dict) and "target_asin" in gold:
        return float(str(gold["target_asin"]).strip().lower() in pred.lower())
    if isinstance(gold, str):
        return float(pred.strip().lower() == gold.strip().lower())
    return 0.0


def travel_plan_score(pred: str, gold: Any) -> float:
    try:
        pred_json = json.loads(pred)
    except json.JSONDecodeError:
        return 0.0
    if not isinstance(gold, list) or not isinstance(pred_json, list) or not gold:
        return 0.0
    matches = 0
    for expected, actual in zip(gold, pred_json, strict=False):
        if isinstance(expected, dict) and isinstance(actual, dict):
            matches += int(expected.get("current_city") == actual.get("current_city"))
    return matches / len(gold)


def numeric_score(pred: str, gold: Any) -> float:
    pred_nums = re.findall(r"-?\d+\.?\d*", pred)
    gold_nums = re.findall(r"-?\d+\.?\d*", str(gold))
    if not pred_nums or not gold_nums:
        return 0.0
    return float(abs(float(pred_nums[-1]) - float(gold_nums[-1])) < 1e-3)


def score_for_domain(domain: str, pred: str, gold: Any) -> float:
    if domain == "group_travel_planner":
        return travel_plan_score(pred, gold)
    if domain.startswith("formal_reasoning"):
        return numeric_score(pred, gold)
    return exact_match_score(pred, gold)


def _compact_answer(answer: Any) -> str:
    return json.dumps(answer, ensure_ascii=False, sort_keys=True)


def _expected_prior_terms(answer: Any) -> tuple[str, ...]:
    if isinstance(answer, dict) and "target_asin" in answer:
        return (str(answer["target_asin"]),)
    if isinstance(answer, str):
        return (answer,)
    text = _compact_answer(answer)
    tokens = tuple(token for token in re.findall(r"[A-Za-z0-9_-]{3,}", text)[:5])
    return tokens or (text[:40],)


async def recall_context(
    client: GraphknowsMCPClient, session_id: str, query: str, top_k: int
) -> list[dict[str, Any]]:
    result = await client.call_tool(
        "memory_recall",
        {"query": query, "session_id": session_id, "top_k": top_k, "scope": "both"},
    )
    hits = result.get("hits", []) if isinstance(result, dict) else []
    return [hit for hit in hits if isinstance(hit, dict)]


async def store_interaction(
    client: GraphknowsMCPClient, session_id: str, question: str, answer: str, title: str
) -> None:
    await client.call_tool(
        "memory_ingest",
        {"text": f"Q: {question}\nA: {answer}", "session_id": session_id, "title": title},
    )


async def fetch_memoryarena_rows(domain: str, offset: int, limit: int) -> tuple[list[dict[str, Any]], int]:
    return fetch_dataset_rows(
        dataset=MEMORYARENA_DATASET,
        config=DOMAIN_CONFIGS[domain],
        split="test",
        offset=offset,
        length=limit,
    )


async def run_memoryarena_case(
    client: GraphknowsMCPClient,
    *,
    domain: str,
    offset: int,
    subtask_index: int,
    run_id: str,
    top_k: int,
    purge: bool = True,
) -> BenchmarkCaseResult:
    rows, total_rows = await fetch_memoryarena_rows(domain, offset, 1)
    task = rows[0]
    questions = list(task.get("questions", []))
    answers = list(task.get("answers", []))
    if subtask_index <= 0:
        raise ValueError("subtask_index must be > 0 so prior memory exists")
    if subtask_index >= len(questions):
        raise ValueError(f"subtask_index {subtask_index} outside task length {len(questions)}")

    session_id = f"{run_id}-memoryarena-{domain}-{task.get('id', offset)}"
    recall: list[dict[str, Any]] = []
    ingest_results: list[dict[str, Any]] = []
    try:
        for idx in range(subtask_index):
            result = await client.call_tool(
                "memory_ingest",
                {
                    "text": f"Q: {questions[idx]}\nA: {_compact_answer(answers[idx])}",
                    "session_id": session_id,
                    "title": f"memoryarena-{domain}-{task.get('id', offset)}-{idx}",
                },
            )
            ingest_results.append(result if isinstance(result, dict) else {"result": result})
        recall = await recall_context(client, session_id, str(questions[subtask_index]), top_k)
    finally:
        if purge:
            await client.call_tool("memory_purge", {"scope": "both", "session_id": session_id})

    prior_terms = _expected_prior_terms(answers[subtask_index - 1])
    recalled_text = "\n".join(str(hit.get("text", "")) for hit in recall).lower()
    prior_answer_support_hit = any(term.lower() in recalled_text for term in prior_terms if term)
    return BenchmarkCaseResult(
        case_id=str(task.get("id", offset)),
        question=str(questions[subtask_index]),
        gold=answers[subtask_index],
        memory_context=recall,
        metadata={
            "dataset": MEMORYARENA_DATASET,
            "domain": domain,
            "offset": offset,
            "official_rows": total_rows,
            "task_id": task.get("id", offset),
            "subtask_index": subtask_index,
            "session_id": session_id,
            "ingest_results": ingest_results,
        },
        metrics={
            "recall_hit_count": len(recall),
            "prior_answer_support_hit": prior_answer_support_hit,
            "ingest_calls": len(ingest_results),
            "ingested_chunks": sum(int(result.get("chunks", 0)) for result in ingest_results),
            "ingested_entities": sum(int(result.get("entities", 0)) for result in ingest_results),
        },
    )


async def run_memoryarena_small(
    client: GraphknowsMCPClient,
    *,
    domain: str,
    offset: int,
    limit: int,
    subtask_index: int,
    run_id: str,
    top_k: int,
    purge: bool = True,
) -> list[BenchmarkCaseResult]:
    cases: list[BenchmarkCaseResult] = []
    for row_offset in range(offset, offset + limit):
        cases.append(
            await run_memoryarena_case(
                client,
                domain=domain,
                offset=row_offset,
                subtask_index=subtask_index,
                run_id=run_id,
                top_k=top_k,
                purge=purge,
            )
        )
    return cases


async def evaluate_domain_with_predictions(
    client: GraphknowsMCPClient,
    domain: str,
    rows: list[dict[str, Any]],
    predictions: list[list[str]],
    session_prefix: str = "eval-memoryarena",
    top_k: int = 5,
) -> dict[str, Any]:
    task_results: list[dict[str, Any]] = []
    for task_idx, task in enumerate(rows):
        session_id = f"{session_prefix}-{domain}-{task.get('id', task_idx)}"
        questions = list(task.get("questions", []))
        answers = list(task.get("answers", []))
        task_predictions = predictions[task_idx]
        subtask_results = []
        for sub_idx, (question, gold) in enumerate(zip(questions, answers, strict=False)):
            await recall_context(client, session_id, str(question), top_k)
            pred = task_predictions[sub_idx]
            score = score_for_domain(domain, pred, gold)
            await store_interaction(
                client,
                session_id,
                str(question),
                pred,
                f"memoryarena-prediction-{task.get('id', task_idx)}-{sub_idx}",
            )
            subtask_results.append(
                {
                    "subtask_idx": sub_idx,
                    "question": question,
                    "gold": gold,
                    "predicted": pred,
                    "score": score,
                }
            )
        sr = float(all(row["score"] == 1.0 for row in subtask_results))
        ps = sum(float(row["score"]) for row in subtask_results) / len(subtask_results) if subtask_results else 0.0
        task_results.append({"task_id": task.get("id", task_idx), "sr": sr, "ps": ps, "subtasks": subtask_results})
    avg_sr = sum(row["sr"] for row in task_results) / len(task_results) if task_results else 0.0
    avg_ps = sum(row["ps"] for row in task_results) / len(task_results) if task_results else 0.0
    return {"domain": domain, "avg_sr": avg_sr, "avg_ps": avg_ps, "tasks": task_results}

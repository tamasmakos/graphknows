"""LongMemEval benchmark helpers for graphknows evaluation."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from evaluation.benchmarks.huggingface import download_if_missing
from evaluation.benchmarks.types import BenchmarkCaseResult
from evaluation.core.client import GraphknowsMCPClient

LONGMEMEVAL_DATASET = "xiaowu0162/longmemeval-v2"
LONGMEMEVAL_RESOLVE = "https://huggingface.co/datasets/xiaowu0162/longmemeval-v2/resolve/main"
BOUNDARY = "\n\n--- LongMemEval trajectory capsule boundary ---\n\n"


@dataclass(frozen=True)
class CapsuleOptions:
    max_states: int = 8
    max_chars_per_state: int = 700
    max_chars_per_capsule: int = 4000


def _tokens(text: str) -> set[str]:
    return {token for token in re.findall(r"[a-z0-9][a-z0-9_-]{2,}", text.lower()) if token}


def _clip(value: object, limit: int) -> str:
    text = str(value or "").strip()
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)].rstrip() + "..."


def _state_text(state: dict[str, Any], max_chars: int) -> str:
    fields = (
        ("step", state.get("step")),
        ("url", state.get("url")),
        ("thought", state.get("thought")),
        ("action", state.get("action")),
        ("observation", state.get("observation")),
    )
    return _clip(" | ".join(f"{key}={value}" for key, value in fields if value), max_chars)


def load_lme_metadata(data_root: Path) -> tuple[list[dict[str, Any]], dict[str, list[str]]]:
    download_if_missing(f"{LONGMEMEVAL_RESOLVE}/questions.jsonl", data_root / "questions.jsonl")
    download_if_missing(
        f"{LONGMEMEVAL_RESOLVE}/haystacks/lme_v2_small.json",
        data_root / "haystacks" / "lme_v2_small.json",
    )
    questions = [
        json.loads(line)
        for line in (data_root / "questions.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    haystack = json.loads((data_root / "haystacks" / "lme_v2_small.json").read_text(encoding="utf-8"))
    return questions, haystack


def select_lme_questions(
    questions: list[dict[str, Any]],
    haystack: dict[str, list[str]],
    *,
    question_id: str | None,
    max_questions: int,
) -> list[dict[str, Any]]:
    by_id = {str(row["id"]): row for row in questions}
    if question_id:
        return [by_id[question_id]]
    ordered_ids = sorted(haystack, key=lambda item: (len(haystack[item]), item))
    return [by_id[qid] for qid in ordered_ids[:max_questions]]


def extract_lme_trajectories(
    data_root: Path, trajectory_ids: list[str], max_trajectories: int
) -> list[dict[str, Any]]:
    selected = list(trajectory_ids[:max_trajectories])
    selected_set = set(selected)
    out_path = data_root / "one_question_trajectories.jsonl"
    found: dict[str, dict[str, Any]] = {}
    if out_path.exists():
        for line in out_path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if row.get("id") in selected_set:
                found[str(row["id"])] = row
    if len(found) == len(selected_set):
        return [found[item] for item in selected]

    download_if_missing(f"{LONGMEMEVAL_RESOLVE}/trajectories.jsonl", data_root / "trajectories.jsonl")
    with (data_root / "trajectories.jsonl").open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            trajectory_id = str(row.get("id"))
            if trajectory_id in selected_set:
                found[trajectory_id] = row
                if len(found) == len(selected_set):
                    break
    out_path.write_text(
        "\n".join(json.dumps(found[item], ensure_ascii=True) for item in selected if item in found),
        encoding="utf-8",
    )
    missing = selected_set - set(found)
    if missing:
        raise RuntimeError(f"missing LongMemEval trajectories: {sorted(missing)}")
    return [found[item] for item in selected]


def select_states(trajectory: dict[str, Any], question: str, *, max_states: int) -> list[dict[str, Any]]:
    states = [state for state in trajectory.get("states", []) if isinstance(state, dict)]
    if not states or max_states <= 0:
        return []

    selected_indexes: set[int] = {0, len(states) - 1}
    query_tokens = _tokens(question)
    scored: list[tuple[int, int]] = []
    for index, state in enumerate(states):
        overlap = len(query_tokens & _tokens(json.dumps(state, ensure_ascii=False)))
        if overlap:
            scored.append((-overlap, index))
    for _negative_overlap, index in sorted(scored):
        selected_indexes.add(index)
        if len(selected_indexes) >= max_states:
            break
    return [states[index] for index in sorted(selected_indexes)[:max_states]]


def build_trajectory_capsule(trajectory: dict[str, Any], question: str, options: CapsuleOptions) -> str:
    parts = [
        f"Trajectory id: {trajectory.get('id')}",
        f"Domain: {trajectory.get('domain')}",
        f"Environment: {trajectory.get('environment')}",
        f"Goal: {_clip(trajectory.get('goal'), 500)}",
        f"Outcome: {_clip(trajectory.get('outcome'), 500)}",
    ]
    for state in select_states(trajectory, question, max_states=options.max_states):
        text = _state_text(state, options.max_chars_per_state)
        if text:
            parts.append(text)
    return _clip("\n".join(part for part in parts if part.strip()), options.max_chars_per_capsule)


def capsule_documents(
    trajectories: list[dict[str, Any]],
    question: str,
    options: CapsuleOptions,
) -> list[dict[str, str]]:
    documents: list[dict[str, str]] = []
    for trajectory in trajectories:
        trajectory_id = str(trajectory.get("id", len(documents)))
        documents.append(
            {
                "id": trajectory_id,
                "title": f"longmemeval-capsule-{trajectory_id}",
                "source": "longmemeval-v2",
                "text": build_trajectory_capsule(trajectory, question, options),
            }
        )
    return documents


def score_longmemeval_answer(prediction: str, gold: Any) -> float:
    pred = prediction.strip().lower()
    expected = str(gold).strip().lower()
    if not pred or not expected:
        return 0.0
    if expected in pred or pred in expected:
        return 1.0
    pred_nums = re.findall(r"-?\d+\.?\d*", pred)
    gold_nums = re.findall(r"-?\d+\.?\d*", expected)
    if pred_nums and gold_nums and pred_nums[-1] == gold_nums[-1]:
        return 1.0
    return 0.0


def _raw_trajectory_text(trajectory: dict[str, Any], max_states: int, max_chars: int) -> str:
    parts = [
        f"Trajectory id: {trajectory.get('id')}",
        f"Domain: {trajectory.get('domain')}",
        f"Environment: {trajectory.get('environment')}",
        f"Goal: {trajectory.get('goal')}",
        f"Outcome: {trajectory.get('outcome')}",
    ]
    for state in list(trajectory.get("states", []))[:max_states]:
        if not isinstance(state, dict):
            continue
        parts.append(
            " | ".join(
                str(value)
                for value in (
                    f"step={state.get('step')}",
                    f"url={state.get('url')}",
                    f"thought={state.get('thought')}",
                    f"action={state.get('action')}",
                    f"observation={state.get('observation')}",
                )
                if value is not None
            )
        )
    return "\n".join(parts)[:max_chars]


async def run_longmemeval_case(
    client: GraphknowsMCPClient,
    *,
    data_root: Path,
    question_id: str | None,
    run_id: str,
    top_k: int,
    max_trajectories: int,
    max_states: int,
    max_chars_per_trajectory: int,
    ingest_strategy: str,
    capsule_options: CapsuleOptions | None = None,
    purge: bool = True,
) -> BenchmarkCaseResult:
    questions, haystack = load_lme_metadata(data_root)
    question = select_lme_questions(questions, haystack, question_id=question_id, max_questions=1)[0]
    qid = str(question["id"])
    trajectory_ids = list(haystack[qid])
    trajectories = extract_lme_trajectories(data_root, trajectory_ids, max_trajectories)
    session_id = f"{run_id}-longmemeval-{qid}"
    options = capsule_options or CapsuleOptions(
        max_states=max_states,
        max_chars_per_state=max(500, max_chars_per_trajectory // max(max_states, 1)),
        max_chars_per_capsule=max_chars_per_trajectory,
    )
    recall: list[dict[str, Any]] = []
    ingest_results: list[dict[str, Any]] = []
    try:
        if ingest_strategy in {"trajectory_capsules", "batched_capsules"}:
            docs = capsule_documents(trajectories, str(question["question"]), options)
            if ingest_strategy == "batched_capsules":
                result = await client.call_tool(
                    "memory_ingest",
                    {
                        "text": BOUNDARY.join(doc["text"] for doc in docs),
                        "session_id": session_id,
                        "title": f"longmemeval-question-{qid}-capsules",
                    },
                )
                ingest_results.append(result if isinstance(result, dict) else {"result": result})
            else:
                for doc in docs:
                    result = await client.call_tool(
                        "memory_ingest",
                        {"text": doc["text"], "session_id": session_id, "title": doc["title"]},
                    )
                    ingest_results.append(result if isinstance(result, dict) else {"result": result})
        else:
            for trajectory in trajectories:
                result = await client.call_tool(
                    "memory_ingest",
                    {
                        "text": _raw_trajectory_text(
                            trajectory,
                            max_states=max_states,
                            max_chars=max_chars_per_trajectory,
                        ),
                        "session_id": session_id,
                        "title": f"longmemeval-trajectory-{trajectory.get('id')}",
                    },
                )
                ingest_results.append(result if isinstance(result, dict) else {"result": result})
        result = await client.call_tool(
            "memory_recall",
            {
                "query": str(question["question"]),
                "session_id": session_id,
                "top_k": top_k,
                "scope": "stm",
            },
        )
        recall = list(result.get("hits", []))
    finally:
        if purge:
            await client.call_tool("memory_purge", {"scope": "both", "session_id": session_id})

    answer = str(question.get("answer", "")).strip().lower()
    recalled_text = "\n".join(str(hit.get("text", "")) for hit in recall).lower()
    return BenchmarkCaseResult(
        case_id=qid,
        question=str(question["question"]),
        gold=question.get("answer"),
        memory_context=recall,
        metadata={
            "dataset": LONGMEMEVAL_DATASET,
            "question_id": qid,
            "domain": question.get("domain"),
            "question_type": question.get("question_type"),
            "official_haystack_size": len(trajectory_ids),
            "indexed_trajectories": len(trajectories),
            "ingest_strategy": ingest_strategy,
            "session_id": session_id,
            "ingest_documents": len(trajectories) if ingest_strategy != "raw" else len(trajectories),
            "ingest_results": ingest_results,
        },
        metrics={
            "recall_hit_count": len(recall),
            "answer_support_hit": bool(answer and answer in recalled_text),
            "ingest_calls": len(ingest_results),
            "ingested_chunks": sum(int(result.get("chunks", 0)) for result in ingest_results),
            "ingested_entities": sum(int(result.get("entities", 0)) for result in ingest_results),
        },
    )


async def run_longmemeval_small(
    client: GraphknowsMCPClient,
    *,
    data_root: Path,
    question_id: str | None,
    max_questions: int,
    run_id: str,
    top_k: int,
    max_trajectories: int,
    max_states: int,
    max_chars_per_trajectory: int,
    ingest_strategy: str,
    purge: bool = True,
) -> list[BenchmarkCaseResult]:
    questions, haystack = load_lme_metadata(data_root)
    selected = select_lme_questions(
        questions,
        haystack,
        question_id=question_id,
        max_questions=max_questions,
    )
    cases: list[BenchmarkCaseResult] = []
    for question in selected:
        cases.append(
            await run_longmemeval_case(
                client,
                data_root=data_root,
                question_id=str(question["id"]),
                run_id=run_id,
                top_k=top_k,
                max_trajectories=max_trajectories,
                max_states=max_states,
                max_chars_per_trajectory=max_chars_per_trajectory,
                ingest_strategy=ingest_strategy,
                purge=purge,
            )
        )
    return cases

"""Optional MemoryArena runner backed by graphknows MCP memory."""

from __future__ import annotations

import argparse
import asyncio
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from evaluation.graphknows_mcp_client import GraphknowsMCPClient

DOMAIN_CONFIGS = {
    "bundled_shopping": "bundled_shopping",
    "progressive_search": "progressive_search",
    "group_travel_planner": "group_travel_planner",
    "formal_reasoning_math": "formal_reasoning_math",
    "formal_reasoning_phys": "formal_reasoning_phys",
}


@dataclass
class MemoryArenaResult:
    """MemoryArena domain result."""

    domain: str
    avg_sr: float
    avg_ps: float
    tasks: list[dict[str, Any]]


def exact_match_score(pred: str, gold: Any) -> float:
    """Score shopping/search tasks with exact or target-ASIN matching."""
    if isinstance(gold, dict) and "target_asin" in gold:
        return float(str(gold["target_asin"]).strip().lower() in pred.lower())
    if isinstance(gold, str):
        return float(pred.strip().lower() == gold.strip().lower())
    return 0.0


def travel_plan_score(pred: str, gold: Any) -> float:
    """Score travel planner outputs by matching city per day."""
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
    """Score formal reasoning outputs by final numeric answer."""
    pred_nums = re.findall(r"-?\d+\.?\d*", pred)
    gold_nums = re.findall(r"-?\d+\.?\d*", str(gold))
    if not pred_nums or not gold_nums:
        return 0.0
    return float(abs(float(pred_nums[-1]) - float(gold_nums[-1])) < 1e-3)


def score_for_domain(domain: str, pred: str, gold: Any) -> float:
    """Dispatch MemoryArena scoring by domain."""
    if domain == "group_travel_planner":
        return travel_plan_score(pred, gold)
    if domain.startswith("formal_reasoning"):
        return numeric_score(pred, gold)
    return exact_match_score(pred, gold)


async def recall_context(
    client: GraphknowsMCPClient, session_id: str, query: str, top_k: int
) -> str:
    """Recall graphknows context for a MemoryArena subtask."""
    result = await client.call_tool(
        "memory_recall",
        {"query": query, "session_id": session_id, "top_k": top_k, "scope": "both"},
    )
    hits = result.get("hits", []) if isinstance(result, dict) else []
    return "\n".join(str(hit.get("text", "")) for hit in hits if isinstance(hit, dict))


async def store_interaction(
    client: GraphknowsMCPClient, session_id: str, question: str, answer: str
) -> None:
    """Store a MemoryArena subtask interaction in graphknows STM."""
    await client.call_tool(
        "memory_ingest",
        {
            "text": f"Q: {question}\nA: {answer}",
            "session_id": session_id,
            "title": "memoryarena-interaction",
        },
    )


async def evaluate_domain_with_predictions(
    client: GraphknowsMCPClient,
    domain: str,
    rows: list[dict[str, Any]],
    predictions: list[list[str]],
    session_prefix: str = "eval-memoryarena",
    top_k: int = 5,
) -> MemoryArenaResult:
    """Evaluate supplied model predictions while exercising graphknows memory I/O."""
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
            await store_interaction(client, session_id, str(question), pred)
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
        ps = (
            sum(float(row["score"]) for row in subtask_results) / len(subtask_results)
            if subtask_results
            else 0.0
        )
        task_results.append({"task_id": task.get("id", task_idx), "sr": sr, "ps": ps})
    avg_sr = sum(row["sr"] for row in task_results) / len(task_results) if task_results else 0.0
    avg_ps = sum(row["ps"] for row in task_results) / len(task_results) if task_results else 0.0
    return MemoryArenaResult(domain=domain, avg_sr=avg_sr, avg_ps=avg_ps, tasks=task_results)


async def amain(argv: list[str] | None = None) -> int:
    """Load MemoryArena datasets and optionally score supplied predictions."""
    parser = argparse.ArgumentParser(description="Run a MemoryArena graphknows adapter pass.")
    parser.add_argument("--command", default="graphknows-mcp")
    parser.add_argument("--domain", choices=sorted(DOMAIN_CONFIGS), default="bundled_shopping")
    parser.add_argument("--limit", type=int, default=2)
    parser.add_argument("--predictions-json", default="")
    parser.add_argument("--session-prefix", default="eval-memoryarena")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output", default="evaluation/results/memoryarena/result.json")
    args = parser.parse_args(argv)

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit("Install the eval dependency group to use MemoryArena.") from exc

    dataset = load_dataset("ZexueHe/memoryarena", DOMAIN_CONFIGS[args.domain])
    rows = list(dataset["test"])[: args.limit]
    if args.predictions_json:
        predictions = json.loads(Path(args.predictions_json).read_text(encoding="utf-8"))
        async with GraphknowsMCPClient(args.command) as client:
            result = await evaluate_domain_with_predictions(
                client,
                args.domain,
                rows,
                predictions,
                session_prefix=args.session_prefix,
                top_k=args.top_k,
            )
        payload = {
            "domain": result.domain,
            "avg_sr": result.avg_sr,
            "avg_ps": result.avg_ps,
            "tasks": result.tasks,
        }
    else:
        payload = {
            "domain": args.domain,
            "tasks_loaded": len(rows),
            "predictions_json_required": True,
        }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {out}")
    return 0


def main() -> None:
    """Run the MemoryArena adapter CLI."""
    raise SystemExit(asyncio.run(amain()))


if __name__ == "__main__":
    main()

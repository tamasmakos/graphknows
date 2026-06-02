"""Research harness for LongMemEval ingestion readiness.

The goal is to measure full-haystack ingestion strategies without changing the
production memory contract. Capsule construction is deterministic and LLM-free.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from evaluation.benchmarks.longmemeval import (
    BOUNDARY,
    CapsuleOptions,
    capsule_documents,
    extract_lme_trajectories,
    load_lme_metadata,
    select_lme_questions,
)
from evaluation.core.client import GraphknowsMCPClient


@dataclass(frozen=True)
class IngestMeasurement:
    strategy: str
    relation_extraction_mode: str
    question_id: str
    official_haystack_size: int
    indexed_trajectories: int
    ingest_calls: int
    ingest_elapsed_s: float
    recall_elapsed_s: float
    chunks: int
    entities: int
    cleanup_ok: bool
    timeout_or_error: str
    recall_hits: list[dict[str, Any]]

async def _ingest_documents(
    client: GraphknowsMCPClient,
    *,
    strategy: str,
    documents: list[dict[str, str]],
    session_id: str,
    batch_size: int,
) -> tuple[int, int, int]:
    calls = 0
    chunks = 0
    entities = 0
    if strategy == "per_trajectory":
        for doc in documents:
            result = await client.call_tool(
                "memory_ingest",
                {"text": doc["text"], "session_id": session_id, "title": doc["title"]},
            )
            calls += 1
            chunks += int(result.get("chunks", 0))
            entities += int(result.get("entities", 0))
        return calls, chunks, entities

    if strategy == "single_capsule_doc":
        batches = [documents]
    else:
        batches = [
            documents[index : index + batch_size]
            for index in range(0, len(documents), max(1, batch_size))
        ]

    for index, batch in enumerate(batches):
        text = BOUNDARY.join(doc["text"] for doc in batch)
        result = await client.call_tool(
            "memory_ingest",
            {
                "text": text,
                "session_id": session_id,
                "title": f"longmemeval-capsule-batch-{index}",
            },
        )
        calls += 1
        chunks += int(result.get("chunks", 0))
        entities += int(result.get("entities", 0))
    return calls, chunks, entities


async def run_lme_ingest_lab(
    *,
    command: str,
    data_root: Path,
    question_id: str | None,
    run_id: str,
    top_k: int,
    max_trajectories: int,
    strategy: str,
    relation_extraction_mode: str,
    request_timeout_s: float,
    batch_size: int,
    capsule_options: CapsuleOptions,
) -> IngestMeasurement:
    """Run one LongMemEval ingestion measurement."""
    questions, haystack = load_lme_metadata(data_root)
    question = select_lme_questions(
        questions,
        haystack,
        question_id=question_id,
        max_questions=1,
    )[0]
    qid = str(question["id"])
    trajectory_ids = list(haystack[qid])
    trajectories = extract_lme_trajectories(data_root, trajectory_ids, max_trajectories)
    session_id = f"{run_id}-longmemeval-{qid}"
    documents = capsule_documents(trajectories, str(question["question"]), capsule_options)

    cleanup_ok = False
    timeout_or_error = ""
    calls = chunks = entities = 0
    ingest_elapsed = recall_elapsed = 0.0
    recall_hits: list[dict[str, Any]] = []
    async with GraphknowsMCPClient(
        command,
        request_timeout_s=request_timeout_s,
        env={"RELATION_EXTRACTION_MODE": relation_extraction_mode},
    ) as client:
        try:
            started = time.perf_counter()
            calls, chunks, entities = await _ingest_documents(
                client,
                strategy=strategy,
                documents=documents,
                session_id=session_id,
                batch_size=batch_size,
            )
            ingest_elapsed = time.perf_counter() - started

            started = time.perf_counter()
            recall = await client.call_tool(
                "memory_recall",
                {
                    "query": str(question["question"]),
                    "session_id": session_id,
                    "top_k": top_k,
                    "scope": "stm",
                },
            )
            recall_elapsed = time.perf_counter() - started
            recall_hits = list(recall.get("hits", []))
        except Exception as exc:
            timeout_or_error = str(exc)
        finally:
            try:
                await client.call_tool("memory_purge", {"scope": "both", "session_id": session_id})
                cleanup_ok = True
            except Exception as exc:
                cleanup_ok = False
                timeout_or_error = f"{timeout_or_error}; cleanup failed: {exc}".strip("; ")

    return IngestMeasurement(
        strategy=strategy,
        relation_extraction_mode=relation_extraction_mode,
        question_id=qid,
        official_haystack_size=len(trajectory_ids),
        indexed_trajectories=len(trajectories),
        ingest_calls=calls,
        ingest_elapsed_s=round(ingest_elapsed, 3),
        recall_elapsed_s=round(recall_elapsed, 3),
        chunks=chunks,
        entities=entities,
        cleanup_ok=cleanup_ok,
        timeout_or_error=timeout_or_error,
        recall_hits=recall_hits,
    )


async def amain() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--command", default="graphknows-mcp")
    parser.add_argument("--run-id", default=f"eval-lme-lab-{int(time.time())}")
    parser.add_argument("--lme-data-root", default="evaluation/data/longmemeval-v2")
    parser.add_argument("--lme-question-id", default="")
    parser.add_argument("--lme-max-trajectories", type=int, default=100)
    parser.add_argument(
        "--strategy",
        choices=["per_trajectory", "batched_capsules", "single_capsule_doc"],
        default="batched_capsules",
    )
    parser.add_argument(
        "--relation-extraction-mode",
        choices=["gliner2", "llm_assisted", "off"],
        default="gliner2",
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--request-timeout-s", type=float, default=900.0)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--max-states", type=int, default=8)
    parser.add_argument("--max-chars-per-state", type=int, default=700)
    parser.add_argument("--max-chars-per-capsule", type=int, default=4000)
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    result = await run_lme_ingest_lab(
        command=args.command,
        data_root=Path(args.lme_data_root),
        question_id=args.lme_question_id or None,
        run_id=args.run_id,
        top_k=args.top_k,
        max_trajectories=args.lme_max_trajectories,
        strategy=args.strategy,
        relation_extraction_mode=args.relation_extraction_mode,
        request_timeout_s=args.request_timeout_s,
        batch_size=args.batch_size,
        capsule_options=CapsuleOptions(
            max_states=args.max_states,
            max_chars_per_state=args.max_chars_per_state,
            max_chars_per_capsule=args.max_chars_per_capsule,
        ),
    )
    payload = asdict(result)
    text = json.dumps(payload, indent=2, ensure_ascii=False)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")
    print(text)
    return 0


def main() -> None:
    raise SystemExit(asyncio.run(amain()))


if __name__ == "__main__":
    main()

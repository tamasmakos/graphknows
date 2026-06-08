"""Unified production entrypoint for graphknows evaluation benchmarks."""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from datetime import UTC, datetime
from dataclasses import replace
from pathlib import Path
from typing import Any

from evaluation.core.hotpotqa import run_hotpotqa_small
from evaluation.core.longmemeval import run_longmemeval_small
from evaluation.core.memoryarena import (
    DOMAIN_CONFIGS,
    evaluate_domain_with_predictions,
    fetch_memoryarena_rows,
    run_memoryarena_small,
)
from evaluation.core.native import NativeConfig, run_native_evaluation
from evaluation.core.research import ResearchMemoryOptions, run_research_memory
from evaluation.core.client import GraphknowsMCPClient
from evaluation.core.reporting import write_benchmark_report, write_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="benchmark", required=True)

    native = subparsers.add_parser("native", help="Run the native MCP harness.")
    native.add_argument("--command", default="graphknows-mcp")
    native.add_argument(
        "--run-id", default=f"eval-native-{datetime.now(UTC).strftime('%Y%m%d%H%M%S')}"
    )
    native.add_argument("--output-dir", default="evaluation/results/native")
    native.add_argument("--top-k", type=int, default=5)
    native.add_argument("--no-cleanup", action="store_true")
    native.add_argument("--scope", choices=["stm", "ltm", "both"], default="both")

    memoryarena = subparsers.add_parser("memoryarena", help="Run a MemoryArena evaluation slice.")
    memoryarena.add_argument("--command", default="graphknows-mcp")
    memoryarena.add_argument(
        "--run-id", default=f"eval-memoryarena-{datetime.now(UTC).strftime('%Y%m%d%H%M%S')}"
    )
    memoryarena.add_argument("--output-dir", default="evaluation/results/memoryarena")
    memoryarena.add_argument("--top-k", type=int, default=5)
    memoryarena.add_argument(
        "--relation-extraction-mode", choices=["gliner2", "llm_assisted", "off"], default="gliner2"
    )
    memoryarena.add_argument("--domain", choices=sorted(DOMAIN_CONFIGS), default="bundled_shopping")
    memoryarena.add_argument("--offset", type=int, default=0)
    memoryarena.add_argument("--limit", type=int, default=1)
    memoryarena.add_argument("--subtask-index", type=int, default=1)
    memoryarena.add_argument("--predictions-json", default="")
    memoryarena.add_argument("--session-prefix", default="eval-memoryarena")

    longmemeval = subparsers.add_parser("longmemeval", help="Run a LongMemEval evaluation slice.")
    longmemeval.add_argument("--command", default="graphknows-mcp")
    longmemeval.add_argument(
        "--run-id", default=f"eval-longmemeval-{datetime.now(UTC).strftime('%Y%m%d%H%M%S')}"
    )
    longmemeval.add_argument("--output-dir", default="evaluation/results/longmemeval")
    longmemeval.add_argument("--top-k", type=int, default=5)
    longmemeval.add_argument(
        "--relation-extraction-mode", choices=["gliner2", "llm_assisted", "off"], default="gliner2"
    )
    longmemeval.add_argument("--data-root", default="evaluation/data/longmemeval-v2")
    longmemeval.add_argument("--question-id", default="")
    longmemeval.add_argument("--max-questions", type=int, default=1)
    longmemeval.add_argument("--max-trajectories", type=int, default=5)
    longmemeval.add_argument("--max-states", type=int, default=8)
    longmemeval.add_argument("--max-chars-per-trajectory", type=int, default=4000)
    longmemeval.add_argument(
        "--ingest-strategy",
        choices=["raw", "trajectory_capsules", "batched_capsules"],
        default="batched_capsules",
    )

    hotpotqa = subparsers.add_parser("hotpotqa", help="Run a HotpotQA support-retrieval slice.")
    hotpotqa.add_argument("--command", default="graphknows-mcp")
    hotpotqa.add_argument(
        "--run-id", default=f"eval-hotpotqa-{datetime.now(UTC).strftime('%Y%m%d%H%M%S')}"
    )
    hotpotqa.add_argument("--output-dir", default="evaluation/results/hotpotqa")
    hotpotqa.add_argument("--top-k", type=int, default=5)
    hotpotqa.add_argument(
        "--relation-extraction-mode", choices=["gliner2", "llm_assisted", "off"], default="gliner2"
    )
    hotpotqa.add_argument("--offset", type=int, default=0)
    hotpotqa.add_argument("--limit", type=int, default=3)
    hotpotqa.add_argument("--max-docs", type=int, default=10)
    hotpotqa.add_argument("--max-chars-per-doc", type=int, default=2500)

    research = subparsers.add_parser(
        "research-memory",
        help="Run deterministic retrieval-only memory experiments with mandatory LTM topics.",
    )
    research.add_argument("--command", default="graphknows-mcp")
    research.add_argument(
        "--run-id", default=f"eval-research-memory-{datetime.now(UTC).strftime('%Y%m%d%H%M%S')}"
    )
    research.add_argument("--output-dir", default="evaluation/results/research-memory")
    research.add_argument(
        "--suite", choices=["hotpotqa", "longmemeval", "memoryarena"], default="hotpotqa"
    )
    research.add_argument("--top-k", type=int, default=5)
    research.add_argument(
        "--relation-extraction-mode", choices=["gliner2", "llm_assisted", "off"], default="gliner2"
    )
    research.add_argument("--offset", type=int, default=0)
    research.add_argument("--limit", type=int, default=1)
    research.add_argument("--domain", choices=sorted(DOMAIN_CONFIGS), default="bundled_shopping")
    research.add_argument("--subtask-index", type=int, default=1)
    research.add_argument("--data-root", default="evaluation/data/longmemeval-v2")
    research.add_argument("--question-id", default="")
    research.add_argument("--max-trajectories", type=int, default=5)
    research.add_argument("--max-states", type=int, default=8)
    research.add_argument("--max-chars-per-trajectory", type=int, default=4000)
    research.add_argument("--max-docs", type=int, default=10)
    research.add_argument("--max-chars-per-doc", type=int, default=2500)

    return parser


async def _run_memoryarena(args: argparse.Namespace) -> int:
    async with GraphknowsMCPClient(
        args.command,
        request_timeout_s=600,
        env={"RELATION_EXTRACTION_MODE": args.relation_extraction_mode},
    ) as client:
        if args.predictions_json:
            rows, total_rows = await fetch_memoryarena_rows(args.domain, args.offset, args.limit)
            predictions = json.loads(Path(args.predictions_json).read_text(encoding="utf-8"))
            result = await evaluate_domain_with_predictions(
                client,
                args.domain,
                rows,
                predictions,
                session_prefix=args.session_prefix,
                top_k=args.top_k,
            )
            cases = [
                {
                    "case_id": str(task["task_id"]),
                    "question": "MemoryArena prediction-backed task",
                    "gold": None,
                    "memory_context": [],
                    "metrics": {"sr": task["sr"], "ps": task["ps"]},
                    "score": float(task["ps"]),
                    "metadata": {"subtasks": task.get("subtasks", [])},
                }
                for task in result["tasks"]
            ]
            paths = write_benchmark_report(
                benchmark="memoryarena",
                run_id=args.run_id,
                output_dir=args.output_dir,
                cases=cases,
                metadata={
                    "domain": args.domain,
                    "official_rows": total_rows,
                    "mode": "predictions",
                },
            )
        else:
            cases = await run_memoryarena_small(
                client,
                domain=args.domain,
                offset=args.offset,
                limit=args.limit,
                subtask_index=args.subtask_index,
                run_id=args.run_id,
                top_k=args.top_k,
                purge=True,
            )
            paths = write_benchmark_report(
                benchmark="memoryarena",
                run_id=args.run_id,
                output_dir=args.output_dir,
                cases=[case.to_record() for case in cases],
                metadata={"domain": args.domain, "mode": "memory_retrieval"},
            )
    print(paths.json_path)
    return 0


async def _run_longmemeval(args: argparse.Namespace) -> int:
    async with GraphknowsMCPClient(
        args.command,
        request_timeout_s=900,
        env={"RELATION_EXTRACTION_MODE": args.relation_extraction_mode},
    ) as client:
        cases = await run_longmemeval_small(
            client,
            data_root=Path(args.data_root),
            question_id=args.question_id or None,
            max_questions=args.max_questions,
            run_id=args.run_id,
            top_k=args.top_k,
            max_trajectories=args.max_trajectories,
            max_states=args.max_states,
            max_chars_per_trajectory=args.max_chars_per_trajectory,
            ingest_strategy=args.ingest_strategy,
            purge=True,
        )
    paths = write_benchmark_report(
        benchmark="longmemeval",
        run_id=args.run_id,
        output_dir=args.output_dir,
        cases=[case.to_record() for case in cases],
        metadata={"ingest_strategy": args.ingest_strategy},
    )
    print(paths.json_path)
    return 0


async def _run_hotpotqa(args: argparse.Namespace) -> int:
    async with GraphknowsMCPClient(
        args.command,
        request_timeout_s=600,
        env={"RELATION_EXTRACTION_MODE": args.relation_extraction_mode},
    ) as client:
        cases = await run_hotpotqa_small(
            client,
            offset=args.offset,
            limit=args.limit,
            run_id=args.run_id,
            top_k=args.top_k,
            max_docs=args.max_docs,
            max_chars_per_doc=args.max_chars_per_doc,
            purge=True,
        )
    paths = write_benchmark_report(
        benchmark="hotpotqa",
        run_id=args.run_id,
        output_dir=args.output_dir,
        cases=[case.to_record() for case in cases],
        metadata={"limit": args.limit, "offset": args.offset},
    )
    print(paths.json_path)
    return 0


async def _run_research_memory(args: argparse.Namespace) -> int:
    async with GraphknowsMCPClient(
        args.command,
        request_timeout_s=1200,
        env={"RELATION_EXTRACTION_MODE": args.relation_extraction_mode},
    ) as client:
        options = ResearchMemoryOptions(
            suite=args.suite,
            run_id=args.run_id,
            top_k=args.top_k,
            offset=args.offset,
            limit=args.limit,
            domain=args.domain,
            subtask_index=args.subtask_index,
            data_root=Path(args.data_root),
            question_id=args.question_id or None,
            max_trajectories=args.max_trajectories,
            max_states=args.max_states,
            max_chars_per_trajectory=args.max_chars_per_trajectory,
            max_docs=args.max_docs,
            max_chars_per_doc=args.max_chars_per_doc,
        )
        cases = await run_research_memory(client, options)
    paths = write_benchmark_report(
        benchmark=f"research-memory-{args.suite}",
        run_id=args.run_id,
        output_dir=args.output_dir,
        cases=[case.to_record() for case in cases],
        metadata={
            "suite": args.suite,
            "relation_extraction_mode": args.relation_extraction_mode,
            "isolation": "per_case_pre_and_post_purge",
            "agentic_workflow": False,
            "topic_summarization": "mandatory",
            "headline_scores": [
                "answer_support_score",
                "evidence_recall_score",
                "topic_entity_score",
            ],
        },
    )
    print(paths.json_path)
    return 0


async def _run_native(args: argparse.Namespace) -> int:
    config = NativeConfig(
        command=args.command,
        output_dir=Path(args.output_dir),
        top_k=args.top_k,
        scope=args.scope,
        cleanup=not args.no_cleanup,
    )
    async with GraphknowsMCPClient(
        command=config.command,
        initialize=config.initialize,
        request_timeout_s=config.request_timeout_s,
    ) as client:
        run = await run_native_evaluation(client, config=config, run_id=args.run_id)
    paths = write_report(
        run.observations,
        output_dir=config.output_dir,
        run_id=run.run_id,
        top_k=config.top_k,
        metadata={"scope": config.scope, "cleanup": config.cleanup},
    )
    print(paths.json_path)
    return 0


async def amain(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.benchmark == "native":
        return await _run_native(args)
    if args.benchmark == "memoryarena":
        return await _run_memoryarena(args)
    if args.benchmark == "longmemeval":
        return await _run_longmemeval(args)
    if args.benchmark == "research-memory":
        return await _run_research_memory(args)
    return await _run_hotpotqa(args)


def main() -> None:
    raise SystemExit(asyncio.run(amain()))


if __name__ == "__main__":
    main()

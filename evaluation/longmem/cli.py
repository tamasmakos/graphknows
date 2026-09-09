"""LongMemEval CLI: ``python -m evaluation longmem``.

Knobs:
  --limit / --offset  slice of the question set to run
  --mode              llm_free | llm_assisted (default: llm_free)
  --compare           run BOTH modes over the same questions and write a
                      side-by-side per-category accuracy table

Everything else is in :mod:`evaluation.longmem.config`.
"""

from __future__ import annotations

import argparse
import asyncio
import os
from datetime import UTC, datetime
from pathlib import Path

from dotenv import load_dotenv

from evaluation.common.datamodels import RunReport
from evaluation.common.pipeline import (
    build_llms,
    drop_namespaces,
    run_units,
    unit_namespace,
)
from evaluation.common.reporting import ReportPaths, compare_modes, write_report
from evaluation.longmem.adapter import LongMemAdapter
from evaluation.longmem.config import LongMemConfig
from evaluation.longmem.dataset import load_units


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="evaluation longmem", description=__doc__)
    parser.add_argument("--limit", type=int, default=1, help="Number of questions to run.")
    parser.add_argument("--offset", type=int, default=0, help="Question index to start from.")
    parser.add_argument(
        "--mode",
        choices=["llm_free", "llm_assisted"],
        default="llm_free",
        help="MemoryMode passed to the MCP server (controls extraction + retrieval).",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help=(
            "Run BOTH llm_free and llm_assisted over the same questions and write a "
            "side-by-side per-category accuracy comparison table. Ignores --mode."
        ),
    )
    parser.add_argument(
        "--no-ingest",
        action="store_true",
        help="Reuse the already-ingested graph; skip ingest and only re-run retrieval+scoring.",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Drop this benchmark's namespace (both databases) for a clean re-ingest, then run.",
    )
    return parser


async def run_one(
    mode: str, limit: int, offset: int, ts: str, *, reuse_ingest: bool = False
) -> tuple[RunReport, ReportPaths]:
    """Run a single mode and return (RunReport, ReportPaths)."""
    os.environ["GRAPHKNOWS_MODE"] = mode
    config = LongMemConfig()
    # Per-mode namespace: llm_free and llm_assisted get physically separate
    # database pairs so ingesting one never contaminates the other's graph.
    config = config.model_copy(update={"namespace": f"{config.namespace}_{mode}"})
    os.environ["GRAPHKNOWS_NAMESPACE"] = config.namespace
    answerer, judge = build_llms(config)
    adapter = LongMemAdapter(config, answerer, judge)
    units = load_units(config, limit=limit, offset=offset)
    run_id = f"eval-{mode}-{ts}"
    try:
        run = await run_units(adapter, config, units, run_id=run_id, reuse_ingest=reuse_ingest)
    finally:
        await answerer.aclose()
        await judge.aclose()
    paths = write_report(run, config.output_dir)
    return run, paths


async def amain(argv: list[str] | None = None) -> int:
    load_dotenv(override=True)
    args = build_parser().parse_args(argv)

    if not os.environ.get("LLM_MODEL", "").strip():
        raise SystemExit("LLM_MODEL is not set in .env — set it before running the evaluation.")

    ts = datetime.now(UTC).strftime("%Y%m%d%H%M%S")

    if args.reset:
        base = LongMemConfig()
        modes = ["llm_free", "llm_assisted"] if args.compare else [args.mode]
        for m in modes:
            cfg = base.model_copy(update={"namespace": f"{base.namespace}_{m}"})
            units = load_units(cfg, limit=args.limit, offset=args.offset)
            namespaces = [unit_namespace(cfg, u.group_id) for u in units]
            print(f"[eval] --reset: dropping {len(namespaces)} per-conversation namespaces for {m}")
            await drop_namespaces(cfg, namespaces)

    if args.compare:
        print(
            f"[eval] --compare: running llm_free + llm_assisted  "
            f"limit={args.limit} offset={args.offset}"
        )
        run_free, paths_free = await run_one(
            "llm_free", args.limit, args.offset, ts, reuse_ingest=args.no_ingest
        )
        run_asst, paths_asst = await run_one(
            "llm_assisted", args.limit, args.offset, ts, reuse_ingest=args.no_ingest
        )

        table = compare_modes(run_free.cases, run_asst.cases)
        compare_path = Path(LongMemConfig().output_dir) / f"compare-{ts}.md"
        compare_path.parent.mkdir(parents=True, exist_ok=True)
        compare_path.write_text(table, encoding="utf-8")

        print("\n" + table)
        print(
            f"\nllm_free     accuracy: {run_free.summary.get('accuracy', 0.0):.3f}  "
            f"over {run_free.summary['case_count']} cases"
        )
        print(
            f"llm_assisted accuracy: {run_asst.summary.get('accuracy', 0.0):.3f}  "
            f"over {run_asst.summary['case_count']} cases"
        )
        print(f"\nComparison table: {compare_path}")
        print(f"llm_free  reports: {paths_free.json_path}")
        print(f"llm_asst  reports: {paths_asst.json_path}")
        return 0

    print(f"[eval] benchmark=longmem mode={args.mode}")
    run, paths = await run_one(args.mode, args.limit, args.offset, ts, reuse_ingest=args.no_ingest)

    print(
        f"\nAccuracy (LLM judge): {run.summary.get('accuracy', 0.0):.3f}  "
        f"(mean judge score: {run.summary.get('avg_judge_score', 0.0):.3f})  "
        f"over {run.summary['case_count']} cases"
    )
    for cat, acc in run.summary.get("accuracy_by_category", {}).items():
        print(f"  {cat}: {acc:.3f}")
    print(paths.json_path)
    print(paths.csv_path)
    print(paths.markdown_path)
    return 0


def main(argv: list[str] | None = None) -> None:
    raise SystemExit(asyncio.run(amain(argv)))

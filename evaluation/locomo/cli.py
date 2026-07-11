"""LoCoMo CLI: ``python -m evaluation locomo``.

Knobs:
  --limit / --offset  slice of the question set to run
  --mode              llm_free | llm_assisted (default: llm_free)
  --compare           run BOTH modes over the same questions and write a
                      side-by-side per-category accuracy table

Everything else is in :mod:`evaluation.locomo.config`.
"""

from __future__ import annotations

import argparse
import asyncio
import os
from datetime import UTC, datetime
from pathlib import Path

from dotenv import load_dotenv

from evaluation.common.datamodels import RunReport
from evaluation.common.pipeline import build_llms, reset_namespace, run_units
from evaluation.common.reporting import ReportPaths, compare_modes, write_report
from evaluation.locomo.adapter import LocomoAdapter
from evaluation.locomo.config import LocomoConfig
from evaluation.locomo.dataset import load_units


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="evaluation locomo", description=__doc__)
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


async def _run_one(
    mode: str, limit: int, offset: int, ts: str, *, reuse_ingest: bool = False
) -> tuple[RunReport, ReportPaths]:
    """Run a single mode and return (RunReport, ReportPaths)."""
    os.environ["GRAPHKNOWS_MODE"] = mode
    config = LocomoConfig()
    os.environ["GRAPHKNOWS_NAMESPACE"] = config.namespace
    answerer, judge = build_llms(config)
    adapter = LocomoAdapter(config, answerer, judge)
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
        os.environ["GRAPHKNOWS_NAMESPACE"] = LocomoConfig().namespace
        print(f"[eval] --reset: dropping namespace {LocomoConfig().namespace}")
        await reset_namespace(LocomoConfig())

    if args.compare:
        print(
            f"[eval] --compare: running llm_free + llm_assisted  "
            f"limit={args.limit} offset={args.offset}"
        )
        run_free, paths_free = await _run_one(
            "llm_free", args.limit, args.offset, ts, reuse_ingest=args.no_ingest
        )
        run_asst, paths_asst = await _run_one(
            "llm_assisted", args.limit, args.offset, ts, reuse_ingest=args.no_ingest
        )

        table = compare_modes(run_free.cases, run_asst.cases)
        compare_path = Path(LocomoConfig().output_dir) / f"compare-{ts}.md"
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

    print(f"[eval] benchmark=locomo mode={args.mode}")
    run, paths = await _run_one(
        args.mode, args.limit, args.offset, ts, reuse_ingest=args.no_ingest
    )

    print(
        f"\nAccuracy (LLM judge): {run.summary.get('accuracy', 0.0):.3f}  "
        f"(mean judge score: {run.summary.get('avg_judge_score', 0.0):.3f})  "
        f"over {run.summary['case_count']} cases"
    )
    metric_summary = run.summary.get("metric_summary", {})
    if "locomo_f1" in metric_summary:
        print(
            f"Lexical diagnostics: locomo_f1={metric_summary['locomo_f1']:.3f}  "
            f"token_f1={metric_summary.get('token_f1', 0.0):.3f}"
        )
    for cat, acc in run.summary.get("accuracy_by_category", {}).items():
        print(f"  {cat}: {acc:.3f}")
    print(paths.json_path)
    print(paths.csv_path)
    print(paths.markdown_path)
    return 0


def main(argv: list[str] | None = None) -> None:
    raise SystemExit(asyncio.run(amain(argv)))

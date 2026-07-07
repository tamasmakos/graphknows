"""Command-line entrypoint: ``python -m evaluation``.

Knobs:
  --limit / --offset  slice of the question set to run
  --mode              llm_free | llm_assisted (default: llm_free)
  --compare           run BOTH modes over the same questions and write a
                      side-by-side per-category SemanticF1 table

Everything else is in :mod:`evaluation.config`.
"""

from __future__ import annotations

import argparse
import asyncio
import os
from datetime import UTC, datetime
from pathlib import Path

from dotenv import load_dotenv

from evaluation.config import EvalConfig
from evaluation.pipeline import run_evaluation
from evaluation.reporting import compare_modes, write_report


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
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
            "side-by-side per-category SemanticF1 comparison table. Ignores --mode."
        ),
    )
    return parser


async def _run_one(
    mode: str,
    gen_model: str,
    limit: int,
    offset: int,
    ts: str,
) -> tuple[object, object]:
    """Run a single mode and return (RunReport, ReportPaths)."""
    os.environ["GRAPHKNOWS_MODE"] = mode
    config = EvalConfig(generation_model=gen_model)
    run_id = f"eval-{mode}-{ts}"
    run = await run_evaluation(config, run_id=run_id, limit=limit, offset=offset)
    paths = write_report(run)
    return run, paths


async def _amain(argv: list[str] | None = None) -> int:
    load_dotenv(override=True)
    args = _build_parser().parse_args(argv)

    gen_model = os.environ.get("LLM_MODEL", "").strip()
    if not gen_model:
        raise SystemExit("LLM_MODEL is not set in .env — set it before running the evaluation.")

    ts = datetime.now(UTC).strftime("%Y%m%d%H%M%S")

    if args.compare:
        print(
            f"[eval] --compare: running llm_free + llm_assisted  "
            f"limit={args.limit} offset={args.offset}  model={gen_model}"
        )

        run_free, paths_free = await _run_one("llm_free", gen_model, args.limit, args.offset, ts)
        run_asst, paths_asst = await _run_one(
            "llm_assisted", gen_model, args.limit, args.offset, ts
        )

        table = compare_modes(run_free.cases, run_asst.cases)

        compare_path = Path(EvalConfig().output_dir) / f"compare-{ts}.md"
        compare_path.parent.mkdir(parents=True, exist_ok=True)
        compare_path.write_text(table, encoding="utf-8")

        print("\n" + table)
        print(
            f"\nllm_free     SemanticF1: {run_free.summary.get('mean_semantic_f1', 0.0):.3f}  "
            f"over {run_free.summary['case_count']} cases"
        )
        print(
            f"llm_assisted SemanticF1: {run_asst.summary.get('mean_semantic_f1', 0.0):.3f}  "
            f"over {run_asst.summary['case_count']} cases"
        )
        print(f"\nComparison table: {compare_path}")
        print(f"llm_free  reports: {paths_free.json_path}")
        print(f"llm_asst  reports: {paths_asst.json_path}")
        return 0

    # Single-mode run.
    os.environ["GRAPHKNOWS_MODE"] = args.mode
    config = EvalConfig(generation_model=gen_model)
    print(f"[eval] mode={args.mode}  model={gen_model}  (from .env)")
    run_id = f"eval-{args.mode}-{ts}"

    run = await run_evaluation(config, run_id=run_id, limit=args.limit, offset=args.offset)
    paths = write_report(run)

    print(
        f"\nMean LoCoMo F1 (official): {run.summary.get('mean_locomo_f1', 0.0):.3f}  "
        f"(SemanticF1: {run.summary.get('mean_semantic_f1', 0.0):.3f}  "
        f"token-F1: {run.summary.get('mean_token_f1', 0.0):.3f})  "
        f"over {run.summary['case_count']} cases"
    )
    loco_cat = run.summary.get("locomo_f1_by_category", {})
    if loco_cat:
        print("LoCoMo F1 by category:")
        for cat, val in loco_cat.items():
            print(f"  {cat}: {val:.3f}")
    print(paths.json_path)
    print(paths.csv_path)
    print(paths.markdown_path)
    return 0


def main() -> None:
    raise SystemExit(asyncio.run(_amain()))


if __name__ == "__main__":
    main()

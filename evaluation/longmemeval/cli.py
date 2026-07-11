"""LongMemEval CLI: ``python -m evaluation longmemeval``.

Knobs:
  --per-type N        stratified sample of N questions per question type
                      (default 5 → 30 questions; deterministic, seed 42)
  --all-questions     run every question of the selected types (500 total)
  --question-types    comma-separated filter over the six question types
  --limit / --offset  cap on the sampled question list
  --mode              llm_free | llm_assisted (default: llm_free)

Each question ingests its own haystack (~40-60 sessions), so cost scales with
question count — start with the default sample.
"""

from __future__ import annotations

import argparse
import asyncio
import os
from datetime import UTC, datetime

from dotenv import load_dotenv

from evaluation.common.pipeline import build_llms, run_units
from evaluation.common.reporting import write_report
from evaluation.longmemeval.adapter import LongMemEvalAdapter
from evaluation.longmemeval.config import LongMemEvalConfig
from evaluation.longmemeval.dataset import QUESTION_TYPES, load_units


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="evaluation longmemeval", description=__doc__)
    parser.add_argument(
        "--per-type", type=int, default=5,
        help="Stratified sample size per question type (default 5).",
    )
    parser.add_argument(
        "--all-questions", action="store_true",
        help="Run every question of the selected types (overrides --per-type).",
    )
    parser.add_argument(
        "--question-types", type=str, default="",
        help=f"Comma-separated subset of: {', '.join(QUESTION_TYPES)}",
    )
    parser.add_argument("--limit", type=int, default=None, help="Cap on sampled questions.")
    parser.add_argument("--offset", type=int, default=0, help="Start index into the sample.")
    parser.add_argument(
        "--mode", choices=["llm_free", "llm_assisted"], default="llm_free",
        help="MemoryMode passed to the MCP server.",
    )
    return parser


async def amain(argv: list[str] | None = None) -> int:
    load_dotenv(override=True)
    args = build_parser().parse_args(argv)

    if not os.environ.get("LLM_MODEL", "").strip():
        raise SystemExit("LLM_MODEL is not set in .env — set it before running the evaluation.")

    os.environ["GRAPHKNOWS_MODE"] = args.mode
    config = LongMemEvalConfig()
    question_types = (
        [t.strip() for t in args.question_types.split(",") if t.strip()]
        if args.question_types
        else None
    )
    if question_types:
        unknown = set(question_types) - set(QUESTION_TYPES)
        if unknown:
            raise SystemExit(f"Unknown question types: {sorted(unknown)}")

    units = load_units(
        config,
        per_type=None if args.all_questions else args.per_type,
        question_types=question_types,
        limit=args.limit,
        offset=args.offset,
    )
    print(f"[eval] benchmark=longmemeval mode={args.mode} questions={len(units)}")

    answerer, judge = build_llms(config)
    adapter = LongMemEvalAdapter(config, answerer, judge)
    ts = datetime.now(UTC).strftime("%Y%m%d%H%M%S")
    run_id = f"eval-{args.mode}-{ts}"
    try:
        run = await run_units(adapter, config, units, run_id=run_id)
    finally:
        await answerer.aclose()
        await judge.aclose()
    paths = write_report(run, config.output_dir)

    print(
        f"\nAccuracy (LLM judge): {run.summary.get('accuracy', 0.0):.3f}  "
        f"over {run.summary['case_count']} cases"
    )
    for qtype, acc in run.summary.get("accuracy_by_category", {}).items():
        print(f"  {qtype}: {acc:.3f}")
    print(paths.json_path)
    print(paths.csv_path)
    print(paths.markdown_path)
    return 0


def main(argv: list[str] | None = None) -> None:
    raise SystemExit(asyncio.run(amain(argv)))

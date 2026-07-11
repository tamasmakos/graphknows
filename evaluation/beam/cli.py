"""BEAM CLI: ``python -m evaluation beam``.

Knobs:
  --chat-sizes      comma-separated size buckets: 100K,500K,1M,10M (default 100K)
  --conversations   index ranges, e.g. "0-4" or "0-9,50,90-99" (default 0-1)
  --question-types  comma-separated filter over the ten ability types
  --mode            llm_free | llm_assisted (default: llm_free)

A single conversation is 100K-10M tokens of chat, so ingestion dominates run
time — widen --conversations gradually.
"""

from __future__ import annotations

import argparse
import asyncio
import os
from datetime import UTC, datetime

from dotenv import load_dotenv

from evaluation.beam.adapter import BeamAdapter
from evaluation.beam.config import CHAT_SIZES, BeamConfig
from evaluation.beam.dataset import QUESTION_TYPES, load_units
from evaluation.common.pipeline import build_llms, run_units
from evaluation.common.reporting import write_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="evaluation beam", description=__doc__)
    parser.add_argument(
        "--chat-sizes", type=str, default="100K",
        help=f"Comma-separated size buckets from: {', '.join(CHAT_SIZES)} (default 100K).",
    )
    parser.add_argument(
        "--conversations", type=str, default="0-1",
        help='Conversation index ranges, e.g. "0-4" or "0-9,50,90-99" (default "0-1").',
    )
    parser.add_argument(
        "--question-types", type=str, default="",
        help=f"Comma-separated subset of: {', '.join(QUESTION_TYPES)}",
    )
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

    chat_sizes = [s.strip() for s in args.chat_sizes.split(",") if s.strip()]
    unknown_sizes = set(chat_sizes) - set(CHAT_SIZES)
    if unknown_sizes:
        raise SystemExit(f"Unknown chat sizes: {sorted(unknown_sizes)}")
    question_types = (
        [t.strip() for t in args.question_types.split(",") if t.strip()]
        if args.question_types
        else None
    )
    if question_types:
        unknown = set(question_types) - set(QUESTION_TYPES)
        if unknown:
            raise SystemExit(f"Unknown question types: {sorted(unknown)}")

    os.environ["GRAPHKNOWS_MODE"] = args.mode
    config = BeamConfig()
    units = load_units(
        config,
        chat_sizes=chat_sizes,
        conversations=args.conversations,
        question_types=question_types,
    )
    total_q = sum(len(u.cases) for u in units)
    print(
        f"[eval] benchmark=beam mode={args.mode} "
        f"conversations={len(units)} questions={total_q}"
    )

    answerer, judge = build_llms(config)
    adapter = BeamAdapter(config, answerer, judge)
    ts = datetime.now(UTC).strftime("%Y%m%d%H%M%S")
    run_id = f"eval-{args.mode}-{ts}"
    try:
        run = await run_units(adapter, config, units, run_id=run_id)
    finally:
        await answerer.aclose()
        await judge.aclose()
    paths = write_report(run, config.output_dir)

    print(
        f"\nPass rate (nugget judge, ≥0.5): {run.summary.get('accuracy', 0.0):.3f}  "
        f"(mean rubric score: {run.summary.get('avg_judge_score', 0.0):.3f})  "
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

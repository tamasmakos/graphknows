"""Command-line entrypoint: ``python -m evaluation``.

The only knobs are ``--limit`` / ``--offset`` (which slice of the question set
to run). Everything else is the hardcoded workflow in :mod:`evaluation.config`.
"""

from __future__ import annotations

import argparse
import asyncio
import os
from datetime import UTC, datetime

from dotenv import load_dotenv

from evaluation.config import EvalConfig
from evaluation.pipeline import run_evaluation
from evaluation.reporting import write_report


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
    return parser


async def _amain(argv: list[str] | None = None) -> int:
    load_dotenv(override=True)  # .env wins over the container/shell env (stale model footgun)
    args = _build_parser().parse_args(argv)
    os.environ["MODE"] = args.mode
    # Resolve the generation+judge model from .env explicitly and pass it through,
    # so a stale shell/compose LLM_MODEL can never silently leak into the run.
    gen_model = os.environ.get("LLM_MODEL", "").strip()
    if not gen_model:
        raise SystemExit("LLM_MODEL is not set in .env — set it before running the evaluation.")
    config = EvalConfig(generation_model=gen_model)
    print(f"[eval] mode={args.mode}  generation+judge model={gen_model}  (from .env)")
    run_id = f"eval-{args.mode}-{datetime.now(UTC).strftime('%Y%m%d%H%M%S')}"

    run = await run_evaluation(config, run_id=run_id, limit=args.limit, offset=args.offset)
    paths = write_report(run)

    print(f"\nMean token-F1: {run.summary['avg_score']:.3f} over {run.summary['case_count']} cases")
    print(paths.json_path)
    print(paths.csv_path)
    print(paths.markdown_path)
    return 0


def main() -> None:
    raise SystemExit(asyncio.run(_amain()))


if __name__ == "__main__":
    main()

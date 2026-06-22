"""Command-line entrypoint: ``python -m evaluation``.

The only knobs are ``--limit`` / ``--offset`` (which slice of the question set
to run). Everything else is the hardcoded workflow in :mod:`evaluation.config`.
"""

from __future__ import annotations

import argparse
import asyncio
from datetime import UTC, datetime

from dotenv import load_dotenv

from evaluation.config import EvalConfig
from evaluation.pipeline import run_evaluation
from evaluation.reporting import write_report


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=1, help="Number of questions to run.")
    parser.add_argument("--offset", type=int, default=0, help="Question index to start from.")
    return parser


async def _amain(argv: list[str] | None = None) -> int:
    load_dotenv()
    args = _build_parser().parse_args(argv)
    config = EvalConfig()
    run_id = f"eval-{datetime.now(UTC).strftime('%Y%m%d%H%M%S')}"

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

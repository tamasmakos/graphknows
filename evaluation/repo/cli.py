"""Repository-transcript CLI: ``python -m evaluation repo``.

Knobs:
  --limit / --offset  slice of the gold questions to run
  --mode              llm_free | llm_assisted (default: llm_free)

Everything else is in :mod:`evaluation.repo.config`. The row is compared
against its own recorded runs by ``python -m evaluation panel``, so this CLI
carries no --compare of its own.
"""

from __future__ import annotations

import argparse
import asyncio
import os
from datetime import UTC, datetime

from dotenv import load_dotenv

from evaluation.common.datamodels import RunReport
from evaluation.common.pipeline import build_llms, run_units
from evaluation.common.reporting import ReportPaths, write_report
from evaluation.repo.adapter import RepoAdapter
from evaluation.repo.config import RepoConfig
from evaluation.repo.dataset import load_units


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="evaluation repo", description=__doc__)
    parser.add_argument("--limit", type=int, default=1, help="Number of questions to run.")
    parser.add_argument("--offset", type=int, default=0, help="Question index to start from.")
    parser.add_argument(
        "--mode",
        choices=["llm_free", "llm_assisted"],
        default="llm_free",
        help="MemoryMode passed to the MCP server (controls extraction + retrieval).",
    )
    parser.add_argument(
        "--no-ingest",
        action="store_true",
        help="Reuse the already-ingested graph; skip ingest and only re-run retrieval+scoring.",
    )
    return parser


async def run_one(
    mode: str, limit: int, offset: int, ts: str, *, reuse_ingest: bool = False
) -> tuple[RunReport, ReportPaths]:
    """Run a single mode and return (RunReport, ReportPaths)."""
    os.environ["GRAPHKNOWS_MODE"] = mode
    # Per-mode namespace: llm_free and llm_assisted get physically separate
    # database pairs so ingesting one never contaminates the other's graph.
    config = RepoConfig()
    config = config.model_copy(update={"namespace": f"{config.namespace}_{mode}"})
    os.environ["GRAPHKNOWS_NAMESPACE"] = config.namespace
    answerer, judge = build_llms(config)
    adapter = RepoAdapter(config, answerer, judge)
    units = load_units(config, limit=limit, offset=offset)
    try:
        run = await run_units(
            adapter, config, units, run_id=f"eval-{mode}-{ts}", reuse_ingest=reuse_ingest
        )
    finally:
        await answerer.aclose()
        await judge.aclose()
    return run, write_report(run, config.output_dir)


async def amain(argv: list[str] | None = None) -> int:
    load_dotenv(override=True)
    args = build_parser().parse_args(argv)

    if not os.environ.get("LLM_MODEL", "").strip():
        raise SystemExit("LLM_MODEL is not set in .env — set it before running the evaluation.")

    ts = datetime.now(UTC).strftime("%Y%m%d%H%M%S")
    print(f"[eval] benchmark=repo mode={args.mode}")
    run, paths = await run_one(args.mode, args.limit, args.offset, ts, reuse_ingest=args.no_ingest)

    print(
        f"\nAccuracy (LLM judge): {run.summary.get('accuracy', 0.0):.3f}  "
        f"over {run.summary['case_count']} cases"
    )
    for kind, acc in run.summary.get("accuracy_by_category", {}).items():
        print(f"  {kind}: {acc:.3f}")
    print(paths.json_path)
    return 0


def main(argv: list[str] | None = None) -> None:
    raise SystemExit(asyncio.run(amain(argv)))

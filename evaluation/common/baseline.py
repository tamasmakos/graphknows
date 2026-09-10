"""Baseline CLI: ``python -m evaluation baseline --rows locomo,longmem,beam --repeats 3``.

Records what the current core scores, so every later change is judged against a
number rather than a feeling (FR-041). Each row runs *--repeats* times through
that benchmark's own ``run_one`` — no second runner — and the repeats collapse
into the contract's ``median``/``band``/``runs``/``commit`` block per metric,
one JSON file per row under ``evaluation/results/baseline/``
(contracts/panel-report.md).

A row that cannot be provisioned is written with its ``not_run`` reason. It is
never omitted and never estimated: the baseline's job is to write down what the
old core does, gaps included.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from evaluation.common.reporting import BaselineMetric, PanelRow

# Row -> the benchmark CLI whose `run_one` drives it, as `evaluation.__main__`
# maps commands to modules. The metrics a baseline records, per row.
ROW_MODULES = {
    "locomo": "evaluation.locomo.cli",
    "longmem": "evaluation.longmem.cli",
    "beam": "evaluation.beam.cli",
    "repo": "evaluation.repo.cli",
}
EXTRACTORS = {"llm_free": "local", "llm_assisted": "llm"}
_METRICS = ("accuracy", "evidence_recall")
_DEFAULT_OUT = Path(__file__).resolve().parents[2] / "evaluation" / "results" / "baseline"


def row_list(value: str) -> list[str]:
    rows = [row.strip() for row in value.split(",") if row.strip()]
    if unknown := sorted(set(rows) - set(ROW_MODULES)):
        raise argparse.ArgumentTypeError(f"unknown rows: {', '.join(unknown)}")
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="evaluation baseline", description=__doc__)
    parser.add_argument(
        "--rows",
        type=row_list,
        default=list(ROW_MODULES),
        help=f"Comma-separated panel rows ({', '.join(ROW_MODULES)}).",
    )
    parser.add_argument("--repeats", type=int, default=3, help="Runs per row (contract: 3).")
    parser.add_argument("--limit", type=int, default=1, help="Questions per row and run.")
    parser.add_argument("--mode", choices=sorted(EXTRACTORS), default="llm_free")
    parser.add_argument(
        "--out", type=Path, default=_DEFAULT_OUT, help="Where the rows are written."
    )
    return parser


def head_commit() -> str:
    """The commit the baseline is recorded against."""
    proc = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
    return proc.stdout.strip()


async def run_row(row: str, args: argparse.Namespace, *, reuse_ingest: bool = False) -> PanelRow:
    """Run one row once and report it in the panel's row shape.

    ``reuse_ingest`` answers repeats two and three against the graph repeat one
    built: the band measures generation and judge noise, not re-ingest.
    """
    extractor = EXTRACTORS[args.mode]
    module = importlib.import_module(ROW_MODULES[row])
    ts = datetime.now(UTC).strftime("%Y%m%d%H%M%S")
    try:
        run, _paths = await module.run_one(args.mode, args.limit, 0, ts, reuse_ingest=reuse_ingest)
    except OSError as exc:  # an unprovisioned corpus is recorded, not dropped
        return PanelRow(row=row, feeding_mode="", extractor=extractor, not_run=str(exc))
    summary = run.summary
    return PanelRow(
        row=row,
        feeding_mode=summary.get("feeding_mode", ""),
        extractor=extractor,
        accuracy=float(summary.get("accuracy", 0.0)),
        evidence_recall=float(
            summary.get("metric_summary", {}).get("evidence_recall_in_context", 0.0)
        ),
        by_category=summary.get("accuracy_by_category", {}),
        n=int(summary.get("case_count", 0)),
    )


def record_row(repeats: list[PanelRow], commit: str) -> dict[str, Any]:
    """Collapse a row's repeated runs into one baseline record per metric."""
    last = repeats[-1]
    record: dict[str, Any] = {
        "row": last.row,
        "feeding_mode": last.feeding_mode,
        "extractor": last.extractor,
        "n": last.n,
        "not_run": last.not_run,
    }
    for metric in _METRICS:
        runs = [float(getattr(row, metric)) for row in repeats]
        record[metric] = BaselineMetric(runs=runs, commit=commit).to_record()
    return record


async def amain(argv: list[str] | None = None) -> int:
    load_dotenv(override=True)
    args = build_parser().parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    commit = head_commit()

    for row in args.rows:
        repeats = [await run_row(row, args, reuse_ingest=i > 0) for i in range(args.repeats)]
        path = args.out / f"{row}.json"
        path.write_text(json.dumps(record_row(repeats, commit), indent=2), encoding="utf-8")
        print(f"[baseline] {row}: {args.repeats} runs -> {path}")
    return 0


def main(argv: list[str] | None = None) -> None:
    raise SystemExit(asyncio.run(amain(argv)))

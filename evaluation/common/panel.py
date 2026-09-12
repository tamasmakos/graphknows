"""Panel CLI: ``python -m evaluation panel --compare evaluation/results/baseline``.

The merge gate (FR-042, SC-004). One run of the whole panel clears it only when all three
of the contract's rules hold (contracts/panel-report.md):

1. no row falls below its baseline ``median - band`` on accuracy or evidence recall,
2. identity pairwise precision is at or above its baseline median — no band, it must not
   drop at all,
3. the run's dead weight is clean.

Every failing check is named at once rather than the first one found: no single row is the
headline, so a change is judged on the panel as a whole. Rows are run through
``evaluation.common.baseline`` — the same runner the baseline was recorded with — and a row
with no baseline yet is recorded from this run and gated against it from then on (FR-041).
"""

from __future__ import annotations

import argparse
import asyncio
import json
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from evaluation.common import baseline
from evaluation.common.reporting import InternalMetrics, PanelRow
from evaluation.deadweight import DeadWeight

_ROW_METRICS = ("accuracy", "evidence_recall")
# Baselines are recorded rounded; compare at the same resolution so float noise
# (0.9 - 0.09999999999999998) is never read as a regression.
_PLACES = 4


@dataclass(frozen=True)
class PanelRun:
    """One run of the whole panel: every row, plus the once-per-run internal metrics."""

    rows: tuple[PanelRow, ...]
    internal: InternalMetrics = field(default_factory=InternalMetrics)


@dataclass(frozen=True)
class GateVerdict:
    """Why a panel run does or does not clear the merge gate."""

    regressions: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return not self.regressions

    def gate(self) -> None:
        """Fail the run when the panel regressed; a no-op when it holds."""
        if not self.passed:
            raise SystemExit("merge gate: " + "; ".join(self.regressions))


def _baseline_record(directory: Path, name: str) -> dict[str, Any] | None:
    """The committed baseline for *name*, or None when none was ever recorded."""
    path = directory / f"{name}.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def _row_regressions(row: PanelRow, record: dict[str, Any]) -> Iterator[str]:
    """Rule 1: neither metric of *row* may fall below ``median - band``."""
    for metric in _ROW_METRICS:
        block = record[metric]
        floor = round(block["median"] - block["band"], _PLACES)
        value = round(float(getattr(row, metric)), _PLACES)
        if value < floor:
            yield f"{row.row} {metric} {value} below median-band {floor}"


def _identity_regression(internal: InternalMetrics, directory: Path) -> str | None:
    """Rule 2: identity precision at or above the baseline median — no band (SC-004)."""
    record = _baseline_record(directory, "internal")
    if record is None:
        return None
    floor = round(record["identity"]["median"], _PLACES)
    value = round(float(internal.identity.get("pairwise_precision", 0.0)), _PLACES)
    if value < floor:
        return f"identity pairwise_precision {value} below baseline median {floor}"
    return None


def _dead_weight_regression(internal: InternalMetrics) -> str | None:
    """Rule 3: nothing written that no reader reads (FR-024, SC-005)."""
    if "unread_types" not in internal.dead_weight:
        return "dead weight not recorded for this run"
    dead = DeadWeight(tuple(internal.dead_weight["unread_types"]))
    if dead.passed:
        return None
    return f"dead weight: written but never read: {', '.join(dead.unread_types)}"


def compare(run: PanelRun, directory: Path) -> GateVerdict:
    """Gate *run* against the baseline committed in *directory*, naming every failure."""
    regressions = [
        line
        for row in run.rows
        if not row.not_run and (record := _baseline_record(directory, row.row))
        for line in _row_regressions(row, record)
    ]
    regressions += [
        line
        for line in (
            _identity_regression(run.internal, directory),
            _dead_weight_regression(run.internal),
        )
        if line
    ]
    return GateVerdict(tuple(regressions))


def _seed_baseline(row: PanelRow, directory: Path) -> None:
    """A row introduced after the baseline is gated against its own first run (FR-041)."""
    path = directory / f"{row.row}.json"
    if row.not_run or path.is_file():
        return
    record = baseline.record_row([row], baseline.head_commit())
    path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(f"[panel] {row.row}: no baseline, recorded this run as its own -> {path}")


def _internal_metrics(path: Path | None) -> InternalMetrics:
    """This run's internal metrics, as written by the identity, audit and dead-weight steps."""
    if path is None:
        return InternalMetrics()
    return InternalMetrics(**json.loads(path.read_text(encoding="utf-8")))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="evaluation panel", description=__doc__)
    parser.add_argument(
        "--compare",
        type=Path,
        required=True,
        help="Baseline directory to gate this run against.",
    )
    parser.add_argument(
        "--internal",
        type=Path,
        help="JSON in the internal-metrics shape for this run (identity, fact precision, "
        "dead weight). Without it the run has no dead-weight result and cannot merge.",
    )
    parser.add_argument(
        "--rows",
        type=baseline.row_list,
        default=list(baseline.ROW_MODULES),
        help=f"Comma-separated panel rows ({', '.join(baseline.ROW_MODULES)}).",
    )
    parser.add_argument("--limit", type=int, default=1, help="Questions per row.")
    parser.add_argument("--mode", choices=sorted(baseline.EXTRACTORS), default="llm_free")
    return parser


async def amain(argv: list[str] | None = None) -> int:
    load_dotenv(override=True)
    args = build_parser().parse_args(argv)
    rows = [await baseline.run_row(row, args) for row in args.rows]
    for row in rows:
        _seed_baseline(row, args.compare)

    verdict = compare(PanelRun(tuple(rows), _internal_metrics(args.internal)), args.compare)
    for line in verdict.regressions:
        print(f"[panel] {line}")
    verdict.gate()
    print("[panel] merge gate: clear")
    return 0


def main(argv: list[str] | None = None) -> None:
    raise SystemExit(asyncio.run(amain(argv)))

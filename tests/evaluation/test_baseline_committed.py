"""The baseline is committed for every panel row and internal metric (FR-041, SC-003).

Phase 2 exists to put a number on disk, against the unmodified core, before the first line
of new core code. This guards the committed files themselves — that they are there, carry
the contract's ``median``/``band``/``runs``/``commit`` per metric and were all recorded
against one commit (contracts/panel-report.md) — not the scores inside them.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any

import pytest

pytestmark = pytest.mark.unit

BASELINE_DIR = Path(__file__).resolve().parents[2] / "evaluation" / "results" / "baseline"
ROWS = ("locomo", "longmem", "beam")
INTERNAL_METRICS = ("identity", "fact_precision", "dead_weight")
ROW_METRICS = ("accuracy", "evidence_recall")


def _record(name: str) -> dict[str, Any]:
    path = BASELINE_DIR / f"{name}.json"
    assert path.is_file(), f"baseline not committed: {path}"
    return json.loads(path.read_text(encoding="utf-8"))


def _assert_metric(record: dict[str, Any], metric: str) -> None:
    """*metric* is recorded in the contract's median/band/runs/commit shape."""
    block = record[metric]
    assert set(block) == {"median", "band", "runs", "commit"}
    runs = block["runs"]
    assert runs, f"{metric} has no runs: a baseline is runs, not a single number"
    assert block["median"] == pytest.approx(statistics.median(runs))
    assert block["band"] == pytest.approx(max(runs) - min(runs))
    assert len(block["commit"]) == 40, "commit must be the full baseline sha"


@pytest.mark.parametrize("row", ROWS)
def test_every_panel_row_is_recorded(row: str) -> None:
    record = _record(row)
    assert record["row"] == row
    assert record["extractor"] in {"local", "llm"}
    for metric in ROW_METRICS:
        _assert_metric(record, metric)


@pytest.mark.parametrize("metric", INTERNAL_METRICS)
def test_every_internal_metric_is_recorded(metric: str) -> None:
    _assert_metric(_record("internal"), metric)


def test_one_baseline_commit_across_every_file() -> None:
    """A baseline split over two commits is not a baseline."""
    records = [_record(name) for name in (*ROWS, "internal")]
    commits = {
        block["commit"]
        for record in records
        for block in record.values()
        if isinstance(block, dict) and "commit" in block
    }
    assert len(commits) == 1, f"baseline recorded against several commits: {sorted(commits)}"

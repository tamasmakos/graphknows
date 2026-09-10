"""``python -m evaluation baseline`` writes the contract's baseline files.

Per row: one JSON file whose every metric carries ``median``/``band``/``runs``/
``commit`` over *--repeats* runs (contracts/panel-report.md). The rows are
driven through each benchmark CLI's ``run_one``, faked here — the shape of the
recorded file is what this guards, not the benchmarks themselves.
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

from evaluation.common import baseline
from evaluation.common.datamodels import RunReport

pytestmark = pytest.mark.unit


def _report(accuracy: float) -> RunReport:
    return RunReport(
        run_id="eval-test",
        benchmark="locomo",
        summary={
            "case_count": 7,
            "accuracy": accuracy,
            "accuracy_by_category": {"single_hop": accuracy},
            "feeding_mode": "turn_by_turn",
            "metric_summary": {"evidence_recall_in_context": 0.9},
        },
    )


@pytest.fixture
def fake_rows(monkeypatch):
    """Serve every row from a fake benchmark CLI whose accuracy varies per run."""
    accuracies = iter([0.4, 0.6, 0.5])

    async def run_one(mode, limit, offset, ts, **kwargs):
        return _report(next(accuracies)), None

    monkeypatch.setattr(baseline, "head_commit", lambda: "abc123")
    monkeypatch.setattr(
        baseline.importlib, "import_module", lambda name: SimpleNamespace(run_one=run_one)
    )


def test_each_row_is_written_with_median_band_runs_and_commit(fake_rows, tmp_path) -> None:
    asyncio.run(baseline.amain(["--rows", "locomo", "--repeats", "3", "--out", str(tmp_path)]))

    record = json.loads((tmp_path / "locomo.json").read_text(encoding="utf-8"))
    assert record["row"] == "locomo"
    assert record["feeding_mode"] == "turn_by_turn"
    assert record["extractor"] == "local"
    assert record["n"] == 7
    assert record["accuracy"] == {
        "median": 0.5,
        "band": pytest.approx(0.2),
        "runs": [0.4, 0.6, 0.5],
        "commit": "abc123",
    }
    assert record["evidence_recall"]["median"] == 0.9
    assert record["evidence_recall"]["runs"] == [0.9, 0.9, 0.9]


def test_a_row_that_cannot_run_isrecorded_not_omitted(monkeypatch, tmp_path) -> None:
    async def run_one(mode, limit, offset, ts, **kwargs):
        raise FileNotFoundError("corpus not downloaded")

    monkeypatch.setattr(baseline, "head_commit", lambda: "abc123")
    monkeypatch.setattr(
        baseline.importlib, "import_module", lambda name: SimpleNamespace(run_one=run_one)
    )

    asyncio.run(baseline.amain(["--rows", "beam", "--repeats", "1", "--out", str(tmp_path)]))

    record = json.loads((tmp_path / "beam.json").read_text(encoding="utf-8"))
    assert record["not_run"] == "corpus not downloaded"
    assert record["accuracy"]["runs"] == [0.0]


def test_unknown_row_is_rejected() -> None:
    with pytest.raises(SystemExit):
        baseline.build_parser().parse_args(["--rows", "nonsense"])

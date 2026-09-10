"""The merge gate: ``panel --compare`` against the committed baseline (FR-042, SC-004).

Guards the three rules of contracts/panel-report.md — no row below ``median - band``,
identity precision not dropped, dead weight clean — and that a failing panel names every
failure at once, since no single row is the headline.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from evaluation.common.panel import PanelRun, compare
from evaluation.common.reporting import InternalMetrics, PanelRow

pytestmark = pytest.mark.unit

CLEAN = {"unread_types": [], "passed": True}


def _metric(median: float, band: float) -> dict[str, object]:
    return {"median": median, "band": band, "runs": [median], "commit": "a" * 40}


@pytest.fixture
def baseline_dir(tmp_path: Path) -> Path:
    """A committed baseline: locomo at 0.9 ± 0.1 accuracy, identity precision 0.8."""
    (tmp_path / "locomo.json").write_text(
        json.dumps(
            {
                "row": "locomo",
                "accuracy": _metric(0.9, 0.09999999999999998),
                "evidence_recall": _metric(0.6, 0.0),
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "internal.json").write_text(
        json.dumps({"identity": _metric(0.8, 0.0)}), encoding="utf-8"
    )
    return tmp_path


def _run(
    *,
    accuracy: float = 0.9,
    evidence_recall: float = 0.6,
    precision: float = 0.8,
    row: str = "locomo",
    not_run: str | None = None,
    dead_weight: dict[str, object] | None = None,
) -> PanelRun:
    return PanelRun(
        rows=(
            PanelRow(
                row=row,
                feeding_mode="turn_by_turn",
                extractor="local",
                accuracy=accuracy,
                evidence_recall=evidence_recall,
                not_run=not_run,
            ),
        ),
        internal=InternalMetrics(
            identity={"pairwise_precision": precision},
            dead_weight=CLEAN if dead_weight is None else dead_weight,
        ),
    )


class TestRowFloor:
    def test_a_panel_that_holds_clears_the_gate(self, baseline_dir: Path) -> None:
        verdict = compare(_run(), baseline_dir)
        assert verdict.passed
        assert verdict.gate() is None

    def test_exactly_at_median_minus_band_still_clears(self, baseline_dir: Path) -> None:
        """0.9 - 0.09999999999999998 is 0.8: float noise is not a regression."""
        assert compare(_run(accuracy=0.8), baseline_dir).passed

    def test_accuracy_below_median_minus_band_regresses(self, baseline_dir: Path) -> None:
        verdict = compare(_run(accuracy=0.79), baseline_dir)
        assert not verdict.passed
        assert "locomo accuracy" in verdict.regressions[0]

    def test_evidence_recall_is_gated_too(self, baseline_dir: Path) -> None:
        verdict = compare(_run(evidence_recall=0.5), baseline_dir)
        assert [line.split()[1] for line in verdict.regressions] == ["evidence_recall"]

    def test_a_row_that_could_not_run_is_not_gated(self, baseline_dir: Path) -> None:
        assert compare(_run(accuracy=0.0, not_run="corpus missing"), baseline_dir).passed

    def test_a_row_with_no_baseline_is_not_gated(self, baseline_dir: Path) -> None:
        """The repo row arrives after the baseline; its own first run becomes its floor."""
        assert compare(_run(row="repo", accuracy=0.0), baseline_dir).passed


class TestIdentityPrecision:
    def test_equal_to_the_baseline_median_clears(self, baseline_dir: Path) -> None:
        assert compare(_run(precision=0.8), baseline_dir).passed

    def test_any_drop_regresses_there_is_no_band(self, baseline_dir: Path) -> None:
        verdict = compare(_run(precision=0.79), baseline_dir)
        assert not verdict.passed
        assert "identity pairwise_precision" in verdict.regressions[0]


class TestDeadWeight:
    def test_a_written_type_with_no_reader_regresses(self, baseline_dir: Path) -> None:
        dead = {"unread_types": ["Mention"], "passed": False}
        verdict = compare(_run(dead_weight=dead), baseline_dir)
        assert "Mention" in verdict.regressions[0]

    def test_a_run_without_a_dead_weight_result_cannot_merge(self, baseline_dir: Path) -> None:
        verdict = compare(_run(dead_weight={}), baseline_dir)
        assert verdict.regressions == ("dead weight not recorded for this run",)


class TestVerdict:
    def test_every_failure_is_named_no_single_row_is_the_headline(self, baseline_dir: Path) -> None:
        verdict = compare(
            _run(accuracy=0.0, evidence_recall=0.0, precision=0.0, dead_weight={}),
            baseline_dir,
        )
        assert len(verdict.regressions) == 4

    def test_gate_exits_non_zero_naming_the_regressions(self, baseline_dir: Path) -> None:
        with pytest.raises(SystemExit, match="locomo accuracy"):
            compare(_run(accuracy=0.0), baseline_dir).gate()

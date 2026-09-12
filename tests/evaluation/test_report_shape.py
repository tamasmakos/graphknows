"""The panel report shape is a contract (contracts/panel-report.md), not a format."""

from __future__ import annotations

from dataclasses import asdict

import pytest

from evaluation.common.reporting import BaselineMetric, InternalMetrics, PanelRow


class TestPanelRow:
    def test_record_keys_match_contract(self) -> None:
        row = PanelRow(row="locomo", feeding_mode="turn_by_turn", extractor="local")
        assert asdict(row) == {
            "row": "locomo",
            "feeding_mode": "turn_by_turn",
            "extractor": "local",
            "accuracy": 0.0,
            "evidence_recall": 0.0,
            "by_category": {},
            "n": 0,
            "not_run": None,
        }

    def test_unprovisioned_row_carries_a_reason(self) -> None:
        row = PanelRow(
            row="repo",
            feeding_mode="turn_by_turn",
            extractor="llm",
            not_run="corpus not downloaded",
        )
        assert asdict(row)["not_run"] == "corpus not downloaded"


class TestInternalMetrics:
    def test_record_keys_match_contract(self) -> None:
        metrics = InternalMetrics(
            identity={"pairwise_precision": 0.9, "pairwise_recall": 0.8, "n_clusters": 200},
            fact_precision={"precision": 0.7},
            dead_weight={"unread_types": [], "passed": True},
        )
        assert set(asdict(metrics)) == {"identity", "fact_precision", "dead_weight"}
        assert asdict(metrics)["dead_weight"]["passed"] is True


class TestBaselineMetric:
    def test_median_and_band_are_derived_from_runs(self) -> None:
        metric = BaselineMetric(runs=[0.4, 0.6, 0.5], commit="abc123")
        record = metric.to_record()
        assert set(record) == {"median", "band", "runs", "commit"}
        assert record["median"] == 0.5
        assert record["band"] == pytest.approx(0.2)
        assert record["runs"] == [0.4, 0.6, 0.5]
        assert record["commit"] == "abc123"

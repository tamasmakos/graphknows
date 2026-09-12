"""Per-batch resolution time does not follow namespace size (SC-006)."""

from __future__ import annotations

import pytest

from evaluation.scenarios import SCENARIOS, resolve_scaling, seconds_per_batch

pytestmark = pytest.mark.unit


class TestResolveScaling:
    def test_a_flat_curve_passes_and_reports_both_timings(self) -> None:
        result = resolve_scaling(0.010, 0.011)
        assert result.passed
        assert "10.0 ms" in result.detail and "11.0 ms" in result.detail

    def test_a_tenfold_rise_fails(self) -> None:
        assert not resolve_scaling(0.010, 0.100).passed

    def test_twice_the_base_is_not_under_twice_the_base(self) -> None:
        assert not resolve_scaling(0.010, 0.020).passed

    def test_a_measured_batch_stays_flat_across_a_tenfold_namespace(self) -> None:
        assert resolve_scaling(seconds_per_batch(10_000), seconds_per_batch(100_000)).passed


def test_the_scenario_is_offered_by_the_cli() -> None:
    assert "resolve-scaling" in SCENARIOS

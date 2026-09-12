"""The recall verb answers inside the agent hook's budget (SC-010)."""

from __future__ import annotations

import pytest

from evaluation.scenarios import (
    HOOK_BUDGET,
    HOOK_CEILING,
    HOOK_SAMPLES,
    SCENARIOS,
    hook_latency,
)

pytestmark = pytest.mark.unit


def samples(*tail: float, base: float = 0.1) -> list[float]:
    """A full run of *base*-second recalls whose slowest ones are *tail*."""
    return [base] * (HOOK_SAMPLES - len(tail)) + list(tail)


class TestHookLatency:
    def test_a_fast_run_passes_and_reports_both_numbers(self) -> None:
        result = hook_latency(samples())
        assert result.passed
        assert "p95 100 ms" in result.detail and "200 samples" in result.detail

    def test_a_p95_over_the_budget_fails(self) -> None:
        result = hook_latency(samples(*[HOOK_BUDGET] * 20))
        assert not result.passed
        assert "budget" in result.detail

    def test_one_recall_past_the_ceiling_fails_a_run_inside_the_budget(self) -> None:
        result = hook_latency(samples(HOOK_CEILING))
        assert not result.passed
        assert "ceiling" in result.detail

    def test_too_few_samples_fails_instead_of_passing_on_a_thin_tail(self) -> None:
        result = hook_latency(samples()[:10])
        assert not result.passed
        assert "10 samples measured" in result.detail


def test_the_scenario_is_offered_by_the_cli() -> None:
    assert "hook-latency" in SCENARIOS

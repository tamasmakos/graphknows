"""Tests for the supported evaluation benchmark registry."""

from __future__ import annotations

from evaluation.core.benchmark_registry import (
    BENCHMARK_ORDER,
    SUPPORTED_BENCHMARKS,
    expand_benchmark_selection,
)


def test_registry_contains_only_current_measurement_landscape() -> None:
    assert BENCHMARK_ORDER == ("locomo", "longmemeval", "beam")
    assert set(SUPPORTED_BENCHMARKS) == {"locomo", "longmemeval", "beam"}


def test_expand_all_preserves_supported_order() -> None:
    assert expand_benchmark_selection(["all"]) == ["locomo", "longmemeval", "beam"]


def test_expand_deduplicates_explicit_selection() -> None:
    assert expand_benchmark_selection(["beam", "locomo", "beam"]) == ["beam", "locomo"]

"""A drifted transcript record is counted and never fatal (SC-014)."""

from __future__ import annotations

import pytest

from evaluation.scenarios import (
    DRIFT_COUNTERS,
    DRIFTED_TRANSCRIPT,
    SCENARIOS,
    transcript_drift,
)
from graphknows.models.report import Counters

pytestmark = pytest.mark.unit


def counted(**overrides: int) -> Counters:
    """Counters with every drift kind seen once, unless *overrides* says otherwise."""
    return Counters(**({name: 1 for name in DRIFT_COUNTERS} | overrides))


class TestTranscriptDrift:
    def test_counted_drift_beside_kept_segments_passes(self) -> None:
        result = transcript_drift(lambda: ([object()], counted()))  # type: ignore[list-item]
        assert result.passed
        assert "3 drifted parts counted" in result.detail

    def test_uncounted_drift_fails_and_is_named(self) -> None:
        silent = DRIFT_COUNTERS[-1]
        drifted = counted(**{silent: 0})
        result = transcript_drift(lambda: ([object()], drifted))  # type: ignore[list-item]
        assert not result.passed
        assert silent in result.detail

    def test_losing_the_sound_records_fails(self) -> None:
        assert not transcript_drift(lambda: ([], counted())).passed

    def test_an_aborted_parse_is_reported_not_raised(self) -> None:
        def abort() -> tuple[list, Counters]:
            raise ValueError("unknown block type")

        result = transcript_drift(abort)
        assert not result.passed
        assert "unknown block type" in result.detail


class TestTheShippedTranscript:
    async def test_the_real_parser_counts_the_drift_and_keeps_the_rest(self) -> None:
        result = await SCENARIOS["transcript-drift"]("scenario")
        assert result.passed, result.detail

    def test_it_carries_one_drift_of_each_counted_kind(self) -> None:
        assert len(DRIFTED_TRANSCRIPT.splitlines()) == len(DRIFT_COUNTERS) + 1

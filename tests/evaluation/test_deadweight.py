"""A written type with no reader fails the run, but never the baseline (FR-040/FR-041)."""

from __future__ import annotations

import pytest

from evaluation.deadweight import check

WRITTEN = ["Entity", "Episode", "Mention"]


class TestCheck:
    def test_every_written_type_read_back_passes(self) -> None:
        result = check(WRITTEN, ["Mention", "Entity", "Episode"])
        assert result.unread_types == ()
        assert result.passed

    def test_unread_types_are_reported_sorted(self) -> None:
        result = check(WRITTEN, ["Episode"])
        assert result.unread_types == ("Entity", "Mention")
        assert not result.passed

    def test_reading_a_type_that_was_never_written_is_not_dead_weight(self) -> None:
        assert check(WRITTEN, [*WRITTEN, "Segment"]).passed

    def test_record_matches_the_report_contract(self) -> None:
        assert check(WRITTEN, ["Episode"]).to_record() == {
            "unread_types": ["Entity", "Mention"],
            "passed": False,
        }


class TestGate:
    def test_a_dead_plane_exits_non_zero_naming_the_types(self) -> None:
        with pytest.raises(SystemExit, match="Entity, Mention"):
            check(WRITTEN, ["Episode"]).gate()

    def test_a_clean_run_is_a_no_op(self) -> None:
        assert check(WRITTEN, WRITTEN).gate() is None

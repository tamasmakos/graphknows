"""A forgotten record leaves recall while its tombstone stays readable (SC-012)."""

from __future__ import annotations

import pytest

from evaluation.scenarios import SCENARIOS, ForgetObservation, forget_roundtrip

pytestmark = pytest.mark.unit

MERGE_LOG = ("acme1->acme3", "acme2->acme3")


def observed(**overrides: object) -> ForgetObservation:
    """A clean roundtrip: recall lost the record, the store still reads it."""
    fields: dict[str, object] = {
        "record_id": "f1",
        "recalled_after": ("f2",),
        "state_after": "forgotten",
        "merge_log_before": MERGE_LOG,
        "merge_log_after": MERGE_LOG,
    }
    return ForgetObservation(**(fields | overrides))  # type: ignore[arg-type]


class TestForgetRoundtrip:
    def test_a_tombstoned_record_gone_from_recall_passes(self) -> None:
        result = forget_roundtrip(observed())
        assert result.passed
        assert "f1 tombstoned" in result.detail
        assert "2 merge entries" in result.detail

    def test_a_record_still_recalled_fails_and_is_named(self) -> None:
        result = forget_roundtrip(observed(recalled_after=("f1", "f2")))
        assert not result.passed
        assert "f1 is still recalled" in result.detail

    def test_a_deleted_record_fails_rather_than_counting_as_forgotten(self) -> None:
        result = forget_roundtrip(observed(state_after=""))
        assert not result.passed
        assert "gone" in result.detail

    def test_a_record_left_active_fails(self) -> None:
        assert not forget_roundtrip(observed(state_after="active")).passed

    def test_a_merge_log_the_forget_shortened_fails(self) -> None:
        result = forget_roundtrip(observed(merge_log_after=MERGE_LOG[:1]))
        assert not result.passed
        assert "2 entries before" in result.detail and "1 after" in result.detail

    def test_recalling_nothing_to_forget_fails_instead_of_passing_vacuously(self) -> None:
        result = forget_roundtrip(ForgetObservation("", (), "", (), ()))
        assert not result.passed
        assert "nothing was recalled" in result.detail


def test_the_scenario_is_offered_by_the_cli() -> None:
    assert "forget-roundtrip" in SCENARIOS

"""The fact plane: polarity, modality, validity and the tombstone state."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from graphknows.models import Fact, FactState, Mention, Modality, Polarity, Validity


def _fact(**overrides: object) -> Fact:
    fields: dict[str, object] = {
        "id": "f1",
        "subject": "e1",
        "predicate": "p:owns",
        "object": "e2",
    }
    return Fact(**(fields | overrides))  # type: ignore[arg-type]


class TestFactDefaults:
    def test_a_bare_fact_is_asserted_current_and_active(self) -> None:
        fact = _fact()

        assert fact.polarity is Polarity.ASSERTED
        assert fact.is_current
        assert fact.state is FactState.ACTIVE
        assert fact.modality is None

    def test_a_denial_is_stored_as_a_negated_fact(self) -> None:
        assert _fact(polarity=Polarity.NEGATED).polarity is Polarity.NEGATED

    def test_modality_is_pack_vocabulary_not_a_core_constant(self) -> None:
        assert _fact(modality=Modality("hearsay")).modality == "hearsay"

    def test_forgetting_leaves_a_tombstone_rather_than_a_new_kind_of_fact(self) -> None:
        forgotten = _fact().model_copy(update={"state": FactState.FORGOTTEN})

        assert forgotten.state is FactState.FORGOTTEN
        assert forgotten.id == "f1"

    def test_a_fact_is_immutable(self) -> None:
        with pytest.raises(ValidationError):
            _fact().is_current = False  # type: ignore[misc]


class TestValidity:
    def test_defaults_to_an_open_interval(self) -> None:
        assert _fact().validity == Validity()

    def test_accepts_an_ordered_interval(self) -> None:
        window = Validity(
            valid_from=datetime(2024, 1, 1, tzinfo=UTC),
            valid_to=datetime(2025, 1, 1, tzinfo=UTC),
        )

        assert _fact(validity=window).validity.valid_to == window.valid_to

    def test_rejects_an_interval_that_ends_before_it_starts(self) -> None:
        with pytest.raises(ValidationError):
            Validity(
                valid_from=datetime(2025, 1, 1, tzinfo=UTC),
                valid_to=datetime(2024, 1, 1, tzinfo=UTC),
            )


class TestMention:
    def test_keeps_the_surface_attached_to_its_evidence(self) -> None:
        mention = Mention(
            segment_id="s1",
            entity_id="e1",
            surface="the M5 MacBook Air",
            span=(10, 28),
            confidence=0.8,
            extractor="gliner",
        )

        assert (mention.segment_id, mention.span) == ("s1", (10, 28))
        assert mention.surface == "the M5 MacBook Air"

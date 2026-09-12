"""Segment: evidence carries its own span, role and time anchor."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from graphknows.models import Segment, SegmentKind


def _segment(**overrides: object) -> Segment:
    fields: dict[str, object] = {
        "source_id": "abc",
        "text": "The Tensor Brain has two layers.",
        "kind": SegmentKind.prose,
    }
    return Segment(**(fields | overrides))  # type: ignore[arg-type]


class TestSegmentKind:
    def test_kind_vocabulary_is_closed(self) -> None:
        """An unknown kind has no extractor, so it must not construct."""
        with pytest.raises(ValidationError):
            _segment(kind="diagram")

    def test_covers_the_kinds_ingestion_produces(self) -> None:
        assert {k.value for k in SegmentKind} == {
            "prose",
            "turn",
            "code",
            "table",
            "tool_call",
            "tool_result",
            "citation",
        }


class TestObservationTime:
    def test_parser_supplied_anchor_is_kept_and_not_flagged(self) -> None:
        anchor = datetime(2023, 5, 1, tzinfo=UTC)

        segment = _segment(observed_at=anchor)

        assert segment.observed_at == anchor
        assert segment.observed_at_inferred is False

    def test_missing_anchor_falls_back_to_ingestion_time_and_says_so(self) -> None:
        segment = _segment()

        assert segment.observed_at_inferred is True
        assert segment.observed_at is not None
        assert (datetime.now(UTC) - segment.observed_at).total_seconds() < 60


class TestByteRange:
    def test_span_addresses_the_source(self) -> None:
        segment = _segment(byte_range=(12, 44), path="## Methods / ### NER")

        assert segment.byte_range == (12, 44)
        assert segment.path == "## Methods / ### NER"

    @pytest.mark.parametrize("bad", [(-1, 4), (9, 3)])
    def test_a_span_that_cannot_address_bytes_is_rejected(self, bad: tuple[int, int]) -> None:
        with pytest.raises(ValidationError):
            _segment(byte_range=bad)


class TestSegmentFields:
    def test_role_is_a_free_string(self) -> None:
        """Sources name speakers a closed dialogue role vocabulary has no member for."""
        segment = _segment(kind=SegmentKind.turn, role="reviewer-2")

        assert segment.role == "reviewer-2"

    def test_carries_the_extractor_version_that_annotated_it(self) -> None:
        assert _segment(extractor_version="text/2").extractor_version == "text/2"

    def test_is_frozen(self) -> None:
        with pytest.raises(ValidationError):
            _segment().text = "edited"  # type: ignore[misc]

"""Segment: the unit of embedding and extraction, and the only evidence."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from hashlib import sha256
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator


class SegmentKind(StrEnum):
    """What a segment is, as a closed vocabulary.

    Closed because the kind selects the extractor: an unknown kind has no
    reader, so a free string here would fail silently at extraction time.
    """

    prose = "prose"
    turn = "turn"
    code = "code"
    table = "table"
    tool_call = "tool_call"
    tool_result = "tool_result"
    citation = "citation"


class Segment(BaseModel, frozen=True):
    """One span of a source, carrying everything extraction needs to cite it.

    Attributes:
        source_id: The source this span was cut from.
        text: The span's text.
        kind: What the span is; selects the extractor.
        path: Structure path within the source — a heading breadcrumb, a
            qualified symbol name — free-form because every parser shapes it.
        byte_range: Half-open ``(start, end)`` byte offsets into the source.
        role: Speaker role for conversational segments. Free string, never a
            closed vocabulary: a source may name speakers a dialogue model has
            no member for.
        observed_at: The single time anchor of this span, supplied by the
            parser; relative dates in the text resolve against it.
        observed_at_inferred: True when no anchor was supplied and ingestion
            time was used instead.
        extractor_version: Version of the extractor that produced this
            segment's annotations.
        meta: Parser-supplied extras about the span - a transcript record's
            working directory, branch and side-chain flag.
    """

    source_id: str
    text: str
    kind: SegmentKind
    path: str = ""
    byte_range: tuple[int, int] = (0, 0)
    role: str = ""
    observed_at: datetime | None = None
    observed_at_inferred: bool = False
    extractor_version: str = ""
    meta: dict[str, str] = Field(default_factory=dict)

    @property
    def id(self) -> str:
        """Deterministic identity of this span within its source.

        Derived like :attr:`Source.id`, so re-parsing a source rewrites its
        segments in place instead of forking a second set.
        """
        start, end = self.byte_range
        span = f"{self.source_id}|{self.path}|{start}:{end}"
        return sha256(span.encode()).hexdigest()[:16]

    @field_validator("byte_range")
    @classmethod
    def _check_span(cls, byte_range: tuple[int, int]) -> tuple[int, int]:
        """Reject a span that cannot address bytes of a source."""
        start, end = byte_range
        if start < 0 or end < start:
            raise ValueError(f"byte_range must be a non-negative span, got {byte_range}")
        return byte_range

    @model_validator(mode="before")
    @classmethod
    def _anchor_observed_at(cls, data: Any) -> Any:
        """Fall back to ingestion time, saying so, when the parser gave none (FR-013).

        The model is frozen, so the default is filled before construction.
        """
        if isinstance(data, dict) and data.get("observed_at") is None:
            return {**data, "observed_at": datetime.now(UTC), "observed_at_inferred": True}
        return data

"""Fact: the atom of recall, and the mention layer that feeds it."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import NewType

from pydantic import BaseModel, Field, model_validator

Modality = NewType("Modality", str)
"""How a fact is asserted. Pack-supplied vocabulary, never a core constant."""


class Polarity(StrEnum):
    """Whether a fact asserts its predicate or denies it.

    "did NOT observe X" is stored as a negated fact, never as an asserted one.
    """

    ASSERTED = "asserted"
    NEGATED = "negated"


class FactState(StrEnum):
    """Lifecycle of a fact. ``forget`` tombstones, it never deletes."""

    ACTIVE = "active"
    FORGOTTEN = "forgotten"


class Validity(BaseModel, frozen=True):
    """When a fact holds, resolved against its evidence segment's ``observed_at``.

    Both ends are optional: an open interval means "as far as the evidence says".
    """

    valid_from: datetime | None = None
    valid_to: datetime | None = None

    @model_validator(mode="after")
    def _ends_are_ordered(self) -> Validity:
        if self.valid_from and self.valid_to and self.valid_to < self.valid_from:
            raise ValueError("valid_to precedes valid_from")
        return self


class Mention(BaseModel, frozen=True):
    """One surface form of an entity, as observed in one segment.

    The mention layer keeps the surface attached to its evidence: a surface form
    is never collapsed into an alias string on the entity.

    Attributes:
        segment_id: The segment the surface was read from.
        entity_id: The entity the surface refers to.
        surface: The text as written.
        span: ``(start, end)`` byte offsets of the surface within the segment.
        label: The extractor's type for the surface, named in the pack's label
            set; it is what resolves the entity's concept at labelling time.
        confidence: How sure the extractor is of the reference.
        extractor: Identity of the extractor that produced the mention.
    """

    segment_id: str
    entity_id: str
    surface: str
    span: tuple[int, int]
    label: str = ""
    confidence: float = 1.0
    extractor: str = ""


class Fact(BaseModel, frozen=True):
    """A subject-predicate-object assertion with its epistemic status.

    Attributes:
        id: Identity of the assertion.
        subject: Id of the subject entity.
        predicate: Id of an indexed predicate, never a free-string label.
        object: Id of the object entity.
        polarity: Asserted or negated.
        modality: Pack vocabulary term qualifying the assertion, if any.
        confidence: Extractor confidence.
        validity: When the fact holds.
        is_current: Maintained by the superseding rule for functional predicates.
        extractor: Identity of the extractor that produced the fact.
        extractor_version: Version of that extractor.
        state: Active, or tombstoned by a forget.
    """

    id: str
    subject: str
    predicate: str
    object: str
    polarity: Polarity = Polarity.ASSERTED
    modality: Modality | None = None
    confidence: float = 1.0
    validity: Validity = Field(default_factory=Validity)
    is_current: bool = True
    extractor: str = ""
    extractor_version: str = ""
    state: FactState = FactState.ACTIVE

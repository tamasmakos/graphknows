"""Base Pydantic models for the graphknows domain.

All domain models inherit from one of two bases:

- :class:`GraphKnowsBaseModel` — mutable, validates on assignment, consistent
  serialisation settings.  Use this for models whose fields change over time
  (e.g. :class:`~graphknows.models.pipeline.BuildJob`).
- :class:`ImmutableModel` — frozen, hash-able, safe to use as dict keys or in
  sets.  Use this for value objects (e.g. :class:`~graphknows.models.entity.Entity`).

Example:
    from graphknows.models.base import GraphKnowsBaseModel, ImmutableModel

    class MyValueObject(ImmutableModel):
        id: str
        name: str

    class MyMutableRecord(GraphKnowsBaseModel):
        job_id: str
        status: str = "queued"
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class GraphKnowsBaseModel(BaseModel):
    """Root base model with consistent serialisation and validation settings.

    All graphknows domain models inherit from this class.

    Config:
        populate_by_name: Allows population by field name even when an alias is
            set.
        validate_assignment: Re-validates fields when they are reassigned.
        use_enum_values: Serialises enums as their underlying primitive value.
        extra: Forbids unknown extra fields to catch misconfigured callers.
    """

    model_config = ConfigDict(
        populate_by_name=True,
        validate_assignment=True,
        use_enum_values=True,
        extra="forbid",
    )


class ImmutableModel(GraphKnowsBaseModel):
    """Frozen, hash-able base for value objects.

    Prefer this for domain objects that represent immutable facts extracted
    from the knowledge graph (entities, chunks, edges, topics).

    Config:
        frozen: Prevents attribute mutation after construction.
        extra: Inherited — forbids unknown fields.
    """

    model_config = ConfigDict(
        frozen=True,
        populate_by_name=True,
        validate_assignment=False,
        use_enum_values=True,
        extra="forbid",
    )

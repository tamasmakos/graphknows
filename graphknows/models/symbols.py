"""The symbolic index vocabulary: three symbol kinds, and only three."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, field_validator


class SymbolKind(StrEnum):
    """The three index kinds of the Tensor Brain symbolic layer.

    ``EPISODE`` is part of the vocabulary but nothing materialises episode
    vertices in this slice: time is answered from segment observation times and
    fact validity.
    """

    CONCEPT = "concept"
    PREDICATE = "predicate"
    EPISODE = "episode"


def _require_definition(definition: str) -> str:
    """Reject an empty definition: it is what gets embedded."""
    if not definition.strip():
        raise ValueError("definition is required: it is what gets embedded")
    return definition


class ConceptRef(BaseModel, frozen=True):
    """A concept symbol — a class, sense or frame — owned by a pack.

    Frames are concepts with role predicates; there is no frame symbol kind.

    Attributes:
        uri: Globally unique identity of the concept.
        label: Human-readable name.
        definition: What gets embedded; a concept without one has no index entry.
        pack: Name of the owning pack.
    """

    uri: str
    label: str
    definition: str
    pack: str

    _definition_is_present = field_validator("definition")(_require_definition)


class PredicateRef(BaseModel, frozen=True):
    """A predicate symbol — a named relation, never a free string.

    Attributes:
        id: Pack-scoped, globally unique identity.
        label: Human-readable name.
        definition: What gets embedded; required, same rule as ``ConceptRef``.
        canonical: Surface form used in rendering.
        functional: At most one current object per subject; drives superseding.
        domain: Concept uri the subject must be an instance of, if constrained.
        range: Concept uri the object must be an instance of, if constrained.
        pack: Name of the owning pack.
    """

    id: str
    label: str
    definition: str
    canonical: str
    functional: bool = False
    domain: str | None = None
    range: str | None = None
    pack: str

    _definition_is_present = field_validator("definition")(_require_definition)

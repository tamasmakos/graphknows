"""The per-chunk decode request and the strict response schema.

The item models are what reaches the provider as a strict ``json_schema``, and
they are also what a returned item is held to on the way back — but *item by
item*. Typing the container's sections as ``list[DecodedEntity]`` would let one
malformed ``confidence`` reject the whole response and cost the chunk a
``decoder_failed`` abstention; here it costs that item a ``malformed_item``
(contracts/decoder.md §3, data-model.md §4). Only a payload that is not a
container at all is a decoder failure.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, TypeVar

from pydantic import BaseModel, ConfigDict, Field, ValidationError

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FrameCandidate:
    """A locally-selected frame the LLM may fill roles for. It never proposes its own."""

    frame: str
    trigger: str
    trigger_offset: int  # -1 when the candidate came from the LU path
    core_elements: tuple[tuple[str, str], ...]  # (FE name, FE definition)


@dataclass(frozen=True)
class DecodeRequest:
    """One chunk plus the closed vocabularies it is decoded against. Never persisted."""

    text: str
    entity_labels: tuple[str, ...]
    relation_spec: dict[str, str]  # label -> definition; {} = no relation section
    frame_candidates: tuple[FrameCandidate, ...]  # () = no frame section
    speaker: str = ""  # "" when unresolved
    prompt_addendum: str = ""  # the pack's guidance, appended to the instructions


class DecodedEntity(BaseModel):
    """One returned entity. ``surface`` is verbatim; offsets are recovered locally."""

    model_config = ConfigDict(extra="forbid")

    surface: str
    label: str


class DecodedRelation(BaseModel):
    """One returned relation, carrying the confidence the write path gates on."""

    model_config = ConfigDict(extra="forbid")

    head: str
    predicate: str
    tail: str
    evidence: str
    confidence: float = Field(ge=0.0, le=1.0)


class DecodedFrameInstance(BaseModel):
    """One filled frame: a candidate's name, its trigger, and verbatim role fillers."""

    model_config = ConfigDict(extra="forbid")

    frame: str
    trigger: str
    roles: dict[str, list[str]]


@dataclass(frozen=True)
class DecodedItems:
    """A response's three sections after per-item validation, with the drops counted."""

    entities: tuple[DecodedEntity, ...] = ()
    relations: tuple[DecodedRelation, ...] = ()
    frames: tuple[DecodedFrameInstance, ...] = ()
    malformed_item: int = 0  # named for the abstention gate it feeds


_Item = TypeVar("_Item", bound=BaseModel)


def _validate_each(raw: list[dict[str, Any]], model: type[_Item]) -> tuple[list[_Item], int]:
    """Return the items that pass ``model`` and the count of those that did not."""
    kept: list[_Item] = []
    malformed = 0
    for entry in raw:
        try:
            kept.append(model.model_validate(entry))
        except ValidationError as exc:
            malformed += 1
            logger.debug("malformed_item: %s rejected %s (%s)", model.__name__, entry, exc)
    return kept, malformed


class DecodeResponse(BaseModel):
    """The container. Sections stay raw here so one bad item cannot fail the chunk."""

    entities: list[dict[str, Any]] = Field(default_factory=list)
    relations: list[dict[str, Any]] = Field(default_factory=list)
    frames: list[dict[str, Any]] = Field(default_factory=list)

    def decoded(self) -> DecodedItems:
        """Validate every item against its strict model, dropping and counting failures."""
        entities, entity_drops = _validate_each(self.entities, DecodedEntity)
        relations, relation_drops = _validate_each(self.relations, DecodedRelation)
        frames, frame_drops = _validate_each(self.frames, DecodedFrameInstance)
        return DecodedItems(
            entities=tuple(entities),
            relations=tuple(relations),
            frames=tuple(frames),
            malformed_item=entity_drops + relation_drops + frame_drops,
        )

"""Pydantic models for KG entities shared across graphrag and graphgen services."""
from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field

from graphknows.shared.constants import EntityLabel


class Entity(BaseModel):
    """A named entity node in the knowledge graph.

    Attributes:
        id: Unique entity identifier.
        name: Human-readable entity name.
        label: LKIF entity type.
        community_id: Leiden community the entity belongs to.
        properties: Arbitrary additional properties from extraction.
    """

    id: str
    name: str
    label: EntityLabel
    community_id: Optional[int] = None
    properties: dict[str, Any] = Field(default_factory=dict)


class Chunk(BaseModel):
    """A text chunk from an ingested document.

    Attributes:
        id: Unique chunk identifier.
        doc_id: Parent document identifier.
        text: Raw chunk text.
        entity_count: Number of entities extracted from this chunk.
    """

    id: str
    doc_id: str
    text: str
    entity_count: int = 0


class Edge(BaseModel):
    """A directed relation between two entities.

    Attributes:
        source: Source entity id.
        target: Target entity id.
        relation_type: LKIF relation label.
        weight: Edge weight for RRF / graph traversal.
        properties: Arbitrary additional properties.
    """

    source: str
    target: str
    relation_type: str
    weight: float = 1.0
    properties: dict[str, Any] = Field(default_factory=dict)


class Topic(BaseModel):
    """A Leiden community / topic node.

    Attributes:
        id: Community identifier.
        label: Human-readable topic label from summarisation.
        entity_ids: Member entity ids.
    """

    id: str
    label: str
    entity_ids: list[str] = Field(default_factory=list)

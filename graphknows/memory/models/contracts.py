"""Canonical memory contracts shared by STM, promotion, and retrieval."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from graphknows.memory.adapter_types import MemoryClass


class ExtractedEntity(BaseModel):
    """Entity extracted from one memory record."""

    name: str
    type: str = "ENTITY"
    confidence: float = 1.0
    aliases: list[str] = Field(default_factory=list)


class ExtractedRelation(BaseModel):
    """Relation extracted from one memory record."""

    source: str
    relation: str
    target: str
    confidence: float = 0.5
    metadata: dict[str, Any] = Field(default_factory=dict)


class ExtractionResult(BaseModel):
    """Normalized extraction result used across extractor implementations."""

    entities: list[ExtractedEntity] = Field(default_factory=list)
    relations: list[ExtractedRelation] = Field(default_factory=list)


class MemoryRecord(BaseModel):
    """One normalized text unit stored in memory."""

    id: str
    session_id: str
    text: str
    kind: Literal["chunk", "turn"] = "chunk"
    memory_class: MemoryClass = MemoryClass.raw_evidence
    file_id: str = ""
    dialog_id: str = ""
    role: str = ""
    heading_path: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)
    entities: list[ExtractedEntity] = Field(default_factory=list)
    relations: list[ExtractedRelation] = Field(default_factory=list)
    community_id: int | None = None
    community_weight: float | None = None


class CommunityAssignment(BaseModel):
    """Community metadata assigned during LTM enrichment."""

    community_id: int
    entity_names: list[str] = Field(default_factory=list)
    chunk_ids: list[str] = Field(default_factory=list)
    turn_ids: list[str] = Field(default_factory=list)
    weight: float = 1.0
    algorithm: str = ""


class SessionSnapshot(BaseModel):
    """Full STM snapshot consumed by Promotion."""

    session_id: str
    entities: list[dict[str, Any]] = Field(default_factory=list)
    files: list[dict[str, Any]] = Field(default_factory=list)
    chunks: list[dict[str, Any]] = Field(default_factory=list)
    dialogs: list[dict[str, Any]] = Field(default_factory=list)
    turns: list[dict[str, Any]] = Field(default_factory=list)
    chunk_entity_edges: list[dict[str, Any]] = Field(default_factory=list)
    turn_entity_edges: list[dict[str, Any]] = Field(default_factory=list)
    next_edges: list[dict[str, Any]] = Field(default_factory=list)
    relation_edges: list[dict[str, Any]] = Field(default_factory=list)

    @classmethod
    def from_store(cls, session_id: str, data: dict[str, Any]) -> SessionSnapshot:
        """Build a snapshot from a session graph store response."""
        fields = {key: value for key, value in data.items() if key in cls.model_fields}
        return cls(session_id=session_id, **fields)

    def to_store_dict(self) -> dict[str, Any]:
        """Return the legacy dict shape consumed by current stores."""
        data = self.model_dump()
        data.pop("session_id", None)
        return data


class PromotionAudit(BaseModel):
    """Auditable promotion decisions."""

    entity_aliases: dict[str, list[str]] = Field(default_factory=dict)
    merged_entities: int = 0
    communities: list[CommunityAssignment] = Field(default_factory=list)


class RetrievalHit(BaseModel):
    """Common hit model emitted by deterministic retrieval collectors."""

    chunk_id: str
    text: str = ""
    source: Literal["stm_graph", "ltm_graph", "semantic"] = "semantic"
    rank: int = 0
    doc_id: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)
    entities: list[Any] = Field(default_factory=list)
    boost: float = 1.0

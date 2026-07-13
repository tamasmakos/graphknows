"""Canonical memory contracts shared by STM, promotion, and retrieval."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


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

"""Evidence item DTO."""

from __future__ import annotations

from pydantic import BaseModel, Field

from graphknows.memory.adapter_types import MemoryClass


class EvidenceItem(BaseModel):
    """A retrievable evidence artifact."""

    artifact_id: str
    path: str = ""
    content: str = ""
    mime_type: str = "text/plain"
    memory_class: MemoryClass
    session_id: str | None = None
    metadata: dict = Field(default_factory=dict)

"""Ingest request/result DTOs."""
from __future__ import annotations

from pydantic import BaseModel, Field


class IngestRequest(BaseModel, frozen=True):
    """Parameters for a memory ingest operation."""

    text: str
    session_id: str
    title: str = ""
    metadata: dict = Field(default_factory=dict)


class IngestResult(BaseModel):
    """Result of a memory ingest operation."""

    session_id: str
    file_id: str
    title: str
    chunk_count: int = 0
    entity_count: int = 0
    elapsed_s: float = 0.0
    errors: list[str] = Field(default_factory=list)

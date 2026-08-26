"""Ingest result DTO."""

from __future__ import annotations

from pydantic import BaseModel, Field


class IngestResult(BaseModel):
    """Result of a memory ingest operation."""

    session_id: str
    file_id: str
    title: str
    chunk_count: int = 0
    entity_count: int = 0
    elapsed_s: float = 0.0
    errors: list[str] = Field(default_factory=list)
    # What the gates declined to write, per gate. Empty means "no gate fired",
    # which is a different claim from "no gate ran" — a gate that abstains
    # silently is indistinguishable from an extractor that found nothing.
    abstentions: dict[str, int] = Field(default_factory=dict)

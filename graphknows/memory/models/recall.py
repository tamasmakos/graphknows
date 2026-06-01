"""Recall query/result DTOs."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class RecallResult(BaseModel):
    """A single memory recall hit."""

    text: str
    source_type: Literal["chunk", "turn"]
    scope: Literal["short_term", "long_term"]
    score: float = 0.0
    cosine_score: float = 0.0
    entity_names: list[str] = Field(default_factory=list)
    provenance: dict = Field(default_factory=dict)
    heading_path: str = ""
    metadata: dict = Field(default_factory=dict)


class RecallQuery(BaseModel):
    """Parameters for a memory recall query."""

    query: str
    session_id: str
    top_k: int = 5
    scope: Literal["stm", "ltm", "both"] = "stm"

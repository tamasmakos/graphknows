"""Promotion (STM→LTM flush) request/result DTOs."""

from __future__ import annotations

from pydantic import BaseModel, Field

from graphknows.memory.models.gds import GDSResult


class PromotionRequest(BaseModel, frozen=True):
    """Parameters for a STM→LTM flush operation."""

    session_id: str
    run_gds: bool = True
    run_topics: bool = True
    clear_stm: bool = True
    merge_entities: bool = True


class PromotionRecord(BaseModel):
    """Result of a STM→LTM flush operation."""

    session_id: str
    nodes_written: int = 0
    edges_written: int = 0
    chunks_promoted: int = 0
    turns_promoted: int = 0
    topics_created: int = 0
    merged_entities: int = 0
    elapsed_s: float = 0.0
    errors: list[str] = Field(default_factory=list)
    gds_result: GDSResult | None = None

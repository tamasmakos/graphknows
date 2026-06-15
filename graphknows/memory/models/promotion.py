"""Promotion (STM->LTM flush) request/result DTOs."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from graphknows.memory.models.contracts import PromotionAudit
from graphknows.memory.models.gds import GDSResult

TopicMode = Literal["off", "llm_assisted"]


class PromotionPolicy(BaseModel, frozen=True):
    """Resolved Promotion policy used by the STM->LTM pipeline."""

    run_gds: bool = True
    clear_stm: bool = True
    merge_entities: bool = True
    topic_mode: TopicMode = "off"

    @property
    def run_topics(self) -> bool:
        """Return whether Topic Enrichment should run."""
        return self.topic_mode == "llm_assisted"


class PromotionRequest(BaseModel, frozen=True):
    """Parameters for a STM->LTM flush operation."""

    session_id: str
    run_gds: bool = True
    clear_stm: bool = True
    merge_entities: bool = True
    topic_mode: TopicMode | None = None

    def resolve_policy(self, relation_extraction_mode: str = "gliner2") -> PromotionPolicy:
        """Resolve request into the canonical PromotionPolicy."""
        topic_mode = self.topic_mode
        if topic_mode is None:
            topic_mode = "llm_assisted" if relation_extraction_mode == "llm_assisted" else "off"
        return PromotionPolicy(
            run_gds=self.run_gds,
            clear_stm=self.clear_stm,
            merge_entities=self.merge_entities,
            topic_mode=topic_mode,
        )


class PromotionRecord(BaseModel):
    """Result of a STM->LTM flush operation."""

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
    topic_mode: TopicMode = "off"
    audit: PromotionAudit = Field(default_factory=PromotionAudit)

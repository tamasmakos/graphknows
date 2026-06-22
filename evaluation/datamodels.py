"""Pydantic datamodels for the LoCoMo evaluation framework.

Every value that crosses a module boundary (loading → ingestion → retrieval →
generation → scoring → reporting) is one of these typed models. No bare dicts
in signatures.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from evaluation.config import EvalConfig


class EvalDocument(BaseModel):
    """One conversation session ready to ingest into memory."""

    title: str
    text: str
    anchor_date: str = ""  # ISO datetime of the session, for relative-date resolution


class LocomoCase(BaseModel):
    """One LoCoMo question within a dialogue."""

    case_id: str
    dialogue_id: str
    question: str
    gold: str
    category: str = ""


class Dialogue(BaseModel):
    """A LoCoMo dialogue: its sessions (ingested once) and all its questions."""

    dialogue_id: str
    documents: list[EvalDocument] = Field(default_factory=list)
    cases: list[LocomoCase] = Field(default_factory=list)


class RetrievedPassage(BaseModel):
    """One hit returned by ``memory_query``."""

    text: str
    score: float | None = None
    sources: str = ""

    @property
    def is_ltm_graph(self) -> bool:
        """Whether this hit came from the LTM relation graph collector."""
        return "ltm_graph" in self.sources


class IngestResult(BaseModel):
    """Aggregated ingest (+ optional flush) outcome for one dialogue."""

    documents: int = 0
    chunks: int = 0
    entities: int = 0
    ingest_elapsed_s: float = 0.0
    ingest_error_count: int = 0
    # flush-to-LTM fields (zero when promote_to_ltm is False)
    flush_nodes: int = 0
    flush_edges: int = 0
    flush_chunks: int = 0
    flush_topics: int = 0
    flush_elapsed_s: float = 0.0
    flush_error_count: int = 0
    ltm_entity_count: int = 0


class CaseResult(BaseModel):
    """Scored outcome for one LoCoMo question."""

    case_id: str
    question: str
    gold: str
    category: str = ""
    dialogue_id: str = ""
    passages: list[RetrievedPassage] = Field(default_factory=list)
    generated_answer: str = ""
    score: float = 0.0
    metrics: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def to_record(self) -> dict[str, Any]:
        """JSON-friendly dict for report writing."""
        record = self.model_dump()
        record["memory_context"] = [p.model_dump() for p in self.passages]
        return record


class RunReport(BaseModel):
    """The full result of one evaluation run."""

    run_id: str
    config: EvalConfig
    cases: list[CaseResult] = Field(default_factory=list)
    summary: dict[str, Any] = Field(default_factory=dict)

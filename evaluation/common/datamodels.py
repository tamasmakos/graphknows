"""Pydantic datamodels shared by every benchmark.

Every value that crosses a module boundary (loading → ingestion → retrieval →
generation → judging → reporting) is one of these typed models. No bare dicts
in signatures.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class EvalDocument(BaseModel):
    """One unit of text ready to ingest into memory (usually one session)."""

    title: str
    text: str
    anchor_date: str = ""  # ISO datetime of the session, for relative-date resolution


class EvalCase(BaseModel):
    """One benchmark question."""

    case_id: str
    group_id: str  # the ingest unit this question runs against
    question: str
    gold: str
    category: str = ""  # question category / type (benchmark-specific vocabulary)
    question_date: str = ""  # "today" for the question, when the benchmark defines one
    extras: dict[str, Any] = Field(default_factory=dict)  # benchmark-specific payload


class IngestUnit(BaseModel):
    """A group of documents ingested once, plus all questions asked against it."""

    group_id: str
    documents: list[EvalDocument] = Field(default_factory=list)
    cases: list[EvalCase] = Field(default_factory=list)
    reference_date: str = ""  # date of the newest document (anchors relative time)


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
    """Aggregated ingest (+ optional flush) outcome for one ingest unit."""

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
    # LLM usage during ingestion (llm_assisted mode only)
    ingest_prompt_tokens: int = 0
    ingest_completion_tokens: int = 0
    ingest_cost_usd: float = 0.0


class CaseResult(BaseModel):
    """Scored outcome for one question."""

    case_id: str
    question: str
    gold: str
    category: str = ""
    group_id: str = ""
    passages: list[RetrievedPassage] = Field(default_factory=list)
    generated_answer: str = ""
    score: float = 0.0  # headline: LLM-judge score in [0, 1]
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
    benchmark: str
    config: dict[str, Any] = Field(default_factory=dict)
    cases: list[CaseResult] = Field(default_factory=list)
    summary: dict[str, Any] = Field(default_factory=dict)

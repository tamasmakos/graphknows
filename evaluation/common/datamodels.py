"""Pydantic datamodels shared by every benchmark.

Every value that crosses a module boundary (loading → ingestion → retrieval →
generation → judging → reporting) is one of these typed models. No bare dicts
in signatures.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class EvalDocument(BaseModel):
    """One unit ready to ingest into memory (usually one session).

    Two ingest shapes, and which one is populated decides the harness:

    * ``text`` — the session concatenated into one blob, ingested eagerly. Its
      chunks are multi-turn transcripts, which is why a document-fed run pins
      ``evidence_recall@probe`` at 1.000 and cannot measure retrieval.
    * ``turns`` — the utterances kept apart, ingested as TURN rows and drained
      by flush, exactly as a live agent does it. Chunking is then the pipeline's
      own 5-turn sliding window, and the benchmark regains headroom.
    """

    title: str
    text: str
    anchor_date: str = ""  # ISO datetime of the session, for relative-date resolution
    # Message dicts (``role``/``content``/``name``/``timestamp``) when the unit
    # is fed turn-by-turn. Empty for the document harness.
    turns: list[dict[str, str]] = []


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
    # The chunk's own timestamp, as the retriever reports it. Without this the
    # answerer can only learn WHEN something happened if the date happens to be
    # written inside the passage text, which is true of a concatenated
    # transcript and false of everything else.
    ts: str = ""

    # Channels that reach a chunk through graph structure (entity/relation
    # traversal, PPR, ontology classes, topics, frames) rather than dense-vector
    # or lexical match. "graph contributed this hit" = any of these in sources.
    _GRAPH_SOURCES = frozenset({"entity", "ppr", "ontology", "topic", "frame"})

    @property
    def is_graph_hit(self) -> bool:
        """Whether a graph-structure channel contributed this hit."""
        return bool(self._GRAPH_SOURCES & set(self.sources.split(",")))


class IngestResult(BaseModel):
    """Aggregated ingest (+ optional flush) outcome for one ingest unit."""

    documents: int = 0
    chunks: int = 0
    entities: int = 0
    # Turns buffered by a turn-fed unit. They mint no chunks or entities until
    # flush drains them, so a turn-fed run legitimately reports chunks=0 here
    # and its real graph size in the flush_* fields below.
    turns: int = 0
    ingest_elapsed_s: float = 0.0
    ingest_error_count: int = 0
    # flush-to-LTM fields (zero when promote_to_ltm is False)
    flush_nodes: int = 0
    flush_edges: int = 0
    flush_chunks: int = 0
    flush_elapsed_s: float = 0.0
    flush_error_count: int = 0
    # Seconds per named flush stage (drain, resolve, pagerank, node2vec, louvain,
    # topics...). `flush_elapsed_s` is their sum, and on its own it cannot say
    # WHICH stage took the time — which is how "the ingest is slow" stayed an
    # unanswered question through several runs.
    flush_stage_s: dict[str, float] = Field(default_factory=dict)
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

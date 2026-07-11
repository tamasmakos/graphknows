"""Shared infrastructure for all evaluation benchmarks.

Everything benchmark-agnostic lives here: typed datamodels, the OpenRouter
chat client, memory formatting, ingestion/retrieval against the graphknows
MCP server, judge caching, lexical diagnostics, reporting, and the shared
benchmark pipeline.
"""

from evaluation.common.config import BaseEvalConfig
from evaluation.common.datamodels import (
    CaseResult,
    EvalCase,
    EvalDocument,
    IngestResult,
    IngestUnit,
    RetrievedPassage,
    RunReport,
)
from evaluation.common.llm import ChatLLM, CreditExhaustedError, extract_answer
from evaluation.common.memories import format_memories

__all__ = [
    "BaseEvalConfig",
    "CaseResult",
    "ChatLLM",
    "CreditExhaustedError",
    "EvalCase",
    "EvalDocument",
    "IngestResult",
    "IngestUnit",
    "RetrievedPassage",
    "RunReport",
    "extract_answer",
    "format_memories",
]

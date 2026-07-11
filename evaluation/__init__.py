"""Memory-benchmark evaluation frameworks for graphknows.

Three benchmarks, one shared pipeline (ingest → retrieve → generate → LLM-judge):

* ``evaluation.locomo``      — long-term conversational memory QA
* ``evaluation.longmemeval`` — long-term interactive assistant memory
* ``evaluation.beam``        — very-long-conversation memory abilities

Run with ``python -m evaluation <benchmark> [flags]``; bare flags default to
locomo for backward compatibility. Shared infrastructure (datamodels, LLM
client, ingestion/retrieval, judging, reporting) lives in ``evaluation.common``.
"""

from evaluation.common.config import BaseEvalConfig
from evaluation.common.datamodels import (
    CaseResult,
    EvalCase,
    EvalDocument,
    IngestUnit,
    RetrievedPassage,
    RunReport,
)

__all__ = [
    "BaseEvalConfig",
    "CaseResult",
    "EvalCase",
    "EvalDocument",
    "IngestUnit",
    "RetrievedPassage",
    "RunReport",
]

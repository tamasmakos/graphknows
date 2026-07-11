"""Memory-benchmark evaluation framework for graphknows.

One shared pipeline (ingest → retrieve → generate → LLM-judge):

* ``evaluation.locomo`` — long-term conversational memory QA

Run with ``python -m evaluation [flags]``. Shared infrastructure (datamodels,
LLM client, ingestion/retrieval, judging, reporting) lives in
``evaluation.common``.
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

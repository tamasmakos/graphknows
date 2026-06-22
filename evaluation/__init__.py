"""LoCoMo evaluation framework for graphknows.

One workflow: ingest → retrieve → DSPy generate → token-F1. Run with
``python -m evaluation --limit N``.
"""

from __future__ import annotations

from evaluation.config import EvalConfig
from evaluation.datamodels import CaseResult, LocomoCase, RetrievedPassage, RunReport
from evaluation.pipeline import run_evaluation

__all__ = [
    "CaseResult",
    "EvalConfig",
    "LocomoCase",
    "RetrievedPassage",
    "RunReport",
    "run_evaluation",
]

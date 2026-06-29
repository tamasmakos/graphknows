"""LoCoMo evaluation framework for graphknows.

One workflow: ingest → retrieve → DSPy generate → SemanticF1 (headline) + token-F1 (diagnostic).
Run with ``python -m evaluation --limit N``.
"""

from __future__ import annotations

from evaluation.config import EvalConfig
from evaluation.datamodels import CaseResult, LocomoCase, RetrievedPassage, RunReport

__all__ = [
    "CaseResult",
    "EvalConfig",
    "LocomoCase",
    "RetrievedPassage",
    "RunReport",
    "run_evaluation",
]


def __getattr__(name: str):  # noqa: ANN001
    """Lazy-import ``run_evaluation`` so that ``import evaluation`` does not eagerly
    pull in dspy/generation at collection time (dspy may be absent in the test env)."""
    if name == "run_evaluation":
        from evaluation.pipeline import run_evaluation  # noqa: PLC0415

        return run_evaluation
    raise AttributeError(f"module 'evaluation' has no attribute {name!r}")

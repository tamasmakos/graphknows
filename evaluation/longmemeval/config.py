"""LongMemEval workflow configuration."""

from __future__ import annotations

from pathlib import Path

from evaluation.common.config import BaseEvalConfig

DATASET_URL = (
    "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/"
    "longmemeval_s_cleaned.json"
)


class LongMemEvalConfig(BaseEvalConfig):
    """The LongMemEval evaluation workflow.

    Each question is its own ingest unit (the benchmark ships one haystack per
    question), so ingestion cost scales linearly with the number of questions —
    sample with ``--per-type`` before running wide.
    """

    # ── Dataset ──────────────────────────────────────────────────────────────
    data_root: Path = Path("evaluation/data/longmemeval")
    dataset_file: str = "longmemeval_s_cleaned.json"
    dataset_url: str = DATASET_URL
    max_chars_per_document: int = 12_000  # haystack sessions are short; generous cap

    # ── Lifecycle / output ───────────────────────────────────────────────────
    session_prefix: str = "eval-lme"
    output_dir: Path = Path("evaluation/results/longmemeval")

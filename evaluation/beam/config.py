"""BEAM workflow configuration."""

from __future__ import annotations

from pathlib import Path

from evaluation.common.config import BaseEvalConfig

# HuggingFace dataset ids; the 10M bucket ships as its own dataset.
HF_DATASET = "Mohammadta/BEAM"
HF_DATASET_10M = "Mohammadta/BEAM-10M"
CHAT_SIZES = ("100K", "500K", "1M", "10M")


class BeamConfig(BaseEvalConfig):
    """The BEAM evaluation workflow.

    A single conversation is 100K-10M tokens, so ingestion dominates run time —
    select conversations explicitly (``--conversations 0-4``) before running wide.
    """

    # ── Dataset ──────────────────────────────────────────────────────────────
    data_root: Path = Path("evaluation/data/beam")
    max_chars_per_document: int = 8000  # chat batches are split, never truncated

    # ── Lifecycle / output ───────────────────────────────────────────────────
    namespace: str = "eval_beam"
    session_prefix: str = "eval-beam"
    output_dir: Path = Path("evaluation/results/beam")

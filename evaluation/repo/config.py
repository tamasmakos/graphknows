"""Repository-transcript row configuration.

The corpus is this repository's own agent transcripts, so there is no download
step and no per-question haystack: one transcript is one document, and the gold
set lives beside it under ``data_root``.
"""

from __future__ import annotations

from pathlib import Path

from evaluation.common.config import BaseEvalConfig


class RepoConfig(BaseEvalConfig):
    """The repository-transcript evaluation workflow."""

    # ── Dataset ──────────────────────────────────────────────────────────────
    data_root: Path = Path("evaluation/data/repo")
    questions_file: str = "gold.jsonl"
    # A session transcript is long; keep enough of it to hold the decision.
    max_chars_per_document: int = 40000

    # ── Lifecycle / output ───────────────────────────────────────────────────
    namespace: str = "eval_repo"
    session_prefix: str = "eval-repo"
    output_dir: Path = Path("evaluation/results/repo")

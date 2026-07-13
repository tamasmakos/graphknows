"""LongMemEval workflow configuration.

Each question ships its own haystack of ~53 sessions, so ``max_documents`` is
raised well above the LoCoMo default. Retrieval/concurrency knobs are inherited
from :class:`~evaluation.common.config.BaseEvalConfig`.
"""

from __future__ import annotations

from pathlib import Path

from evaluation.common.config import BaseEvalConfig


class LongMemConfig(BaseEvalConfig):
    """The LongMemEval evaluation workflow."""

    # ── Dataset ──────────────────────────────────────────────────────────────
    data_root: Path = Path("evaluation/data/longmemeval")
    questions_file: str = "longmemeval_s_cleaned.json"
    # Each question's haystack has 53 sessions — ingest all of them.
    max_documents: int = 60
    max_chars_per_document: int = 8000

    # ── Lifecycle / output ───────────────────────────────────────────────────
    namespace: str = "eval_longmem"
    session_prefix: str = "eval-longmem"
    output_dir: Path = Path("evaluation/results/longmem")

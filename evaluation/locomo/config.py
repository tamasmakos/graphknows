"""LoCoMo workflow configuration.

Retrieval knobs carry over from the tuned baseline (see the inline history):
top_k=5 for the scored diagnostic, gen_context_k=25 as the efficiency knee
(a k∈{50,25,15,10,5} sweep showed 25 and 50 equal on quality while 25 cuts
prompt tokens ~18%; below 25 accuracy declines), probe_k=50 as the free
reachability diagnostic.
"""

from __future__ import annotations

from pathlib import Path

from evaluation.common.config import BaseEvalConfig


class LocomoConfig(BaseEvalConfig):
    """The LoCoMo evaluation workflow."""

    # ── Dataset ──────────────────────────────────────────────────────────────
    data_root: Path = Path("evaluation/data/locomo")
    questions_file: str = "questions.jsonl"
    max_documents: int = 50
    # 8000 covers the longest LoCoMo session (7187 chars). At 4000, 24.6% of
    # sessions were truncated — silently dropping 7.1% of all conversation text
    # (and any gold evidence in the tail).
    max_chars_per_document: int = 8000

    # ── Lifecycle / output ───────────────────────────────────────────────────
    namespace: str = "eval_locomo"
    session_prefix: str = "eval-locomo"
    output_dir: Path = Path("evaluation/results/locomo")

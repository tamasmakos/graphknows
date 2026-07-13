"""BEAM workflow configuration.

BEAM conversations are large (the 100K-token tier here), so per-document text is
left effectively unclipped — the memory system's own chunking handles the volume,
which is exactly what the benchmark stresses. Retrieval/concurrency knobs are
inherited from :class:`~evaluation.common.config.BaseEvalConfig`.
"""

from __future__ import annotations

from pathlib import Path

from evaluation.common.config import BaseEvalConfig


class BeamConfig(BaseEvalConfig):
    """The BEAM evaluation workflow."""

    # ── Dataset ──────────────────────────────────────────────────────────────
    data_root: Path = Path("evaluation/data/beam")
    questions_file: str = "beam_100K.json"
    # A conversation is a handful of long sessions plus a profile doc.
    max_documents: int = 12
    # Effectively no clipping — BEAM's point is large-context ingestion; the
    # ingest pipeline chunks each session itself.
    max_chars_per_document: int = 200_000

    # ── Lifecycle / output ───────────────────────────────────────────────────
    namespace: str = "eval_beam"
    session_prefix: str = "eval-beam"
    output_dir: Path = Path("evaluation/results/beam")

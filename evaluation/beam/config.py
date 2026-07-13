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
    # BEAM sessions are 150K-185K chars each. The loader splits every session on
    # message boundaries into bounded documents (a single oversized ingest crashes
    # the extraction model), so one conversation yields ~30-40 documents — enough
    # headroom to ingest the full 100K-token history without dropping content.
    max_documents: int = 80
    # Per-document size bound the loader splits sessions to (not a clip).
    max_chars_per_document: int = 16_000

    # ── Lifecycle / output ───────────────────────────────────────────────────
    namespace: str = "eval_beam"
    session_prefix: str = "eval-beam"
    output_dir: Path = Path("evaluation/results/beam")

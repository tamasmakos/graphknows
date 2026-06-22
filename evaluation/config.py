"""The single hardcoded LoCoMo evaluation workflow.

There is exactly one supported configuration — the generation-mode,
paper-comparable token-F1 pipeline (gliner2 extraction, promote-to-LTM,
DSPy answer generation). The CLI only overrides ``limit``/``offset``;
everything else lives here. Edit this file to change the workflow.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict

# Best-known LoCoMo config — the `genfull-conv26` baseline (mean token-F1 0.314,
# see project_generation_baseline memory). top_k=5 is deliberate: more chunks add
# distractors and hurt list aggregation without lifting the Unknown rate.


class EvalConfig(BaseModel):
    """Frozen description of the one true LoCoMo evaluation workflow."""

    model_config = ConfigDict(frozen=True)

    # ── MCP server ───────────────────────────────────────────────────────────
    command: str = "graphknows-mcp"
    request_timeout_s: float = 900.0

    # ── Dataset ──────────────────────────────────────────────────────────────
    data_root: Path = Path("evaluation/data/locomo")
    questions_file: str = "questions.jsonl"
    max_documents: int = 50
    max_chars_per_document: int = 4000

    # ── Retrieval ────────────────────────────────────────────────────────────
    top_k: int = 5  # scored retrieval depth
    gen_context_k: int = 8  # candidate pool handed to the generator
    probe_k: int = 20  # gold-reachability probe depth
    recall_scope: str = "both"  # stm | ltm | both
    relation_extraction_mode: str = "gliner2"
    promote_to_ltm: bool = True

    # ── Generation ───────────────────────────────────────────────────────────
    generation_model: str = ""  # → OPENROUTER_MODEL env or moonshotai/kimi-k2

    # ── Lifecycle ────────────────────────────────────────────────────────────
    session_prefix: str = "eval-locomo"
    purge: bool = True  # purge each dialogue session after querying
    pre_run_purge_ltm: bool = True  # wipe all LTM before the run (no cross-run contamination)

    # ── Output ───────────────────────────────────────────────────────────────
    output_dir: Path = Path("evaluation/results/locomo")

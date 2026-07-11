"""Base configuration shared by every benchmark.

Each benchmark subclasses :class:`BaseEvalConfig` with its own dataset paths,
session prefix, and output directory. The CLI overrides the run window and
mode; everything else lives in the config classes.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict


class BaseEvalConfig(BaseModel):
    """Benchmark-agnostic knobs for one evaluation workflow."""

    model_config = ConfigDict(frozen=True)

    # ── MCP server ───────────────────────────────────────────────────────────
    command: str = "graphknows-mcp"
    request_timeout_s: float = 3600.0

    # ── Retrieval ────────────────────────────────────────────────────────────
    top_k: int = 5  # scored retrieval depth (for the avg_hit_score diagnostic)
    gen_context_k: int = 25  # passages handed to the generator (the expensive knob)
    probe_k: int = 50  # gold-reachability probe depth (diagnostic only; not fed to LLM)
    recall_scope: str = "both"  # stm | ltm | both
    promote_to_ltm: bool = True

    # ── Generation / judging ─────────────────────────────────────────────────
    generation_model: str = ""  # → LLM_MODEL from .env (required — no hardcoded fallback)
    judge_model: str = ""  # → JUDGE_MODEL from .env, falling back to generation_model

    # ── Lifecycle ────────────────────────────────────────────────────────────
    session_prefix: str = "eval"
    purge: bool = True  # purge each unit's session after querying
    pre_run_purge_ltm: bool = True  # wipe all LTM before the run (no cross-run contamination)

    # ── Paths ────────────────────────────────────────────────────────────────
    data_root: Path = Path("evaluation/data")
    output_dir: Path = Path("evaluation/results")

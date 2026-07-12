"""Base configuration shared by every benchmark.

Each benchmark subclasses :class:`BaseEvalConfig` with its own dataset paths,
session prefix, and output directory. The CLI overrides the run window and
mode; everything else lives in the config classes.
"""

from __future__ import annotations

import os
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


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

    # ── Isolation ────────────────────────────────────────────────────────────
    # Each benchmark runs in its own physical database pair (stm_<ns>/ltm_<ns>), so
    # benchmarks coexist without contaminating each other and a run can reuse an
    # already-ingested graph (--no-ingest) instead of re-ingesting. Set per benchmark.
    namespace: str = "eval"

    # ── Concurrency ──────────────────────────────────────────────────────────
    # Each ingest unit (one conversation) lives in its OWN physical database pair
    # (namespace = <benchmark>_<mode>_<group_id>), so conversations share no graph
    # and can run fully in parallel — the ENTITY(name) UNIQUE MERGE race that
    # forced sequential ingest only exists *within* one namespace. These caps bound
    # the fan-out against the MCP server / ArcadeDB / the LLM provider's rate limit.
    unit_concurrency: int = Field(
        default_factory=lambda: int(os.environ.get("EVAL_UNIT_CONCURRENCY", "3"))
    )
    question_concurrency: int = Field(
        default_factory=lambda: int(os.environ.get("EVAL_QUESTION_CONCURRENCY", "5"))
    )

    # ── Lifecycle ────────────────────────────────────────────────────────────
    session_prefix: str = "eval"
    # Persist ingested data by default so a follow-up --no-ingest run can reuse it.
    # Namespace isolation (not purging) is what keeps benchmarks from contaminating
    # each other; a clean slate for one benchmark is `--reset` (drop its namespace).
    purge: bool = False  # purge each unit's session after querying

    # ── Paths ────────────────────────────────────────────────────────────────
    data_root: Path = Path("evaluation/data")
    output_dir: Path = Path("evaluation/results")

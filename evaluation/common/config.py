"""Base configuration shared by every benchmark.

Each benchmark subclasses :class:`BaseEvalConfig` with its own dataset paths,
session prefix, and output directory. The CLI overrides the run window and
mode; everything else lives in the config classes.
"""

from __future__ import annotations

import os
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator


class BaseEvalConfig(BaseModel):
    """Benchmark-agnostic knobs for one evaluation workflow."""

    model_config = ConfigDict(frozen=True)

    # ── MCP server ───────────────────────────────────────────────────────────
    command: str = "graphknows-mcp"
    request_timeout_s: float = 3600.0

    # ── Retrieval ────────────────────────────────────────────────────────────
    top_k: int = 5  # scored retrieval depth (for the avg_hit_score diagnostic)
    # Passages handed to the generator (the expensive knob). Overridable per run
    # via EVAL_GEN_CONTEXT_K so a caller can sweep 25/50/100 without code edits.
    # NOTE: the MCP memory_query tool clamps top_k at 100, so 100 is the ceiling
    # a single retrieval can feed the generator.
    # Default 50, raised from 25: evidence-rank attribution on conv-26 found 29 of
    # 81 failures had their gold evidence retrieved at ranks 26-36 — reachable,
    # fetched, then cut by the 25 window. The adapter re-sorts passages
    # chronologically for the prompt, so inclusion is what matters, not rank.
    gen_context_k: int = Field(
        default_factory=lambda: int(os.environ.get("EVAL_GEN_CONTEXT_K", "50"))
    )
    probe_k: int = 50  # gold-reachability probe depth (diagnostic only; not fed to LLM)
    recall_scope: str = "both"  # stm | ltm | both
    # Retrieval strategy override: "" (server default), "det", or "agentic".
    # Independent of GRAPHKNOWS_MODE, so an A/B can hold ingestion fixed and
    # vary only the retriever -- previously impossible without a second ingest.
    retrieval_strategy: str = ""
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
    # Ingest and answer are decoupled because they stress different resources:
    #
    #   Ingest is a bounded QUEUE, and its width is 1. Every conversation shares
    #   ONE extraction model in the MCP-server process, and that model's
    #   inference() is not reentrant — it is serialised behind a lock (see
    #   gliner_model._SerialInference). So concurrent conversations cannot
    #   actually extract in parallel: they only queue on that lock, and the wait
    #   counts against each document's MCP request timeout. At width 3 that
    #   timed out 20 documents across four conversations and left their graphs
    #   half-built. Width 1 is not a throughput sacrifice — the model was always
    #   the serial resource — it just stops the queue forming inside the timeout.
    #
    #   Answer/retrieve is PARALLEL. Once a conversation is ingested it is dropped
    #   from the ingest queue and its questions answer concurrently; answering is
    #   read-only + a remote LLM call (no local models), so it is cheap. Every
    #   conversation's questions share one global `question_concurrency` cap, which
    #   bounds total in-flight LLM calls against the provider's rate limit. This is
    #   what keeps a full run's wall-clock down: later conversations ingest while
    #   earlier ones answer.
    ingest_workers: int = Field(
        default_factory=lambda: int(os.environ.get("EVAL_INGEST_WORKERS", "1"))
    )
    question_concurrency: int = Field(
        default_factory=lambda: int(os.environ.get("EVAL_QUESTION_CONCURRENCY", "8"))
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

    @model_validator(mode="after")
    def _probe_covers_gen_context(self) -> BaseEvalConfig:
        """Guarantee the probe returns at least as many passages as the generator
        consumes. The pipeline slices ``context = probe[:gen_context_k]``, so if
        ``probe_k < gen_context_k`` the generator is silently starved (e.g. a
        gen_context_k of 100 against the default probe_k of 50 would only ever
        feed 50 passages). Auto-raise probe_k to cover it (frozen model → bypass
        the setattr guard once, during validation)."""
        if self.probe_k < self.gen_context_k:
            object.__setattr__(self, "probe_k", self.gen_context_k)
        return self

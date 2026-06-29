"""The single hardcoded LoCoMo evaluation workflow.

There is exactly one supported configuration — the generation-mode pipeline
(gliner2 extraction, promote-to-LTM, DSPy answer generation) scored with
**DSPy SemanticF1** as the headline metric and lexical token-F1 as a cheap
diagnostic secondary column. No flag matrix; no metric toggle.

The CLI overrides ``limit``, ``offset``, ``mode``, and ``--compare``;
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
    request_timeout_s: float = 3600.0

    # ── Dataset ──────────────────────────────────────────────────────────────
    data_root: Path = Path("evaluation/data/locomo")
    questions_file: str = "questions.jsonl"
    max_documents: int = 50
    # 8000 covers the longest LoCoMo session (7187 chars). At 4000, 24.6% of
    # sessions were truncated — silently dropping 7.1% of all conversation text
    # (and any gold evidence in the tail). This is the MISSING bucket in the
    # retrieval probe.
    max_chars_per_document: int = 8000

    # ── Retrieval ────────────────────────────────────────────────────────────
    # "Full blast": feed every question the maximum retrieval depth (the retriever
    # caps top_k at 25) and hand all of it to the generator — no per-question-type
    # gating. Reachability@20 was only 55%, so we surface as much as we can.
    top_k: int = 5  # scored retrieval depth (for the avg_hit_score diagnostic)
    # gen_context_k is the EXPENSIVE knob: generation input ≈ gen_context_k passages
    # (~7.7k tokens/q at 50, ~3.9k at 25) and dominates credit spend. Keep it lean.
    # probe_k stays 50 only to measure reach@50 as a FREE diagnostic (the retrieval
    # call is deterministic/no-LLM; only the passages fed to the generator cost).
    # With FUSION_ALPHA=0.05, reach@25=0.53 (> the 0.49 pre-tuning baseline), so 25
    # keeps F1 above baseline at half the input cost of 50. A cosine rerank was tried
    # to feed fewer (~12) but did NOT help: gold is spread across ranks 1-50, not
    # buried-but-high-cosine, so reach@12 stayed ~0.41 for any rerank weight. Cutting
    # below 25 trades F1 for cost on a real curve (reach@10=0.40, @25=0.53, @50=0.65);
    # further reduction needs better retrieval RECALL (chunking/embeddings), not rerank.
    gen_context_k: int = 25  # passages handed to the generator (head of the probe)
    probe_k: int = 50  # gold-reachability probe depth (diagnostic only; not fed to LLM)
    recall_scope: str = "both"  # stm | ltm | both
    promote_to_ltm: bool = True

    # ── Generation ───────────────────────────────────────────────────────────
    generation_model: str = ""  # → LLM_MODEL from .env (required — no hardcoded fallback)

    # ── Lifecycle ────────────────────────────────────────────────────────────
    session_prefix: str = "eval-locomo"
    purge: bool = True  # purge each dialogue session after querying
    pre_run_purge_ltm: bool = True  # wipe all LTM before the run (no cross-run contamination)

    # ── Output ───────────────────────────────────────────────────────────────
    output_dir: Path = Path("evaluation/results/locomo")

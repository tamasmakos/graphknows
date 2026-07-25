"""Local cross-encoder reranker — the sole ranker of the fused pool.

Public API
----------
rerank(query, candidates, top_k) -> list[dict]
    Re-score *candidates* (fused result dicts from DETRetriever) with a local
    CPU cross-encoder and return the top *top_k* sorted by descending score.
    By default the ENTIRE candidate pool is scored — there is no pre-ranking
    upstream left to trust, so nothing may be silently dropped before scoring.

No fallbacks
------------
There is no identity/no-op path: if ``sentence_transformers`` is missing or the
cross-encoder model cannot be loaded, ``rerank()`` raises. The configured model
is the only one used.

The reranker is lazy-initialised: the model is loaded at most once per process
the first time ``rerank()`` is called with a non-empty candidate list.  No
import-time cost is incurred.
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Any

log = logging.getLogger(__name__)

# The reranker. No fallback — this exact model is loaded or rerank() raises.
# A small (~22M) MiniLM cross-encoder: fast on CPU or GPU, negligible memory.
_CROSS_ENCODER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L6-v2"

# Optional escape hatch to bound latency on very large RRF pools. Default is
# unbounded (0 = no cap): the cross-encoder is the only ranker, so it must see
# every candidate unless an operator explicitly opts into truncation.
_RERANK_CANDIDATE_CAP: int = int(os.environ.get("GRAPHKNOWS_RERANK_CANDIDATE_CAP", "0"))

# Singleton state ─ protected by a module-level lock so the first concurrent
# call does not race on initialisation.
_lock = threading.Lock()
_model: Any = None  # CrossEncoder instance
_model_loaded: bool = False  # True once the model has been loaded


def _load_model() -> Any:
    """Load the cross-encoder. Raises on any failure — there is no fallback.

    Device comes from settings (``rerank_device``: "auto"/"cuda"/"cpu").
    """
    import torch
    from sentence_transformers import CrossEncoder  # type: ignore[import-untyped]

    from graphknows.settings import GraphKnowsSettings

    s = GraphKnowsSettings()
    want = (getattr(s, "rerank_device", "auto") or "auto").lower()
    device = "cuda" if want in ("auto", "cuda") and torch.cuda.is_available() else "cpu"

    ce = CrossEncoder(_CROSS_ENCODER_MODEL, max_length=512, device=device)
    log.info("Cross-encoder loaded: %s (device=%s)", _CROSS_ENCODER_MODEL, device)
    return ce


def _get_model() -> Any:
    """Return the (cached) cross-encoder, loading it on first call."""
    global _model, _model_loaded
    if not _model_loaded:
        with _lock:
            if not _model_loaded:  # double-checked locking
                _model = _load_model()
                _model_loaded = True
    return _model


def rerank(
    query: str,
    candidates: list[dict[str, Any]],
    top_k: int,
) -> list[dict[str, Any]]:
    """Re-rank *candidates* with a local cross-encoder and return top *top_k*.

    Args:
        query: The original natural-language query.
        candidates: Fused result dicts (must contain at least a ``"text"`` key).
            They are not mutated — a new list is returned.
        top_k: Maximum number of results to return.

    Returns:
        A new list of result dicts, sorted by cross-encoder score descending,
        truncated to ``top_k``.
    """
    if not candidates:
        return candidates

    model = _get_model()

    # Score the whole pool by default. GRAPHKNOWS_RERANK_CANDIDATE_CAP (0 =
    # unbounded) is an explicit opt-in escape hatch for latency-bound deployments.
    pool = candidates[:_RERANK_CANDIDATE_CAP] if _RERANK_CANDIDATE_CAP > 0 else candidates
    texts = [c.get("text") or "" for c in pool]

    pairs = [(query, t) for t in texts]
    # MiniLM cross-encoder returns a single relevance logit per pair; higher = more
    # relevant. Downstream only sorts on these, so raw scores are used as-is.
    scores: list[float] = model.predict(pairs, show_progress_bar=False).tolist()

    # Attach cross-encoder score and sort.
    scored = sorted(
        zip(scores, pool, strict=False),
        key=lambda x: x[0],
        reverse=True,
    )

    # Rebuild result dicts, overwriting rrf_score with the cross-encoder score
    # and updating rank indices so downstream consumers see a coherent view.
    reranked: list[dict[str, Any]] = []
    for new_rank, (ce_score, item) in enumerate(scored[:top_k]):
        updated = dict(item)
        updated["rrf_score"] = round(float(ce_score), 6)
        updated["rank"] = new_rank + 1
        reranked.append(updated)

    return reranked

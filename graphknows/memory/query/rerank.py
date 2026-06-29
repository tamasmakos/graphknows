"""Local cross-encoder reranker for post-RRF refinement.

Public API
----------
rerank(query, candidates, top_k) -> list[dict]
    Re-score *candidates* (fused result dicts from DETRetriever) with a local
    CPU cross-encoder and return the top *top_k* sorted by descending score.

Offline / unavailable model behaviour
--------------------------------------
When ``sentence_transformers`` or ``torch`` are not installed, or when the
cross-encoder model cannot be loaded (network unavailable, disk absent), the
function logs a single WARNING and returns the input list unchanged (identity
pass-through).  All callers must handle the no-op path — the function never
raises.

The reranker is lazy-initialised: the model is loaded at most once per process
the first time ``rerank()`` is called with a non-empty candidate list.  No
import-time cost is incurred.
"""

from __future__ import annotations

import logging
import threading
from typing import Any

log = logging.getLogger(__name__)

# BGE-family cross-encoder — small and CPU-efficient.  Falls back to no-op if
# the model file is absent or the library is not installed.
_CROSS_ENCODER_MODEL: str = "BAAI/bge-reranker-base"

# Limit the number of candidates passed to the cross-encoder so latency stays
# predictable even with large RRF pools.
_RERANK_CANDIDATE_CAP: int = 50

# Singleton state ─ protected by a module-level lock so the first concurrent
# call does not race on initialisation.
_lock = threading.Lock()
_model: Any = None          # CrossEncoder instance or sentinel _NO_MODEL
_model_loaded: bool = False  # True once a load attempt has been made


class _NoModel:
    """Sentinel object returned when the cross-encoder cannot be loaded."""


_NO_MODEL = _NoModel()


def _load_model() -> Any:
    """Attempt to load the cross-encoder.  Returns ``_NO_MODEL`` on any failure."""
    try:
        from sentence_transformers import CrossEncoder  # type: ignore[import-untyped]

        ce = CrossEncoder(_CROSS_ENCODER_MODEL, max_length=512, device="cpu")
        log.info("Cross-encoder loaded: %s", _CROSS_ENCODER_MODEL)
        return ce
    except ImportError:
        log.warning(
            "sentence_transformers not installed — cross-encoder rerank is a no-op; "
            "install sentence-transformers[cpu] to enable."
        )
        return _NO_MODEL
    except Exception as exc:  # model file missing, network error, etc.
        log.warning(
            "Cross-encoder model %r could not be loaded (%s) — rerank is a no-op.",
            _CROSS_ENCODER_MODEL,
            exc,
        )
        return _NO_MODEL


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
        truncated to ``top_k``.  When the model is unavailable the input list
        is returned as-is (already sorted, already truncated by the caller).
    """
    if not candidates:
        return candidates

    model = _get_model()
    if isinstance(model, _NoModel):
        # No-op path: return unchanged list (caller already applied top_k via
        # shaped_combined[:top_k], so we just honour whatever arrived).
        return candidates[:top_k]

    # Cap the candidates fed to the encoder to bound latency.
    pool = candidates[:_RERANK_CANDIDATE_CAP]
    texts = [c.get("text") or "" for c in pool]

    try:
        pairs = [(query, t) for t in texts]
        scores: list[float] = model.predict(pairs, show_progress_bar=False).tolist()
    except Exception as exc:
        log.warning("Cross-encoder inference failed (%s) — returning pre-rerank order.", exc)
        return candidates[:top_k]

    # Attach cross-encoder score and sort.
    scored = sorted(
        zip(scores, pool),
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

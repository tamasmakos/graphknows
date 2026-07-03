"""Neural coreference resolution (llm_free-legal: local deterministic model, no LLM/API).

Wraps the cached fastcoref model to rewrite third-person pronouns to their
antecedent named mention BEFORE relation extraction, so facts buried behind
"she / he / they" attach to the right entity node instead of being dropped.

First- and second-person pronouns (I / you / we) are deliberately left untouched:
in multi-speaker transcripts they are turn-relative and are resolved
deterministically downstream against the ``Speaker:`` label. This replaces the
earlier recency ("nearest preceding name") heuristic with a real model.
"""

from __future__ import annotations

import logging
import threading
from typing import Any

log = logging.getLogger(__name__)

# Third-person SINGULAR animate pronouns rewritten to their cluster's named
# representative. First/second person stay turn-relative (resolved downstream
# against the ``Speaker:`` label). Plural "they/them" is deliberately excluded:
# in casual dialogue it clusters to inanimate/plural antecedents ("books",
# "nature and quality time") and corrupts the text with no fact-coverage gain.
_THIRD_PERSON = frozenset(
    {
        "he", "him", "his", "himself",
        "she", "her", "hers", "herself",
    }
)

_model: Any = None
_load_failed = False
_lock = threading.Lock()


def _shim_transformers() -> None:
    """Backfill transformers>=5's per-instance ``all_tied_weights_keys`` attribute.

    fastcoref's custom model classes predate it, so ``from_pretrained`` raises
    ``AttributeError`` during weight loading; an empty class-level default is a
    safe no-op fallback (these models have no tied weights we rely on).
    """
    for mod, cls in (
        ("fastcoref.coref_models.modeling_fcoref", "FCorefModel"),
        ("fastcoref.coref_models.modeling_lingmess", "LingMessModel"),
    ):
        try:
            klass = getattr(__import__(mod, fromlist=[cls]), cls)
            if not hasattr(klass, "all_tied_weights_keys"):
                klass.all_tied_weights_keys = {}
        except Exception:  # pragma: no cover - model class may be absent
            pass


def _get_model() -> Any:
    """Lazily load a single CPU coref model; on failure, disable coref (soft-fail)."""
    global _model, _load_failed
    if _model is not None or _load_failed:
        return _model
    with _lock:
        if _model is None and not _load_failed:
            try:
                _shim_transformers()
                from fastcoref import FCoref

                _model = FCoref(device="cpu")
            except Exception as exc:  # pragma: no cover - environment guard
                log.warning("coref unavailable, skipping (%s)", exc)
                _load_failed = True
    return _model


def _is_name(mention: str) -> bool:
    """A usable antecedent: a capitalised, non-pronominal mention (a name)."""
    m = mention.strip()
    return bool(m) and m.lower() not in _THIRD_PERSON and m[:1].isupper()


def resolve_coref(text: str) -> str:
    """Rewrite third-person pronouns to their coreferent named mention.

    Returns ``text`` unchanged if the model is unavailable or nothing resolves,
    so callers can enable coref without a hard dependency at ingest time.
    """
    model = _get_model()
    if model is None or not text.strip():
        return text
    try:
        pred = model.predict(texts=[text])[0]
        clusters_spans = pred.get_clusters(as_strings=False)
        clusters_strs = pred.get_clusters(as_strings=True)
    except Exception as exc:  # pragma: no cover - inference guard
        log.debug("coref predict failed: %s", exc)
        return text

    repls: list[tuple[int, int, str]] = []
    for spans, strs in zip(clusters_spans, clusters_strs, strict=False):
        rep = next((s for s in strs if _is_name(s)), None)
        if rep is None:
            continue  # cluster has no named antecedent -> leave pronouns as-is
        for (start, end), mention in zip(spans, strs, strict=False):
            if mention.strip().lower() in _THIRD_PERSON:
                repls.append((start, end, rep))
    for start, end, rep in sorted(repls, key=lambda x: -x[0]):  # right-to-left, offset-safe
        text = text[:start] + rep + text[end:]
    return text

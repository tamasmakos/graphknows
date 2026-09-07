"""Shared spaCy model loading.

A spaCy model such as ``en_core_web_lg`` is not on PyPI — PyPI rejects
direct-URL requirements — so it can never be a ``pyproject.toml`` dependency
or extra. It only ever reaches an environment via the documented post-install
step (``python -m spacy download <model>``, see ``docs/configuration.md``).
The only defence against a missing model is therefore one loud, actionable
error raised at load time, shared by every call site instead of each one
surfacing spaCy's raw, remedy-less ``OSError``.

Every call also memoizes on ``(name, options)``: a large model costs hundreds
of MB, so only one pipeline is resident per configuration no matter how many
callers ask for it. Ingest workers reach this via ``asyncio.to_thread``, so the
memoization is guarded by a lock (the same double-checked pattern
``symbolic/ontology/catalog.py`` uses) rather than racing several threads into
loading the same model concurrently on a cold start.
"""

from __future__ import annotations

import threading
from typing import Any

from graphknows.exceptions import MissingModelError

_lock = threading.Lock()
_MODELS: dict[tuple[Any, ...], Any] = {}


def load_spacy_model(name: str, **options: Any) -> Any:
    """Load spaCy model *name*, raising :class:`MissingModelError` if absent.

    Returns the pipeline already loaded for this ``(name, options)``, if any.
    """
    key = (
        name,
        tuple(sorted((k, tuple(v) if isinstance(v, list) else v) for k, v in options.items())),
    )
    if key in _MODELS:
        return _MODELS[key]

    with _lock:
        if key in _MODELS:  # another thread won the race while we waited
            return _MODELS[key]

        import spacy

        try:
            model = spacy.load(name, **options)
        except OSError as exc:
            raise MissingModelError(name) from exc
        _MODELS[key] = model
        return model

"""Shared spaCy model loading.

A spaCy model such as ``en_core_web_lg`` is not on PyPI — PyPI rejects
direct-URL requirements — so it can never be a ``pyproject.toml`` dependency
or extra. It only ever reaches an environment via the documented post-install
step (``python -m spacy download <model>``, see ``docs/configuration.md``).
The only defence against a missing model is therefore one loud, actionable
error raised at load time, shared by every call site instead of each one
surfacing spaCy's raw, remedy-less ``OSError``.
"""

from __future__ import annotations

from typing import Any

from graphknows.exceptions import MissingModelError


def load_spacy_model(name: str, **options: Any) -> Any:
    """Load spaCy model *name*, raising :class:`MissingModelError` if absent."""
    import spacy

    try:
        return spacy.load(name, **options)
    except OSError as exc:
        raise MissingModelError(name) from exc

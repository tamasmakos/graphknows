"""Shared handle every golden-layer writer holds."""

from __future__ import annotations

from graphknows.storage.arcadedb._base import ArcadeStoreBase


class _Writer:
    """One database, held by every writer; the writes themselves differ."""

    def __init__(self, store: ArcadeStoreBase) -> None:
        self._store = store

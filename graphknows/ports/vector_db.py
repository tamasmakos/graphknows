"""Deprecated alias module — use :mod:`graphknows.ports.vector_store` instead.

``IVectorDb`` was renamed to ``IVectorStore`` in the 1.0 port set. This shim
keeps the old import path working for one release; it will be removed in 2.0.
"""

from __future__ import annotations

from graphknows.ports.vector_store import IVectorStore

# Backwards-compatible alias (deprecated).
IVectorDb = IVectorStore

__all__ = ["IVectorDb"]

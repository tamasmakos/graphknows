"""Stores sub-package: embedder, Neo4j, pgvector, Memgraph.

.. deprecated::
    Direct imports from ``graphknows.retrieval.stores`` are deprecated.
    Import concrete adapters from ``graphknows.adapters`` instead.
    This module will be removed in Wave 5.
"""

from __future__ import annotations

import importlib
import warnings

__all__ = [
    "MemgraphStore",
    "Neo4jStore",
    "PgVectorStore",
    "embed",
    "embed_one",
    "get_embedder",
]

_LAZY_MAP: dict[str, tuple[str, str]] = {
    "MemgraphStore": ("graphknows.retrieval.stores.memgraph", "MemgraphStore"),
    "Neo4jStore": ("graphknows.retrieval.stores.neo4j", "Neo4jStore"),
    "PgVectorStore": ("graphknows.retrieval.stores.pgvector", "PgVectorStore"),
    "embed": ("graphknows.retrieval.stores.embedder", "embed"),
    "embed_one": ("graphknows.retrieval.stores.embedder", "embed_one"),
    "get_embedder": ("graphknows.retrieval.stores.embedder", "get_embedder"),
}

_REPLACEMENT: dict[str, str] = {
    "MemgraphStore": "graphknows.adapters.memgraph_adapter.MemgraphAdapter",
    "Neo4jStore": "graphknows.adapters.neo4j_adapter.Neo4jAdapter",
    "PgVectorStore": "graphknows.adapters.pgvector_adapter.PgVectorAdapter",
    "embed": "graphknows.adapters.embedder_adapter.EmbedderAdapter",
    "embed_one": "graphknows.adapters.embedder_adapter.EmbedderAdapter",
    "get_embedder": "graphknows.adapters.embedder_adapter.EmbedderAdapter",
}


def __getattr__(name: str) -> object:
    """PEP 562 lazy import with DeprecationWarning."""
    if name in _LAZY_MAP:
        warnings.warn(
            f"graphknows.retrieval.stores.{name} is deprecated. "
            f"Use {_REPLACEMENT[name]} or the port interfaces instead. "
            "This shim will be removed in Wave 5.",
            DeprecationWarning,
            stacklevel=2,
        )
        mod_name, attr = _LAZY_MAP[name]
        mod = importlib.import_module(mod_name)
        return getattr(mod, attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

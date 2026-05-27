"""Stores sub-package: embedder, Neo4j, pgvector, Memgraph.

.. deprecated::
    Direct imports from ``graphknows.retrieval.stores`` are deprecated.
    Import concrete adapters from ``graphknows.adapters`` instead.
    This module will be removed in Wave 5.
"""

from __future__ import annotations

import warnings

from graphknows.retrieval.stores.embedder import embed, embed_one, get_embedder
from graphknows.retrieval.stores.memgraph import MemgraphStore
from graphknows.retrieval.stores.neo4j import Neo4jStore
from graphknows.retrieval.stores.pgvector import PgVectorStore

__all__ = [
    "MemgraphStore",
    "Neo4jStore",
    "PgVectorStore",
    "embed",
    "embed_one",
    "get_embedder",
]

_DEPRECATED_NAMES: dict[str, str] = {
    "MemgraphStore": "graphknows.adapters.memgraph_adapter.MemgraphAdapter",
    "Neo4jStore": "graphknows.adapters.neo4j_adapter.Neo4jAdapter",
    "PgVectorStore": "graphknows.adapters.pgvector_adapter.PgVectorAdapter",
    "embed": "graphknows.adapters.embedder_adapter.EmbedderAdapter",
    "embed_one": "graphknows.adapters.embedder_adapter.EmbedderAdapter",
    "get_embedder": "graphknows.adapters.embedder_adapter.EmbedderAdapter",
}


def __getattr__(name: str) -> object:
    """PEP 562 module __getattr__ for DeprecationWarning on store imports."""
    if name in _DEPRECATED_NAMES:
        warnings.warn(
            f"graphknows.retrieval.stores.{name} is deprecated. "
            f"Use {_DEPRECATED_NAMES[name]} or the IGraphDb/ISessionDb/IVectorDb ports instead. "
            "This shim will be removed in Wave 5.",
            DeprecationWarning,
            stacklevel=2,
        )
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

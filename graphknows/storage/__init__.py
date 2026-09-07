"""Persistence layer: the ArcadeDB backend, embeddings, and namespace-aware store factories.

The factories build unconnected stores from
:class:`~graphknows.settings.GraphKnowsSettings`
and a namespace; the caller awaits ``connect()`` on its own event loop. They are
mode-neutral — the relation extractor is injected by the ingestion layer, never
built here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from graphknows.settings import GraphKnowsSettings
from graphknows.storage.namespace import db_name

if TYPE_CHECKING:
    from graphknows.storage.arcadedb.client import ArcadeDBClient
    from graphknows.storage.arcadedb.graph_store import GraphStore


def build_arcadedb_client(settings: GraphKnowsSettings) -> ArcadeDBClient:
    """Build an ArcadeDBClient from settings (not yet connected)."""
    from graphknows.storage.arcadedb.client import ArcadeDBClient

    return ArcadeDBClient(
        base_url=settings.arcadedb_url,
        user=settings.arcadedb_user,
        password=settings.arcadedb_password.get_secret_value(),
    )


def build_graph_store(settings: GraphKnowsSettings, namespace: str = "") -> GraphStore:
    """Build a single-database :class:`GraphStore` for a namespace (golden layer).

    The store is not yet connected — the caller awaits ``connect()`` (and
    ``ensure_schema(dims)`` on write paths) on its own event loop.
    """
    from graphknows.storage.arcadedb.graph_store import GraphStore

    return GraphStore(client=build_arcadedb_client(settings), db=db_name(namespace))


def __getattr__(name: str) -> Any:
    """Lazily expose the ArcadeDB adapters + embedder without importing heavy deps.

    (sentence-transformers) at ``import graphknows.storage`` time.
    """
    lazy = {
        "ArcadeDBClient": ("graphknows.storage.arcadedb", "ArcadeDBClient"),
        "GraphStore": ("graphknows.storage.arcadedb.graph_store", "GraphStore"),
        "embed": ("graphknows.storage.embedder", "embed"),
        "embed_one": ("graphknows.storage.embedder", "embed_one"),
    }
    target = lazy.get(name)
    if target is None:
        raise AttributeError(f"module 'graphknows.storage' has no attribute {name!r}")
    import importlib

    module, attr = target
    return getattr(importlib.import_module(module), attr)


__all__ = [
    "build_arcadedb_client",
    "build_graph_store",
    "db_name",
]

"""Persistence layer: store protocols, the ArcadeDB backend, embeddings, and
namespace-aware store factories.

The factories build unconnected stores from :class:`~graphknows.settings.GraphKnowsSettings`
and a namespace; the caller awaits ``connect()`` on its own event loop. They are
mode-neutral — the relation extractor is injected by the ingestion layer, never
built here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from graphknows.settings import GraphKnowsSettings
from graphknows.storage.base import IEmbedder, ILTMStore, ISTMStore
from graphknows.storage.namespace import db_name, db_names

if TYPE_CHECKING:
    from graphknows.storage.arcadedb.client import ArcadeDBClient
    from graphknows.storage.arcadedb.graph_store import GraphStore
    from graphknows.storage.arcadedb.ltm_store import ArcadeDBLTMStore
    from graphknows.storage.arcadedb.stm_store import ArcadeDBSTMStore


def build_arcadedb_client(settings: GraphKnowsSettings) -> ArcadeDBClient:
    """Build an ArcadeDBClient from settings (not yet connected)."""
    from graphknows.storage.arcadedb.client import ArcadeDBClient

    return ArcadeDBClient(
        base_url=settings.arcadedb_url,
        user=settings.arcadedb_user,
        password=settings.arcadedb_password,
    )


def build_arcadedb_stm_store(
    settings: GraphKnowsSettings, namespace: str = ""
) -> ArcadeDBSTMStore:
    """Build an ArcadeDBSTMStore (client not yet connected — call ensure_schema after connect)."""
    from graphknows.storage.arcadedb.stm_store import ArcadeDBSTMStore

    stm_db, _ = db_names(namespace)
    return ArcadeDBSTMStore(client=build_arcadedb_client(settings), db=stm_db)


def build_arcadedb_ltm_store(
    settings: GraphKnowsSettings,
    namespace: str = "",
    relation_extractor: Any | None = None,
) -> ArcadeDBLTMStore:
    """Build an ArcadeDBLTMStore. The relation extractor (if any) is injected by
    the caller — read/purge paths pass ``None``; llm_assisted ingestion passes the
    mode-appropriate extractor."""
    from graphknows.storage.arcadedb.ltm_store import ArcadeDBLTMStore

    _, ltm_db = db_names(namespace)
    return ArcadeDBLTMStore(
        client=build_arcadedb_client(settings),
        relation_extractor=relation_extractor,
        db=ltm_db,
    )


def build_graph_store(settings: GraphKnowsSettings, namespace: str = "") -> GraphStore:
    """Build a single-database :class:`GraphStore` for a namespace (golden layer).

    The store is not yet connected — the caller awaits ``connect()`` (and
    ``ensure_schema(dims)`` on write paths) on its own event loop.
    """
    from graphknows.storage.arcadedb.graph_store import GraphStore

    return GraphStore(client=build_arcadedb_client(settings), db=db_name(namespace))


def __getattr__(name: str) -> Any:
    """Lazily expose the ArcadeDB adapters + embedder without importing heavy deps
    (sentence-transformers) at ``import graphknows.storage`` time."""
    lazy = {
        "ArcadeDBClient": ("graphknows.storage.arcadedb", "ArcadeDBClient"),
        "ArcadeDBLTMStore": ("graphknows.storage.arcadedb", "ArcadeDBLTMStore"),
        "ArcadeDBSTMStore": ("graphknows.storage.arcadedb", "ArcadeDBSTMStore"),
        "GraphStore": ("graphknows.storage.arcadedb.graph_store", "GraphStore"),
        "embed": ("graphknows.storage.embedder", "embed"),
        "embed_one": ("graphknows.storage.embedder", "embed_one"),
        "get_embedder": ("graphknows.storage.embedder", "get_embedder"),
    }
    target = lazy.get(name)
    if target is None:
        raise AttributeError(f"module 'graphknows.storage' has no attribute {name!r}")
    import importlib

    module, attr = target
    return getattr(importlib.import_module(module), attr)


__all__ = [
    "IEmbedder",
    "ILTMStore",
    "ISTMStore",
    "build_arcadedb_client",
    "build_arcadedb_ltm_store",
    "build_arcadedb_stm_store",
    "build_graph_store",
    "db_name",
    "db_names",
]

"""Lazy settings + store accessors for MCP tool modules."""

from __future__ import annotations

from typing import Any

from graphknows.config import GraphKnowsSettings

_settings: GraphKnowsSettings | None = None
_pg: Any = None
_pg_connected: bool = False
_pg_bridge: Any = None
_pg_bridge_connected: bool = False
_retriever: Any = None
_runtime: Any = None


def get_settings() -> GraphKnowsSettings:
    """Return singleton settings instance."""
    global _settings
    if _settings is None:
        _settings = GraphKnowsSettings()
    return _settings


def get_runtime() -> Any:
    """Return the singleton transport-neutral graphknows runtime."""
    global _runtime
    if _runtime is None:
        from graphknows.runtime import GraphKnowsRuntime

        _runtime = GraphKnowsRuntime(get_settings())
    return _runtime


def get_mg_store() -> Any:
    """Build a MemgraphStore from current settings."""
    from graphknows.config.builders.stores import build_memgraph_store

    return build_memgraph_store(get_settings())


async def get_pg_store() -> Any:
    """Return a connected PgVectorStore singleton."""
    global _pg, _pg_connected
    if _pg is None:
        from graphknows.config.builders.stores import build_pgvector_store

        _pg = build_pgvector_store(get_settings())
    if not _pg_connected:
        await _pg.connect()
        _pg_connected = True
    return _pg


def get_pg_store_bridge() -> Any:
    """Return a connected PgVectorStore singleton bound to the background recall loop."""
    global _pg_bridge, _pg_bridge_connected
    if _pg_bridge is None:
        from graphknows.config.builders.stores import build_pgvector_store

        _pg_bridge = build_pgvector_store(get_settings())
    if not _pg_bridge_connected:
        from graphknows.memory.stm._async_bridge import _run

        _run(_pg_bridge.connect())
        _pg_bridge_connected = True
    return _pg_bridge


def get_neo_store() -> Any:
    """Build a Neo4jStore from current settings."""
    from graphknows.config.builders.stores import build_neo4j_store

    return build_neo4j_store(get_settings())


async def get_or_init_retriever() -> Any:
    """Return the singleton retriever, initialising it on first call."""
    global _retriever
    if _retriever is not None:
        return _retriever
    s = get_settings()
    pg = await get_pg_store()
    neo = get_neo_store()
    if s.retrieval_mode == "agentic":
        from graphknows.memory.query.agentic import AgenticRetriever

        _retriever = AgenticRetriever(
            settings=s,
            pg=pg,
            neo=neo,
            memgraph=get_mg_store(),
        )
    else:
        from graphknows.memory.query.retriever import DETRetriever

        _retriever = DETRetriever(pg=pg, neo=neo, mg=get_mg_store(), embed_model=s.embed_model)
    return _retriever

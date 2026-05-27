"""Lazy settings + store accessors for MCP tool modules."""

from __future__ import annotations

from typing import Any

from graphknows.config import GraphKnowsSettings

_settings: GraphKnowsSettings | None = None
_pg: Any = None
_pg_connected: bool = False


def get_settings() -> GraphKnowsSettings:
    """Return singleton settings instance."""
    global _settings
    if _settings is None:
        _settings = GraphKnowsSettings()
    return _settings


def get_mg_store() -> Any:
    """Build a MemgraphStore from current settings."""
    from graphknows.builders.stores import build_memgraph_store

    return build_memgraph_store(get_settings())


async def get_pg_store() -> Any:
    """Return a connected PgVectorStore singleton."""
    global _pg, _pg_connected
    if _pg is None:
        from graphknows.builders.stores import build_pgvector_store

        _pg = build_pgvector_store(get_settings())
    if not _pg_connected:
        await _pg.connect()
        _pg_connected = True
    return _pg


def get_neo_store() -> Any:
    """Build a Neo4jStore from current settings."""
    from graphknows.builders.stores import build_neo4j_store

    return build_neo4j_store(get_settings())

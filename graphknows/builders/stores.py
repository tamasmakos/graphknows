"""Shared store factory functions."""

from __future__ import annotations

from graphknows.config import GraphKnowsSettings
from graphknows.retrieval.stores.memgraph import MemgraphStore
from graphknows.retrieval.stores.neo4j import Neo4jStore
from graphknows.retrieval.stores.pgvector import PgVectorStore


def build_neo4j_store(settings: GraphKnowsSettings) -> Neo4jStore:
    """Build a Neo4jStore from settings."""
    return Neo4jStore(
        uri=settings.neo4j_uri,
        user=settings.neo4j_user,
        password=settings.neo4j_password,
        database=settings.neo4j_database,
    )


def build_pgvector_store(settings: GraphKnowsSettings) -> PgVectorStore:
    """Build a PgVectorStore from settings."""
    return PgVectorStore(dsn=settings.pgvector_dsn)


def build_memgraph_store(settings: GraphKnowsSettings) -> MemgraphStore:
    """Build a MemgraphStore from settings."""
    return MemgraphStore(uri=settings.memgraph_uri)

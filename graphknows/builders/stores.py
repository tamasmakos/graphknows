"""Shared store factory functions.

Provides both concrete store factories (for backward compatibility) and
adapter factories that return port-typed instances.
"""

from __future__ import annotations

from graphknows.adapters.embedder_adapter import EmbedderAdapter
from graphknows.adapters.memgraph_adapter import MemgraphAdapter
from graphknows.adapters.neo4j_adapter import Neo4jAdapter
from graphknows.adapters.pgvector_adapter import PgVectorAdapter
from graphknows.adapters.ports.embedder import IEmbedder
from graphknows.adapters.ports.graph_db import IGraphDb
from graphknows.adapters.ports.session_db import ISessionDb
from graphknows.adapters.ports.vector_db import IVectorDb
from graphknows.config import GraphKnowsSettings
from graphknows.retrieval.stores.memgraph import MemgraphStore
from graphknows.retrieval.stores.neo4j import Neo4jStore
from graphknows.retrieval.stores.pgvector import PgVectorStore


def build_neo4j_store(settings: GraphKnowsSettings) -> Neo4jStore:
    """Build a Neo4jStore from settings (backward-compat)."""
    return Neo4jStore(
        uri=settings.neo4j_uri,
        user=settings.neo4j_user,
        password=settings.neo4j_password,
        database=settings.neo4j_database,
    )


def build_pgvector_store(settings: GraphKnowsSettings) -> PgVectorStore:
    """Build a PgVectorStore from settings (backward-compat)."""
    return PgVectorStore(dsn=settings.pgvector_dsn)


def build_memgraph_store(settings: GraphKnowsSettings) -> MemgraphStore:
    """Build a MemgraphStore from settings (backward-compat)."""
    return MemgraphStore(uri=settings.memgraph_uri)


# --- Port-typed adapter factories (preferred for new code) ---


def build_graph_db(settings: GraphKnowsSettings) -> IGraphDb:
    """Build an IGraphDb adapter from settings."""
    return Neo4jAdapter(build_neo4j_store(settings))


def build_vector_db(settings: GraphKnowsSettings) -> IVectorDb:
    """Build an IVectorDb adapter from settings."""
    return PgVectorAdapter(build_pgvector_store(settings))


def build_session_db(settings: GraphKnowsSettings) -> ISessionDb:
    """Build an ISessionDb adapter from settings."""
    return MemgraphAdapter(build_memgraph_store(settings))


def build_embedder(model_name: str = "openai/text-embedding-3-small") -> IEmbedder:
    """Build an IEmbedder adapter."""
    return EmbedderAdapter(model_name=model_name)

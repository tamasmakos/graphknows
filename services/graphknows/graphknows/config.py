"""graphknows runtime configuration.

Importing this module has no side-effects (no FastAPI, no store connections).
Any agent or library that wants only settings can do::

    from graphknows.config import GraphKnowsSettings
    cfg = GraphKnowsSettings()
"""
from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class GraphKnowsSettings(BaseSettings):
    """Runtime configuration for the graphknows service and in-process container."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    graphknows_env: str = "development"
    graphknows_port: int = 8025

    # Databases
    neo4j_uri: str = "bolt://neo4j:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "neo4jpassword"
    neo4j_database: str = "neo4j"

    pgvector_dsn: str = "postgresql://graphuser:graphpass@pgvector:5432/graphdb"

    memgraph_uri: str = "bolt://memgraph:7687"

    # Embedding model
    embed_model: str = "BAAI/bge-small-en-v1.5"

    # Session TTL (minutes of inactivity before auto-merge)
    session_ttl_minutes: int = Field(default=60, alias="SESSION_TTL_MINUTES")

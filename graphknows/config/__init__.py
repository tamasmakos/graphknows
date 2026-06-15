"""graphknows runtime configuration.

Importing this module has no side-effects (no FastAPI, no store connections).
Any agent or library that wants only settings can do::

    from graphknows.config import GraphKnowsSettings
    cfg = GraphKnowsSettings()

Example:
    from graphknows.config import GraphKnowsSettings

    settings = GraphKnowsSettings()
    settings.check_production_secrets()
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from graphknows.config.env_loader import load_env_file

_MASKED_PGVECTOR_DSN = "postgresql://postgres:***@172.18.0.1:5432/vectordb"


def _clean_env_secret(value: str | None) -> str:
    return (value or "").strip().strip('"').strip("'")


def _find_repo_env() -> Path | None:
    """Locate the nearest repo-level .env file by walking upward from this module.

    Stops at the first directory that contains a pyproject.toml or .git marker
    so the walk never escapes the repository root.
    """
    for parent in Path(__file__).resolve().parents:
        candidate = parent / ".env"
        if candidate.exists():
            return candidate
        # Stop searching once we've reached the repo root.
        if (parent / "pyproject.toml").exists() or (parent / ".git").exists():
            break
    return None


def _load_repo_env() -> dict[str, str]:
    """Load repo-local .env values when present."""
    env_file = _find_repo_env()
    return load_env_file(env_file)


class GraphKnowsSettings(BaseSettings):
    """Runtime configuration for the graphknows service and in-process container.

    Loads from environment variables and an optional ``.env`` file.  Field
    aliases match the conventional upper-case env var names so both
    ``NEO4J_URI`` and ``neo4j_uri`` work as sources.

    Attributes:
        graphknows_env: Runtime environment tag (``"development"`` | ``"production"``).
        graphknows_port: HTTP port for the REST API.
        neo4j_uri: Bolt URI for the persistent Neo4j knowledge graph.
        neo4j_user: Neo4j username.
        neo4j_password: Neo4j password.
        neo4j_database: Neo4j database name.
        pgvector_dsn: Full PostgreSQL DSN for the pgvector store.
        pgvector_host: pgvector host (used to build DSN if DSN not set).
        pgvector_port: pgvector port.
        pgvector_db: pgvector database name.
        pgvector_user: pgvector username.
        pgvector_password: pgvector password.
        memgraph_uri: Bolt URI for the optional Memgraph session store.
        embed_model: OpenRouter embedding model name (e.g. openai/text-embedding-3-small).
        session_ttl_minutes: Minutes of session inactivity before auto-merge.
        llm_model: LLM model name for synthesis and extraction.
        llm_provider: LLM provider (``"openrouter"``).
        openrouter_api_key: OpenRouter API key.
        extraction_model: Model used for relation extraction.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
        populate_by_name=True,
    )

    graphknows_env: str = Field(default="development", alias="GRAPHKNOWS_ENV")
    graphknows_port: int = Field(default=8025, alias="GRAPHKNOWS_PORT")

    # Neo4j
    neo4j_uri: str = Field(default="bolt://172.18.0.1:7687", alias="NEO4J_URI")  # NOSONAR
    neo4j_user: str = Field(default="neo4j", alias="NEO4J_USER")
    neo4j_password: str = Field(default="testpassword123", alias="NEO4J_PASSWORD")
    neo4j_database: str = Field(default="neo4j", alias="NEO4J_DATABASE")

    # pgvector
    pgvector_dsn: str = Field(default=_MASKED_PGVECTOR_DSN, alias="PGVECTOR_DSN")
    pgvector_host: str = Field(default="172.18.0.1", alias="PGVECTOR_HOST")  # NOSONAR
    pgvector_port: int = Field(default=5432, alias="PGVECTOR_PORT")
    pgvector_db: str = Field(default="vectordb", alias="PGVECTOR_DB")
    pgvector_user: str = Field(default="postgres", alias="PGVECTOR_USER")
    pgvector_password: str = Field(default="", alias="PGVECTOR_PASSWORD")

    # Memgraph (optional session store)
    memgraph_uri: str = Field(default="bolt://memgraph:7687", alias="MEMGRAPH_URI")

    # Embeddings
    embed_model: str = Field(default="openai/text-embedding-3-small", alias="EMBED_MODEL")

    # Session lifecycle
    session_ttl_minutes: int = Field(default=60, alias="SESSION_TTL_MINUTES")

    # LLM
    llm_model: str = Field(default="moonshotai/kimi-k2", alias="LLM_MODEL")
    llm_provider: str = Field(default="openrouter", alias="LLM_PROVIDER")
    llm_temperature: float = Field(default=0.0, alias="LLM_TEMPERATURE")
    llm_max_tokens: int = Field(default=2048, alias="LLM_MAX_TOKENS")
    openrouter_api_key: str = Field(default="", alias="OPENROUTER_API_KEY")
    extraction_model: str = Field(default="moonshotai/kimi-k2", alias="EXTRACTION_MODEL")

    # Chunking
    chunk_size: int = Field(default=1200, alias="CHUNK_SIZE")
    chunk_overlap: int = Field(default=100, alias="CHUNK_OVERLAP")

    # Entity extraction
    gliner_model: str = Field(default="fastino/gliner2-multi-v1", alias="GLINER_MODEL")
    gliner_threshold: float = Field(default=0.5, alias="GLINER_THRESHOLD")
    relation_extraction_mode: Literal["gliner2", "llm_assisted", "off"] = Field(
        default="gliner2",
        alias="RELATION_EXTRACTION_MODE",
    )
    retrieval_mode: Literal["deterministic", "agentic"] = Field(
        default="deterministic",
        alias="RETRIEVAL_MODE",
    )
    ontology_file: str = Field(default="", alias="GRAPHKNOWS_ONTOLOGY_FILE")

    # Summarisation
    summarization_model: str = Field(default="moonshotai/kimi-k2", alias="SUMMARISATION_MODEL")

    @model_validator(mode="after")
    def _populate_from_repo_env(self) -> GraphKnowsSettings:
        """Back-fill credentials from the repo-level .env when defaults are in use."""
        env = _load_repo_env()

        if self.neo4j_password == "testpassword123" and env.get("NEO4J_PASSWORD"):
            self.neo4j_password = env["NEO4J_PASSWORD"]

        if not self.pgvector_password:
            self.pgvector_password = env.get("PGVECTOR_PASSWORD", "")
        password = self.pgvector_password

        if self.pgvector_dsn == _MASKED_PGVECTOR_DSN and password:
            host = env.get("PGVECTOR_HOST", self.pgvector_host)
            port = env.get("PGVECTOR_PORT", str(self.pgvector_port))
            db = env.get("PGVECTOR_DB", self.pgvector_db)
            user = env.get("PGVECTOR_USER", self.pgvector_user)
            self.pgvector_dsn = f"postgresql://{user}:{password}@{host}:{port}/{db}"

        if self.openrouter_api_key:
            self.openrouter_api_key = _clean_env_secret(self.openrouter_api_key)
        else:
            self.openrouter_api_key = _clean_env_secret(
                env.get("OPENROUTER_API_KEY") or env.get("OPEN_ROUTER_API")
            )

        # Push credentials into os.environ so embedder.py (and any other code that
        # reads os.environ directly) picks them up without requiring its own dotenv load.
        # embedder._openrouter_api_key() reads os.environ first, then falls back to its
        # own repo-env scan — this push makes the settings the authoritative source.
        if self.openrouter_api_key and not os.environ.get("OPENROUTER_API_KEY"):
            os.environ["OPENROUTER_API_KEY"] = self.openrouter_api_key
        if self.openrouter_api_key and not os.environ.get("OPEN_ROUTER_API"):
            os.environ["OPEN_ROUTER_API"] = self.openrouter_api_key

        return self

    def check_production_secrets(self) -> None:
        """Raise if running in production with default credentials.

        Raises:
            ValueError: When production env is detected with default passwords.
        """
        if self.graphknows_env != "production":
            return
        if self.neo4j_password == "testpassword123":
            raise ValueError("NEO4J_PASSWORD is still the development default.")
        if "***" in self.pgvector_dsn:
            raise ValueError("PGVECTOR_DSN contains the masked development default.")


from graphknows.config.health import check_backends, get_version  # noqa: F401, E402

"""Base service settings shared by all Graphknows microservices.

Each service subclasses BaseServiceSettings and adds service-specific fields.

Example:
    from graphknows.shared.config import BaseServiceSettings

    class GraphRAGSettings(BaseServiceSettings):
        graphrag_port: int = 8001
"""
from __future__ import annotations

from pathlib import Path

from dotenv import dotenv_values
from pydantic import Field, model_validator
from pydantic_settings import BaseSettings

_MASKED_PGVECTOR_DSN = "postgresql://postgres:***@172.18.0.1:5432/vectordb"


def _find_repo_env() -> Path | None:
    """Locate the nearest repo-level .env file by walking upward from this module."""
    for parent in Path(__file__).resolve().parents:
        candidate = parent / ".env"
        if candidate.exists():
            return candidate
    return None


def _load_repo_env() -> dict[str, str]:
    """Load repo-local .env values when present."""
    env_file = _find_repo_env()
    if env_file is None:
        return {}
    return {k: v for k, v in dotenv_values(str(env_file)).items() if isinstance(v, str)}


class BaseServiceSettings(BaseSettings):
    """Shared database and LLM connection settings.

    Attributes:
        neo4j_uri: Bolt URI for the persistent Neo4j KG.
        neo4j_user: Neo4j username.
        neo4j_password: Neo4j password.
        neo4j_database: Neo4j database name.
        pgvector_dsn: PostgreSQL DSN for the pgvector store.
        llm_model: LLM model name for synthesis.
        llm_provider: LLM provider name (openrouter or groq).
        groq_api_key: Groq API key (kept for backward compat).
        openrouter_api_key: OpenRouter API key.
        extraction_model: Model used for relation extraction (defaults to llm_model).
        embed_model: HuggingFace sentence-transformer model name.
        graphknows_env: Environment tag; set to "production" for stricter checks.
    """

    neo4j_uri: str = Field(default="bolt://172.18.0.1:7687", alias="NEO4J_URI")
    neo4j_user: str = Field(default="neo4j", alias="NEO4J_USER")
    neo4j_password: str = Field(default="testpassword123", alias="NEO4J_PASSWORD")
    neo4j_database: str = Field(default="neo4j", alias="NEO4J_DATABASE")
    pgvector_dsn: str = Field(default=_MASKED_PGVECTOR_DSN, alias="PGVECTOR_DSN")
    pgvector_host: str = Field(default="172.18.0.1", alias="PGVECTOR_HOST")
    pgvector_port: int = Field(default=5432, alias="PGVECTOR_PORT")
    pgvector_db: str = Field(default="vectordb", alias="PGVECTOR_DB")
    pgvector_user: str = Field(default="postgres", alias="PGVECTOR_USER")
    pgvector_password: str = Field(default="", alias="PGVECTOR_PASSWORD")
    llm_model: str = Field(default="moonshotai/kimi-k2", alias="LLM_MODEL")
    llm_provider: str = Field(default="openrouter", alias="LLM_PROVIDER")
    groq_api_key: str = Field(default="", alias="GROQ_API_KEY")
    openrouter_api_key: str = Field(default="", alias="OPENROUTER_API_KEY")
    extraction_model: str = Field(default="moonshotai/kimi-k2", alias="EXTRACTION_MODEL")
    embed_model: str = Field(default="BAAI/bge-small-en-v1.5", alias="EMBED_MODEL")
    graphknows_env: str = Field(default="development", alias="GRAPHKNOWS_ENV")

    @model_validator(mode="after")
    def _populate_sensitive_fields_from_repo_env(self) -> "BaseServiceSettings":
        env = _load_repo_env()

        if self.neo4j_password == "testpassword123" and env.get("NEO4J_PASSWORD"):
            self.neo4j_password = env["NEO4J_PASSWORD"]

        password = self.pgvector_password or env.get("PGVECTOR_PASSWORD", "")
        if not self.pgvector_password and password:
            self.pgvector_password = password

        if self.pgvector_dsn == _MASKED_PGVECTOR_DSN and password:
            host = env.get("PGVECTOR_HOST", self.pgvector_host)
            port = env.get("PGVECTOR_PORT", str(self.pgvector_port))
            db = env.get("PGVECTOR_DB", self.pgvector_db)
            user = env.get("PGVECTOR_USER", self.pgvector_user)
            self.pgvector_dsn = f"postgresql://{user}:{password}@{host}:{port}/{db}"

        if not self.openrouter_api_key:
            self.openrouter_api_key = (
                env.get("OPENROUTER_API_KEY")
                or env.get("OPEN_ROUTER_API", "")
            )

        return self

    def check_production_secrets(self) -> None:
        """Raise ValueError if running in production with default credentials.

        Raises:
            ValueError: If production env is detected with default passwords.
        """
        if self.graphknows_env != "production":
            return
        if self.neo4j_password == "testpassword123":
            raise ValueError("NEO4J_PASSWORD is still the development default.")
        if "pgvectorpass123" in self.pgvector_dsn:
            raise ValueError("PGVECTOR_DSN contains the development default password.")

    model_config = {"populate_by_name": True}

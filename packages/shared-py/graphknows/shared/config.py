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

_ENV_FILE = Path(__file__).resolve().parents[4] / ".env"
_MASKED_PGVECTOR_DSN = "postgresql://postgres:***@172.18.0.1:5432/vectordb"


def _load_repo_env() -> dict[str, str]:
    """Load repo-local .env values when present."""
    if not _ENV_FILE.exists():
        return {}
    return {k: v for k, v in dotenv_values(str(_ENV_FILE)).items() if isinstance(v, str)}


class BaseServiceSettings(BaseSettings):
    """Shared database and LLM connection settings.

    Attributes:
        neo4j_uri: Bolt URI for the persistent Neo4j KG.
        neo4j_user: Neo4j username.
        neo4j_password: Neo4j password.
        neo4j_database: Neo4j database name.
        pgvector_dsn: PostgreSQL DSN for the pgvector store.
        llm_model: LLM model name for synthesis.
        groq_api_key: Groq API key (empty string disables LLM tools).
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
    llm_model: str = Field(default="llama-3.3-70b-versatile", alias="LLM_MODEL")
    groq_api_key: str = Field(default="", alias="GROQ_API_KEY")
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

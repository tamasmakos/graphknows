"""Base service settings shared by all Graphknows microservices.

Each service subclasses BaseServiceSettings and adds service-specific fields.

Example:
    from graphknows.shared.config import BaseServiceSettings

    class GraphRAGSettings(BaseServiceSettings):
        graphrag_port: int = 8001
"""
from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings


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
    pgvector_dsn: str = Field(
        default="postgresql://postgres:pgvectorpass123@172.18.0.1:5432/vectordb",
        alias="PGVECTOR_DSN",
    )
    llm_model: str = Field(default="llama-3.3-70b-versatile", alias="LLM_MODEL")
    groq_api_key: str = Field(default="", alias="GROQ_API_KEY")
    embed_model: str = Field(default="BAAI/bge-small-en-v1.5", alias="EMBED_MODEL")
    graphknows_env: str = Field(default="development", alias="GRAPHKNOWS_ENV")

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

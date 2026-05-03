from typing import Optional
from pydantic_settings import BaseSettings


class AppSettings(BaseSettings):
    # Neo4j
    neo4j_uri: str = "bolt://neo4j:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "changeme"
    neo4j_database: str = "neo4j"

    # pgvector
    pgvector_host: str = "pgvector"
    pgvector_port: int = 5432
    pgvector_db: str = "vectordb"
    pgvector_user: str = "postgres"
    pgvector_password: str = "changeme"

    @property
    def pgvector_dsn(self) -> str:
        return (
            f"postgresql://{self.pgvector_user}:{self.pgvector_password}"
            f"@{self.pgvector_host}:{self.pgvector_port}/{self.pgvector_db}"
        )

    # LLM
    llm_provider: str = "groq"
    llm_model: str = "llama-3.3-70b-versatile"
    llm_groq_api_key: Optional[str] = None
    llm_openai_api_key: Optional[str] = None

    # Service-level model overrides
    graphrag_keywords_model: str = "llama-3.1-8b-instant"
    graphrag_chat_model: str = "llama-3.3-70b-versatile"
    graphrag_embedding_model: str = "BAAI/bge-small-en-v1.5"

    # Langfuse
    langfuse_public_key: Optional[str] = None
    langfuse_secret_key: Optional[str] = None
    langfuse_host: str = "http://localhost:3000"

    class Config:
        env_file = ".env"
        extra = "ignore"

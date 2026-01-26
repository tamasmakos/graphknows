from pydantic_settings import BaseSettings
from typing import Optional

class AppSettings(BaseSettings):
    # Define all env vars here with types
    falkordb_host: str = "falkordb"
    falkordb_port: int = 6379
    
    # Postgres
    postgres_host: str = "pgvector"
    postgres_port: int = 5432
    postgres_db: str = "graphknows"
    postgres_user: str = "postgres"
    postgres_password: str = "password"
    postgres_table: str = "hybrid_embeddings"
    postgres_enabled: bool = True

    # Models
    groq_model: str = "llama-3.3-70b-versatile"
    keywords_model: str = "llama-3.1-8b-instant"
    chat_model: str = "llama-3.1-8b-instant"

    openai_api_key: Optional[str] = None
    
    class Config:
        env_file = ".env"
        extra = "ignore"

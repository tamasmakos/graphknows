from pydantic_settings import BaseSettings
from typing import Optional

class PipelineSettings(BaseSettings):
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

    openai_api_key: Optional[str] = None
    # Processing Settings
    input_dir: str = "/app/input"
    output_dir: str = "/app/output"
    file_pattern: str = "*.csv"
    chunk_size: int = 512
    chunk_overlap: int = 20
    
    # GLiNER / Extraction Settings
    gliner_labels: list[str] = ["Person", "Organization", "Location", "Event", "Date", "Award", "Competitions", "Teams", "Concept"]
    extraction_backend: str = "spacy" # Options: "gliner", "spacy"
    spacy_model: str = "en_core_web_lg"

    # Incremental Mode Settings
    speech_limit: int = 10

    max_documents: int = 20
    
    class Config:
        env_file = ".env"
        extra = "ignore"

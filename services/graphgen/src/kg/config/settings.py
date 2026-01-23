from pydantic_settings import BaseSettings
from typing import Optional

class PipelineSettings(BaseSettings):
    # Define all env vars here with types
    falkordb_host: str = "falkordb"
    falkordb_port: int = 6379
    openai_api_key: Optional[str] = None
    # Processing Settings
    input_dir: str = "/app/input"
    output_dir: str = "/app/output"
    file_pattern: str = "*.csv"
    chunk_size: int = 512
    chunk_overlap: int = 20
    
    # GLiNER / Extraction Settings
    gliner_labels: list[str] = ["Person", "Organization", "Location", "Event", "Date", "Award", "Competitions", "Teams", "Concept"]
    extraction_backend: str = "gliner" # Options: "gliner", "spacy"
    spacy_model: str = "en_core_web_lg"

    # Incremental Mode Settings
    speech_limit: int = 10

    max_documents: int = 20
    
    class Config:
        env_file = ".env"
        extra = "ignore"

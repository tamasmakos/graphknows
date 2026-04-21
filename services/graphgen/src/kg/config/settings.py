"""
Configuration Management.

This module rationalizes the setup by separating:
1. Infrastructure & Integration (External connections, handled via .env)
2. Application Logic (Internal tuning, handled via defaults here)

Usage:
- Use .env for: Hostnames, Ports, API Keys, Model Selection.
- Edit this file for: Chunk sizes, Extraction rules, Thresholds.
"""

from typing import List, Optional
from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class InfrastructureSettings(BaseSettings):
    """
    External Integration Settings.
    Crucial for connecting services. Managed via .env / docker-compose.
    """

    # --- Neo4j ---
    neo4j_uri: str = Field("bolt://localhost:7687", alias="NEO4J_URI")
    neo4j_user: str = Field("neo4j", alias="NEO4J_USER")
    neo4j_password: str = Field("changeme", alias="NEO4J_PASSWORD")
    neo4j_database: str = Field("neo4j", alias="NEO4J_DATABASE")

    # --- API Keys ---
    groq_api_key: Optional[SecretStr] = Field(None, alias="GROQ_API_KEY")
    openai_api_key: Optional[SecretStr] = Field(None, alias="OPENAI_API_KEY")

    # --- Filesystem (Docker Volumes) ---
    input_dir: str = Field("/app/input", alias="INPUT_DIR")
    output_dir: str = Field("/app/output", alias="OUTPUT_DIR")


class LLMSettings(BaseSettings):
    """
    Model Configuration.
    Defaults are set here but can be overridden via .env for experimentation.
    """

    base_model: str = Field("llama-3.1-8b-instant", alias="GROQ_MODEL")
    extraction_model: str = Field(
        "meta-llama/llama-4-scout-17b-16e-instruct", alias="EXTRACTION_MODEL"
    )
    summarization_model: str = Field("llama-3.1-8b-instant", alias="SUMMARISATION_MODEL")

    temperature: float = 0.0
    max_retries: int = 3


class ExtractionSettings(BaseSettings):
    """
    Internal Extraction Logic.
    Tuned for the specific data domain. Not typically in .env.
    """

    # Text Splitting
    chunk_size: int = 1200
    chunk_overlap: int = 100

    # Extraction Backend preference
    backend: str = "gliner"  # options: "gliner", "llm"

    # GLiNER Configuration
    gliner_model: str = "urchade/gliner_medium-v2.1"
    gliner_threshold: float = 0.5
    gliner_labels: List[str] = [
        "Person",
        "Organization",
        "Location",
        "Event",
        "Date",
        "Concept",
        "Product",
        "Skill",
        "Award",
        "Competitions",
        "Teams",
    ]

    # Spacy Configuration
    spacy_model: str = "en_core_web_lg"

    # Performance
    max_concurrent_chunks: int = 8

    # Legacy/Misc
    file_pattern: str = "*.csv"
    speech_limit: int = 10
    max_documents: int = 20


class ProcessingSettings(BaseSettings):
    """
    Internal Graph Processing Logic.
    """

    # Graph Pruning
    enable_pruning: bool = True
    pruning_threshold: float = 0.01
    prune_isolated_nodes: bool = True
    min_component_size: int = 3

    # Similarity & Resolution
    similarity_threshold: float = 0.95

    # Community Detection
    resolution: float = 1.0


class EmbeddingSettings(BaseSettings):
    """
    Embedding Model Configuration.
    """

    model_name: str = Field("BAAI/bge-small-en-v1.5", alias="EMBEDDING_MODEL")
    batch_size: int = Field(32, alias="EMBEDDING_BATCH_SIZE")
    device: str = Field("auto", alias="EMBEDDING_DEVICE")
    cache_folder: Optional[str] = Field(None, alias="EMBEDDING_CACHE_FOLDER")


class PipelineSettings(BaseSettings):
    """
    Master Configuration Object.
    Aggregates all specific settings groups.
    """

    infra: InfrastructureSettings = Field(default_factory=InfrastructureSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    extraction: ExtractionSettings = Field(default_factory=ExtractionSettings)
    processing: ProcessingSettings = Field(default_factory=ProcessingSettings)
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)

    # Global/Runtime flags
    debug: bool = False

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

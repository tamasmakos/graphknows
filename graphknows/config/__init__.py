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

from enum import StrEnum

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class MemoryMode(StrEnum):
    """Single switch that governs both extraction and retrieval.

    - ``llm_free``: GLiNER entities + spaCy relations + deterministic retrieval.
      No network LLM calls on the ingestion/retrieval hot paths.
    - ``llm_assisted``: additionally runs DSPy relation extraction, topic
      summarisation, and the agentic retriever.
    """

    llm_free = "llm_free"
    llm_assisted = "llm_assisted"


class GraphKnowsSettings(BaseSettings):
    """Runtime configuration for graphknows.

    Every setting maps to a ``GRAPHKNOWS_``-prefixed environment variable and can
    also be read from a ``.env`` file in the current working directory. Standard
    precedence applies: constructor args > environment > ``.env``.

    LLM configuration is provider-neutral: ``llm_model`` is a litellm model
    string (e.g. ``openrouter/deepseek/deepseek-v4-flash`` or ``openai/gpt-4o-mini``)
    and the provider base URL is inferred from that prefix unless ``llm_api_base``
    is set. Embeddings default to a local sentence-transformers model (no API key);
    set ``embed_api_base`` to use a remote OpenAI-compatible endpoint instead.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    graphknows_env: str = Field(default="development", alias="GRAPHKNOWS_ENV")
    graphknows_port: int = Field(default=8025, alias="GRAPHKNOWS_PORT")

    # Single mode switch driving both extraction and retrieval.
    mode: MemoryMode = Field(default=MemoryMode.llm_free, alias="GRAPHKNOWS_MODE")

    # ArcadeDB
    arcadedb_url: str = Field(default="http://localhost:2480", alias="GRAPHKNOWS_ARCADEDB_URL")
    arcadedb_user: str = Field(default="root", alias="GRAPHKNOWS_ARCADEDB_USER")
    arcadedb_password: str = Field(default="changeme", alias="GRAPHKNOWS_ARCADEDB_PASSWORD")

    # LLM (provider-neutral; litellm model strings)
    llm_model: str = Field(
        default="openrouter/deepseek/deepseek-v4-flash", alias="GRAPHKNOWS_LLM_MODEL"
    )
    llm_api_key: str = Field(default="", alias="GRAPHKNOWS_LLM_API_KEY")
    llm_api_base: str = Field(default="", alias="GRAPHKNOWS_LLM_API_BASE")
    llm_temperature: float = Field(default=0.0, alias="GRAPHKNOWS_LLM_TEMPERATURE")
    llm_max_tokens: int = Field(default=2048, alias="GRAPHKNOWS_LLM_MAX_TOKENS")
    extraction_model: str = Field(default="", alias="GRAPHKNOWS_EXTRACTION_MODEL")
    summarization_model: str = Field(default="", alias="GRAPHKNOWS_SUMMARIZATION_MODEL")

    # Embeddings (local sentence-transformers by default; no API key needed)
    embed_model: str = Field(default="BAAI/bge-small-en-v1.5", alias="GRAPHKNOWS_EMBED_MODEL")
    embed_api_base: str = Field(default="", alias="GRAPHKNOWS_EMBED_API_BASE")
    embed_api_key: str = Field(default="", alias="GRAPHKNOWS_EMBED_API_KEY")

    # Chunking
    chunk_size: int = Field(default=1200, alias="GRAPHKNOWS_CHUNK_SIZE")
    chunk_overlap: int = Field(default=100, alias="GRAPHKNOWS_CHUNK_OVERLAP")

    # Entity extraction
    gliner_model: str = Field(default="fastino/gliner2-multi-v1", alias="GRAPHKNOWS_GLINER_MODEL")
    gliner_threshold: float = Field(default=0.5, alias="GRAPHKNOWS_GLINER_THRESHOLD")
    spacy_model: str = Field(default="en_core_web_lg", alias="GRAPHKNOWS_SPACY_MODEL")
    ontology_file: str = Field(default="", alias="GRAPHKNOWS_ONTOLOGY_FILE")

    # Retrieval channels — modular ingest+retrieve signals (see memory/channels/).
    # Frames (deterministic FrameNet layer) are opt-in until validated at scale.
    enable_frames: bool = Field(default=False, alias="GRAPHKNOWS_ENABLE_FRAMES")

    # Session lifecycle
    session_ttl_minutes: int = Field(default=60, alias="GRAPHKNOWS_SESSION_TTL_MINUTES")

    def check_production_secrets(self) -> None:
        """Raise if running in production with unsafe defaults.

        Raises:
            ValueError: When production env is detected with the default password,
                or with llm_assisted mode selected but no LLM API key configured.
        """
        if self.graphknows_env != "production":
            return
        if self.arcadedb_password == "changeme":
            raise ValueError("GRAPHKNOWS_ARCADEDB_PASSWORD is still the development default.")
        if self.mode is MemoryMode.llm_assisted and not self.llm_api_key:
            raise ValueError("llm_assisted mode in production requires GRAPHKNOWS_LLM_API_KEY.")


from graphknows.config.health import get_version  # noqa: F401, E402

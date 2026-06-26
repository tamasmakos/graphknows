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
from enum import StrEnum
from pathlib import Path

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict

from graphknows.config.env_loader import load_env_file


class MemoryMode(StrEnum):
    """Single switch that governs both extraction and retrieval.

    - ``llm_free``: GLiNER entities + spaCy relations + deterministic retrieval.
      No network LLM calls on the ingestion/retrieval hot paths.
    - ``llm_assisted``: additionally runs DSPy relation extraction, topic
      summarisation, and the agentic retriever.
    """

    llm_free = "llm_free"
    llm_assisted = "llm_assisted"


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
    ``ARCADEDB_URL`` and ``arcadedb_url`` work as sources.

    Attributes:
        graphknows_env: Runtime environment tag (``"development"`` | ``"production"``).
        graphknows_port: HTTP port for the REST API.
        arcadedb_url: HTTP base URL for ArcadeDB.
        arcadedb_user: ArcadeDB server user.
        arcadedb_password: ArcadeDB server password.
        embed_model: OpenRouter embedding model name (e.g. openai/text-embedding-3-small).
        session_ttl_minutes: Minutes of session inactivity before auto-merge.
        llm_model: LLM model name for synthesis and extraction.
        llm_provider: LLM provider (``"openrouter"``).
        openrouter_api_key: OpenRouter API key.
        extraction_model: Model used for relation extraction.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        # .env file takes priority over container/shell env vars so that the
        # repo .env is always the source of truth for model names and API keys.
        return init_settings, dotenv_settings, env_settings, file_secret_settings

    graphknows_env: str = Field(default="development", alias="GRAPHKNOWS_ENV")
    graphknows_port: int = Field(default=8025, alias="GRAPHKNOWS_PORT")

    # ArcadeDB
    arcadedb_url: str = Field(default="http://172.18.0.1:2480", alias="ARCADEDB_URL")
    arcadedb_user: str = Field(default="root", alias="ARCADEDB_USER")
    arcadedb_password: str = Field(default="changeme", alias="ARCADEDB_PASSWORD")

    # Embeddings
    embed_model: str = Field(default="openai/text-embedding-3-small", alias="EMBED_MODEL")

    # Session lifecycle
    session_ttl_minutes: int = Field(default=60, alias="SESSION_TTL_MINUTES")

    # LLM
    llm_model: str = Field(default="", alias="LLM_MODEL")
    llm_provider: str = Field(default="openrouter", alias="LLM_PROVIDER")
    llm_temperature: float = Field(default=0.0, alias="LLM_TEMPERATURE")
    llm_max_tokens: int = Field(default=2048, alias="LLM_MAX_TOKENS")
    openrouter_api_key: str = Field(default="", alias="OPENROUTER_API_KEY")
    extraction_model: str = Field(default="", alias="EXTRACTION_MODEL")

    # Chunking
    chunk_size: int = Field(default=1200, alias="CHUNK_SIZE")
    chunk_overlap: int = Field(default=100, alias="CHUNK_OVERLAP")

    # Entity extraction
    gliner_model: str = Field(default="fastino/gliner2-multi-v1", alias="GLINER_MODEL")
    gliner_threshold: float = Field(default=0.5, alias="GLINER_THRESHOLD")
    # Single mode switch driving both extraction and retrieval.
    mode: MemoryMode = Field(default=MemoryMode.llm_free, alias="MODE")
    # spaCy dependency-pattern relations are LLM-free; toggle for development.
    spacy_relations: bool = Field(default=True, alias="SPACY_RELATIONS")
    # Score-aware fusion: weight applied to normalised cosine similarity as a
    # tiebreaker on top of rank-based RRF in the semantic collector. 0 disables
    # (pure RRF); small values break ties without overriding multi-source agreement.
    fusion_alpha: float = Field(default=0.005, alias="FUSION_ALPHA")
    ontology_file: str = Field(default="", alias="GRAPHKNOWS_ONTOLOGY_FILE")

    # Summarisation
    summarization_model: str = Field(default="", alias="SUMMARISATION_MODEL")

    @model_validator(mode="after")
    def _populate_from_repo_env(self) -> GraphKnowsSettings:
        """Back-fill credentials from the repo-level .env when defaults are in use."""
        env = _load_repo_env()

        if self.openrouter_api_key:
            self.openrouter_api_key = _clean_env_secret(self.openrouter_api_key)
        else:
            self.openrouter_api_key = _clean_env_secret(
                env.get("OPENROUTER_API_KEY") or env.get("OPEN_ROUTER_API")
            )

        # Push credentials and model names into os.environ so any code that reads
        # os.environ directly (embedder.py, generator.py, judge.py) gets .env values.
        if self.openrouter_api_key:
            os.environ["OPENROUTER_API_KEY"] = self.openrouter_api_key
            os.environ["OPEN_ROUTER_API"] = self.openrouter_api_key
        if self.llm_model:
            os.environ["LLM_MODEL"] = self.llm_model
        if self.extraction_model:
            os.environ["EXTRACTION_MODEL"] = self.extraction_model
        if self.summarization_model:
            os.environ["SUMMARISATION_MODEL"] = self.summarization_model
        if self.embed_model:
            os.environ["EMBED_MODEL"] = self.embed_model

        # Back-compat: derive the single mode from legacy env vars when MODE is
        # not set, so callers still exporting RELATION_EXTRACTION_MODE/RETRIEVAL_MODE
        # (e.g. the evaluation harness) keep working.
        if "MODE" not in os.environ and "MODE" not in env:
            legacy_rel = os.environ.get("RELATION_EXTRACTION_MODE") or env.get(
                "RELATION_EXTRACTION_MODE"
            )
            legacy_ret = os.environ.get("RETRIEVAL_MODE") or env.get("RETRIEVAL_MODE")
            if legacy_rel == "llm_assisted" or legacy_ret == "agentic":
                self.mode = MemoryMode.llm_assisted

        return self

    @property
    def relation_extraction_mode(self) -> str:
        """Legacy relation-extraction mode string derived from :attr:`mode`."""
        return "llm_assisted" if self.mode is MemoryMode.llm_assisted else "gliner2"

    @property
    def retrieval_mode(self) -> str:
        """Legacy retrieval mode string derived from :attr:`mode`."""
        return "agentic" if self.mode is MemoryMode.llm_assisted else "deterministic"

    def check_production_secrets(self) -> None:
        """Raise if running in production with default credentials.

        Raises:
            ValueError: When production env is detected with default passwords.
        """
        if self.graphknows_env != "production":
            return
        if self.arcadedb_password == "changeme":
            raise ValueError("ARCADEDB_PASSWORD is still the development default.")


from graphknows.config.health import check_backends, get_version  # noqa: F401, E402

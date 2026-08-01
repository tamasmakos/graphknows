"""graphknows runtime configuration.

Importing this module has no side-effects (no FastAPI, no store connections).
Any agent or library that wants only settings can do::

    from graphknows.settings import GraphKnowsSettings
    cfg = GraphKnowsSettings()

Example:
    from graphknows.settings import GraphKnowsSettings

    settings = GraphKnowsSettings()
    settings.check_production_secrets()
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from graphknows.exceptions import ConfigurationError


class MemoryMode(StrEnum):
    """Named preset over the orthogonal capability knobs.

    The mode is *only* a preset: every capability it implies (topics, DSPy
    relations) is independently overridable, and ontology injection plus the
    FrameNet layer are mode-independent. See
    :data:`_PRESETS` for the expansion and ``docs/modes.md`` for the rationale.

    - ``llm_free``: no network LLM call on the ingestion or retrieval hot path.
    - ``llm_assisted``: adds DSPy relation extraction, LLM topic summaries, and
      LLM-backed capabilities.
    """

    llm_free = "llm_free"
    llm_assisted = "llm_assisted"


class TopicMode(StrEnum):
    """How TOPIC nodes are titled and summarised at flush time.

    - ``none``: the topic layer is not built.
    - ``extractive``: deterministic — PageRank-titled, centroid-nearest
      summary sentences. No LLM, so it is valid in ``llm_free``.
    - ``llm``: prose summaries generated over the community's passages.
    """

    none = "none"
    extractive = "extractive"
    llm = "llm"


@dataclass(frozen=True)
class _Preset:
    """The capability knobs a :class:`MemoryMode` expands into."""

    topics: TopicMode
    dspy_relations: bool


_PRESETS: dict[MemoryMode, _Preset] = {
    MemoryMode.llm_free: _Preset(topics=TopicMode.none, dspy_relations=False),
    MemoryMode.llm_assisted: _Preset(topics=TopicMode.llm, dspy_relations=True),
}


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

    # Named preset over the capability knobs below. Every knob it implies can be
    # overridden independently; read the resolved values through the
    # ``topic_mode`` / ``dspy_relations`` properties, never off these raw fields
    # (``None`` means "take the preset's value").
    mode: MemoryMode = Field(default=MemoryMode.llm_free, alias="GRAPHKNOWS_MODE")
    topics: TopicMode | None = Field(default=None, alias="GRAPHKNOWS_TOPICS")
    enable_dspy_relations: bool | None = Field(default=None, alias="GRAPHKNOWS_DSPY_RELATIONS")

    # ArcadeDB
    arcadedb_url: str = Field(default="http://localhost:2480", alias="GRAPHKNOWS_ARCADEDB_URL")
    arcadedb_user: str = Field(default="root", alias="GRAPHKNOWS_ARCADEDB_USER")
    arcadedb_password: str = Field(default="changeme", alias="GRAPHKNOWS_ARCADEDB_PASSWORD")

    # Multi-tenant isolation boundary. Each namespace maps to its own physical
    # ArcadeDB database pair (stm_<ns>/ltm_<ns>); "" keeps the original stm/ltm
    # databases. Used as the default when an MCP call omits an explicit namespace.
    namespace: str = Field(default="", alias="GRAPHKNOWS_NAMESPACE")

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

    # Embeddings (local sentence-transformers by default; no API key needed).
    #
    # This default MUST match docker-compose.yaml and .env.example: ArcadeDB
    # vector indexes are dimension-typed, so a library user on one default and a
    # container user on another silently build incompatible indexes. bge-small is
    # the shipped choice — 384-dim, ~130MB, baked into the image, no gated repo
    # and no trust_remote_code. Larger embedders (e.g. the 2560-dim 4B
    # zeroentropy/zembed-1-embedding) are opt-in via GRAPHKNOWS_EMBED_MODEL and
    # download on first use; changing this on a populated namespace requires a
    # re-index.
    embed_model: str = Field(default="BAAI/bge-small-en-v1.5", alias="GRAPHKNOWS_EMBED_MODEL")
    embed_api_base: str = Field(default="", alias="GRAPHKNOWS_EMBED_API_BASE")
    embed_api_key: str = Field(default="", alias="GRAPHKNOWS_EMBED_API_KEY")
    # Local-embedding device: "auto" (CUDA if present, else CPU), "cuda", or "cpu".
    embed_device: str = Field(default="auto", alias="GRAPHKNOWS_EMBED_DEVICE")
    # Local-embedding weight quantization (bitsandbytes, CUDA only): "", "8bit",
    # or "4bit". Lets a large embedder (e.g. the 4B zembed-1) fit a small GPU;
    # ignored on CPU, where the model loads at full bf16 precision.
    embed_quantization: str = Field(default="", alias="GRAPHKNOWS_EMBED_QUANTIZATION")
    # Output dimensionality for remote embeddings (0 = model default). Matryoshka
    # models (e.g. text-embedding-3-*) truncate server-side with minimal quality
    # loss. Vector memory in the store scales linearly with this — the system is
    # sized around 384-dim vectors, so keep remote embeddings at parity unless
    # you deliberately re-size the store.
    embed_dimensions: int = Field(default=0, alias="GRAPHKNOWS_EMBED_DIMENSIONS")

    # Entity extraction. The relex model itself is set with GRAPHKNOWS_RELEX_MODEL
    # (see ingestion/extraction/entities/gliner_model.py).
    spacy_model: str = Field(default="en_core_web_lg", alias="GRAPHKNOWS_SPACY_MODEL")

    # Relation verifier — semantic gate that drops relex false positives before
    # they merge into the graph (co-occurring spans relex confidently pairs but
    # that express no real relation). ON by default; disable with
    # GRAPHKNOWS_RELATION_VERIFIER=false. Threshold trades precision for recall:
    # the model card recommends 0.55; we default to 0.50 to protect recall given
    # the KG's already-thin relation signal.
    relation_verifier: bool = Field(default=True, alias="GRAPHKNOWS_RELATION_VERIFIER")
    relation_verifier_threshold: float = Field(
        default=0.5, alias="GRAPHKNOWS_RELATION_VERIFIER_THRESHOLD"
    )
    relation_verifier_model: str = Field(
        default="oneryalcin/gliner2-relation-verifier",
        alias="GRAPHKNOWS_RELATION_VERIFIER_MODEL",
    )

    # Ontology injection — an RDF/OWL file OR a folder of them. Mode-independent:
    # when set, ontology class labels steer entity extraction and property labels
    # steer relation extraction in BOTH modes, and the terms are persisted as
    # ONTOLOGY_CLASS/ONTOLOGY_PROPERTY nodes. Read via ``ontology_source``.
    ontology: str = Field(default="", alias="GRAPHKNOWS_ONTOLOGY")
    # Deprecated alias kept so existing GRAPHKNOWS_ONTOLOGY_FILE setups (and the
    # eval CLIs) keep working; ``ontology`` wins when both are set.
    ontology_file: str = Field(default="", alias="GRAPHKNOWS_ONTOLOGY_FILE")

    # Retrieval channels — modular ingest+retrieve signals (see memory/channels/).
    # Frames (deterministic FrameNet layer) are ON by default; disable with
    # GRAPHKNOWS_ENABLE_FRAMES=false.
    enable_frames: bool = Field(default=True, alias="GRAPHKNOWS_ENABLE_FRAMES")

    @field_validator("topics", "enable_dspy_relations", mode="before")
    @classmethod
    def _empty_means_unset(cls, value: Any) -> Any:
        """Treat an empty env var as "not set", so it falls back to the preset.

        ``GRAPHKNOWS_TOPICS=`` is how docker-compose and shell wrappers spell an
        absent value; without this it reaches pydantic as ``""`` and fails enum
        validation, taking the whole process down at construction.
        """
        return None if isinstance(value, str) and not value.strip() else value

    @property
    def topic_mode(self) -> TopicMode:
        """Resolved topic mode — the explicit setting, else the mode's preset."""
        return self.topics if self.topics is not None else _PRESETS[self.mode].topics

    @property
    def dspy_relations(self) -> bool:
        """Whether DSPy relation extraction runs — explicit setting, else preset."""
        if self.enable_dspy_relations is not None:
            return self.enable_dspy_relations
        return _PRESETS[self.mode].dspy_relations

    @property
    def ontology_source(self) -> str:
        """The ontology file or folder to load, honouring the deprecated alias."""
        return self.ontology or self.ontology_file

    @property
    def uses_llm(self) -> bool:
        """Whether any resolved capability makes network LLM calls."""
        return self.dspy_relations or self.topic_mode is TopicMode.llm

    def check_production_secrets(self) -> None:
        """Raise if running in production with unsafe defaults.

        Runs automatically at construction (see :meth:`_validate_secrets`); also
        callable directly as the strict entry point.

        Raises:
            ConfigurationError: When production env is detected with the default
                password, or with an LLM-using capability enabled but no API key.
        """
        if self.graphknows_env != "production":
            return
        if self.arcadedb_password == "changeme":
            raise ConfigurationError(
                "GRAPHKNOWS_ARCADEDB_PASSWORD is still the development default."
            )
        # Keyed off the resolved capabilities, not the mode: llm_assisted with
        # every LLM capability disabled needs no key.
        if self.uses_llm and not self.llm_api_key:
            raise ConfigurationError(
                "LLM-backed capabilities (DSPy relations, LLM topics) "
                "in production require GRAPHKNOWS_LLM_API_KEY."
            )

    @model_validator(mode="after")
    def _validate_secrets(self) -> GraphKnowsSettings:
        """Fail fast at construction when production secrets are unsafe."""
        self.check_production_secrets()
        return self

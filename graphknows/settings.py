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
from pathlib import Path
from typing import Any

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from graphknows.exceptions import ConfigurationError

# The always-on default ontology, resolved off this file rather than imported
# from ``graphknows.ontology``: settings sits BELOW ontology in the layering
# (see CLAUDE.md), so the dependency may only point the other way.
_BUNDLED_ONTOLOGY = str(
    Path(__file__).resolve().parent / "ontology" / "assets" / "cco" / "cco.json"
)

# The relation vocabulary. This used to be a DIFFERENT file from the ontology
# above: schema.org typed entities well and supplied relation properties badly
# (selecting among its 945 object properties by whole-chunk cosine returned
# topically-close, relationally-irrelevant terms and produced ZERO relations
# over 60 real turns), so a hand-curated 25-property profile carried relations.
#
# The Common Core Ontologies serve both jobs, so the split is gone: 1401 classes
# to type entities, and 224 object properties of which 112 declare the domain
# and range that property selection keys on — including the 86 familial
# relations that are the conversational vocabulary the profile was invented to
# supply. Both names remain so that pointing GRAPHKNOWS_RELATION_ONTOLOGY at a
# separate curated vocabulary is still a supported override.
_BUNDLED_RELATION_PROFILE = _BUNDLED_ONTOLOGY

# The INPUT SKOS overlay: the everyday concepts CCO does not model as binary
# predicates (works for, plans to, likes, cares for), each carrying the
# altLabels that make it lexically reachable and domain/range in CCO's own class
# vocabulary. Data, not code — point this elsewhere to change the vocabulary.
_BUNDLED_OVERLAY = str(
    Path(__file__).resolve().parent / "ontology" / "assets" / "cco" / "conversational.json"
)


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

    # Multi-tenant isolation boundary. Each namespace maps to ONE physical
    # ArcadeDB database (``mem_<ns>``; "" -> ``mem``) — see storage/namespace.py.
    # Short- and long-term memory are not separate databases, they are the
    # ``state`` lifecycle (raw -> consolidated) on that one graph. Used as the
    # default when an MCP call omits an explicit namespace.
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

    # Ontology injection — ALWAYS ON, like the FrameNet layer. Ontology class
    # labels steer entity extraction and property labels steer relation
    # extraction in BOTH modes; classes are persisted as ONTOLOGY_CLASS nodes,
    # while properties are NOT persisted — domain/range validation reads the
    # in-memory OntologyTerm from the loader, so ONTOLOGY_PROPERTY vertices
    # would be write-only. Read via ``ontology_source``.
    #
    # The default is the bundled Common Core Ontologies digest: a pre-parsed
    # JSON of 1401 classes and 224 object properties that loads WITHOUT rdflib
    # (which ships only in the ``assisted`` extra), so a bare `pip install
    # graphknows` gets the same ontology the container does, and both memory
    # modes load it through one identical code path.
    # ``scripts/build_ontology_digest.py`` regenerates it from the TTL sources
    # under ``ontology/assets/cco-*``. GRAPHKNOWS_ONTOLOGY overrides it
    # with any other RDF/OWL/TTL file, digest, or folder of them — an override,
    # not an on/off switch: an EMPTY value means "unset" (which is how
    # docker-compose spells an absent variable,
    # ``GRAPHKNOWS_ONTOLOGY: "${GRAPHKNOWS_ONTOLOGY:-}"``) and still resolves to
    # the bundled digest. The fallback lives in ``ontology_source`` so the
    # deprecated alias below keeps its precedence.
    ontology: str = Field(default="", alias="GRAPHKNOWS_ONTOLOGY")
    # Deprecated alias kept so existing GRAPHKNOWS_ONTOLOGY_FILE setups (and the
    # eval CLIs) keep working; ``ontology`` wins when both are set.
    ontology_file: str = Field(default="", alias="GRAPHKNOWS_ONTOLOGY_FILE")

    # Cosine floor for the per-chunk ontology CLASS hints (properties are chosen
    # structurally by domain/range, not by this floor). Re-measured over 20 real
    # LoCoMo conv-30 turns against the bundled CCO digest with bge-small — the
    # schema.org numbers this replaces do not transfer, because CCO's
    # genus-differentia definitions sit in a tighter band: the whole 1401-class
    # space spans 0.168-0.683 with median 0.409 and p95 0.493, and the weakest
    # genuine per-turn top-1 is 0.497.
    #
    # 0.46 therefore sits above the ontology's average term — a hint has to beat
    # the mean to be offered — and below every turn's best match, so 0/20 turns
    # lose their top candidate. Raising it starves fast on this distribution:
    # 0.54 leaves 3/20 turns with no class hint at all.
    #
    # This bounds hint VOLUME, not quality: absolute cosines are not calibrated
    # enough to separate a right term from a wrong one (a shopping turn's top-3
    # is ``Grocery Store``/``Shop``/``Warehouse`` at 0.68-0.63, which is good,
    # but a thank-you turn tops out at ``Artifact History`` 0.545, which is not).
    # Re-census before retuning.
    ontology_hint_min_sim: float = Field(default=0.46, alias="GRAPHKNOWS_ONTOLOGY_HINT_MIN_SIM")

    # How many consecutive turns the extractor reads at once. A turn extracted
    # alone loses everything the turns before it established — "She said yes"
    # asserts nothing extractable unless the previous turn named her — so the
    # window prepends the recent speaker-attributed turns to the one being
    # ingested. Only the current turn is stored, embedded and retrieved; the
    # window widens what the EXTRACTOR sees, never what memory holds.
    #
    # The same triplet is then found by several overlapping windows, which is
    # intended and costs nothing: ``write_relation`` MERGEs on
    # (head, predicate, tail), appending each asserting chunk to the edge's
    # evidence list and keeping the max confidence and verifier score.
    #
    # 1 disables the window and is an exact replay of the pre-window behaviour,
    # which is what makes it the control arm of an A/B rather than an
    # approximation of one.
    extract_window: int = Field(default=5, ge=1, alias="GRAPHKNOWS_EXTRACT_WINDOW")
    # How many turns consecutive windows SHARE. A relation whose endpoints
    # straddle a window boundary is invisible to every window, so with no
    # overlap each boundary is a permanent blind spot. Measured at equal turn
    # coverage: 2 turns of overlap yields +35% distinct edges, and the duplicate
    # triples cost nothing because write_relation MERGEs on
    # (head, predicate, tail) and accumulates evidence.
    extract_overlap: int = Field(default=2, ge=0, alias="GRAPHKNOWS_EXTRACT_OVERLAP")

    # Label selection by LEXICAL evidence over the SKOS scheme rather than by
    # embedding the chunk against class definitions. Measured on conv-30: the
    # embedding path put 11% of tags in a chunk sharing even one word with it
    # ("Lost my job as a banker" -> Grocery Store), while lexical selection
    # tripled relation richness and 2-4x'd distinct predicates. Kept switchable
    # because it is a large behaviour change, not because the old path is a
    # supported alternative.
    lexical_labels: bool = Field(default=True, alias="GRAPHKNOWS_LEXICAL_LABELS")
    # SKOS overlay layered over the base ontology; empty disables it.
    ontology_overlay: str = Field(default="", alias="GRAPHKNOWS_ONTOLOGY_OVERLAY")

    # The relation vocabulary. Same override semantics as ``ontology`` (empty
    # means unset, not off) and read via ``relation_ontology_source``. Point it
    # at any digest/RDF file with domain- and range-typed properties; point it at
    # the same value as ``ontology`` to go back to one vocabulary for both jobs.
    relation_ontology: str = Field(default="", alias="GRAPHKNOWS_RELATION_ONTOLOGY")

    # Retrieval channels — modular ingest+retrieve signals (see memory/channels/).
    # Frames (deterministic FrameNet layer) are ON by default; disable with
    # GRAPHKNOWS_ENABLE_FRAMES=false.
    enable_frames: bool = Field(default=True, alias="GRAPHKNOWS_ENABLE_FRAMES")
    # The ontology retrieval channel is OPT-IN and defaults OFF: it has never
    # been measured on its own. It went live as a side effect of making ontology
    # INJECTION always-on (its `if settings.ontology_source:` gate could not be
    # false once that property always resolves), and the next full conv-30 run
    # fused six channels instead of five — semantic 1875, text 1588, entity 1498,
    # frame 1312, temporal 625, plus ontology 838 — and scored accuracy 0.766 ->
    # 0.722 with "NOT RETRIEVED" failures 8 -> 13. That run changed the
    # extraction layer too, so this is a suspect and not a verdict; the point of
    # the switch is to restore the measured channel set so the next A/B isolates
    # one change. Turn it on to measure it, and record the number here.
    enable_ontology_channel: bool = Field(default=False, alias="GRAPHKNOWS_ENABLE_ONTOLOGY_CHANNEL")

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
        """The ontology file or folder to load, honouring the deprecated alias.

        Never empty: an unset or empty ``GRAPHKNOWS_ONTOLOGY`` falls back to the
        bundled CCO digest. There is no "no ontology" state — a caller
        that hands ``load_ontology_index`` an empty string bypassed settings,
        and that function logs the fact.
        """
        return self.ontology.strip() or self.ontology_file.strip() or _BUNDLED_ONTOLOGY

    @property
    def overlay_source(self) -> str:
        """The SKOS overlay path — bundled unless overridden; "-" disables it."""
        value = self.ontology_overlay.strip()
        if value == "-":
            return ""
        return value or _BUNDLED_OVERLAY

    @property
    def relation_ontology_source(self) -> str:
        """The vocabulary relation properties are drawn from — never empty.

        Defaults to the same bundled digest as :attr:`ontology_source` — CCO
        supplies both vocabularies — but stays a separate knob so a deployment
        can point relations at a curated domain profile without replacing the
        class vocabulary that types entities.
        """
        return self.relation_ontology.strip() or _BUNDLED_RELATION_PROFILE

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

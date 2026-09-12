"""graphknows runtime configuration.

Importing this module has no side-effects (no FastAPI, no store connections).
Any agent or library that wants only settings can do::

    from graphknows.settings import get_settings
    cfg = get_settings()

Example:
    from graphknows.settings import get_settings

    settings = get_settings()
    settings.check_production_secrets()
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from enum import StrEnum
from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from graphknows.bounds import CALL_TIMEOUT_S
from graphknows.exceptions import ConfigurationError

# The always-on default ontology, resolved off this file rather than imported
# from ``graphknows.symbolic.ontology``: settings sits BELOW ontology in the layering
# (see CLAUDE.md), so the dependency may only point the other way.
_BUNDLED_ONTOLOGY = str(
    Path(__file__).resolve().parent / "symbolic" / "ontology" / "assets" / "cco" / "cco.json"
)

# The relation vocabulary. This used to be a DIFFERENT file from the ontology
# above: schema.org typed entities well and supplied relation properties badly
# (selecting among its 945 object properties by whole-chunk cosine returned
# topically-close, relationally-irrelevant terms and produced ZERO relations
# over 60 real turns), so a hand-curated 25-property profile carried relations.
#
# CCO still types entities (1437 classes, see ``ontology_source`` below) but no
# longer supplies the default relation vocabulary. The conv-30 personal-vs-cco
# A/B pre-registered in #135 (blocked in earlier attempts by an unrelated
# ingestion defect, since fixed) ran to completion in both arms and is written
# up in evaluation/audit/baseline/relation-vocabulary-ab.md: evidence recall and
# accuracy don't discriminate between the two (generation-bound saturation /
# noise), but predicate anchoring does — CCO leaves 48/356 REL edges (13.5%) as
# bare, unanchored ``lemma:`` predicates versus the personal profile's 9/356
# (2.5%), because CCO's enterprise/commercial-frame object properties don't
# cover a sizeable share of what personal, two-person conversation expresses.
# That is the evidence #137 asks for, so the personal profile (9 classes, 25
# object properties, every one declaring a domain and a range, sized to be
# offered WHOLE) is now the bundled default — entity typing and relation
# vocabulary are two separate knobs, and only the latter moves here.
_BUNDLED_PERSONAL_PROFILE = str(
    Path(__file__).resolve().parent
    / "symbolic"
    / "ontology"
    / "assets"
    / "personal"
    / "personal-profile.json"
)
_BUNDLED_RELATION_PROFILE = _BUNDLED_PERSONAL_PROFILE

# CCO's own object properties (264, each declaring domain and range), kept as
# the documented example you switch back to with:
#
#     GRAPHKNOWS_RELATION_ONTOLOGY=cco
#
# See the A/B report above for why it is no longer the default.

# Bare tokens accepted by GRAPHKNOWS_RELATION_ONTOLOGY in place of a path, so an
# A/B arm is one short env var instead of a path into site-packages.
_RELATION_VOCABULARIES = {
    "personal": _BUNDLED_PERSONAL_PROFILE,
    "cco": _BUNDLED_ONTOLOGY,
}

# The INPUT SKOS overlay: the everyday concepts CCO does not model as binary
# predicates (works for, plans to, likes, cares for), each carrying the
# altLabels that make it lexically reachable and domain/range in CCO's own class
# vocabulary. Data, not code — point this elsewhere to change the vocabulary.
_BUNDLED_OVERLAY = str(
    Path(__file__).resolve().parent
    / "symbolic"
    / "ontology"
    / "assets"
    / "cco"
    / "conversational.json"
)


class Env(StrEnum):
    """Which deployment the process believes it is running in.

    Lowercase and closed: ``prod``, ``Production`` and ``PROD`` are rejected at
    construction rather than silently skipping
    :meth:`GraphKnowsSettings.check_production_secrets`.
    """

    development = "development"
    production = "production"


class MemoryMode(StrEnum):
    """Named preset over the orthogonal capability knobs.

    The mode picks the decoder and, as a preset, the topic layer — which stays
    independently overridable. Ontology injection plus the FrameNet layer are
    mode-independent. See :data:`_PRESETS` for the expansion and
    ``docs/modes.md`` for the rationale.

    - ``llm_free``: no network LLM call on the ingestion or retrieval hot path.
    - ``llm_assisted``: the LLM decoder replaces the local extraction stack, and
      topics are LLM summaries.
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


class Decoder(StrEnum):
    """Which component turns a chunk into entities, relations and frame roles.

    - ``local``: the bundled GLiNER2/DeBERTa stack — no network call.
    - ``llm``: one structured LLM call per chunk, *replacing* that stack rather
      than layering on top of it.
    """

    local = "local"
    llm = "llm"


@dataclass(frozen=True)
class _Preset:
    """What a :class:`MemoryMode` expands into."""

    topics: TopicMode
    decoder: Decoder


_PRESETS: dict[MemoryMode, _Preset] = {
    MemoryMode.llm_free: _Preset(topics=TopicMode.none, decoder=Decoder.local),
    MemoryMode.llm_assisted: _Preset(topics=TopicMode.llm, decoder=Decoder.llm),
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

    graphknows_env: Env = Field(default=Env.development, alias="GRAPHKNOWS_ENV")

    # Named preset. The decoder follows the mode alone; the topic layer can be
    # overridden independently. Read the resolved values through the
    # ``decoder`` / ``topic_mode`` properties, never off these raw fields
    # (``None`` means "take the preset's value").
    mode: MemoryMode = Field(default=MemoryMode.llm_free, alias="GRAPHKNOWS_MODE")
    topics: TopicMode | None = Field(default=None, alias="GRAPHKNOWS_TOPICS")
    # Retired (FR-004): under a replacement decoder an "LLM relations" knob names
    # nothing. Kept as a field only so setting it is a deprecation rather than a
    # validation error, and normalised back to ``None`` so nothing can honour it.
    enable_dspy_relations: bool | None = Field(default=None, alias="GRAPHKNOWS_DSPY_RELATIONS")

    # ArcadeDB
    mcp_call_timeout_s: float = Field(
        default=CALL_TIMEOUT_S,
        alias="GRAPHKNOWS_MCP_CALL_TIMEOUT_S",
        description=(
            "Wall-clock budget for one MCP tool call. The first call after a cold "
            "start pays the extraction models' load time; raise this if a "
            "deployment does not warm them."
        ),
    )
    arcadedb_url: str = Field(default="http://localhost:2480", alias="GRAPHKNOWS_ARCADEDB_URL")
    arcadedb_user: str = Field(default="root", alias="GRAPHKNOWS_ARCADEDB_USER")
    arcadedb_password: SecretStr = Field(
        default=SecretStr("changeme"), alias="GRAPHKNOWS_ARCADEDB_PASSWORD"
    )

    # Multi-tenant isolation boundary. Each namespace maps to ONE physical
    # ArcadeDB database (``mem_<ns>``; "" -> ``mem``) — see storage/namespace.py.
    # Short- and long-term memory are not separate databases, they are the
    # ``state`` lifecycle (raw -> consolidated) on that one graph. Used as the
    # default when an MCP call omits an explicit namespace.
    namespace: str = Field(default="", alias="GRAPHKNOWS_NAMESPACE")

    # The destructive MCP surface (``memory_purge``, ``memory_drop_namespace``).
    # Read at import time by ``server/mcp/tools/admin.py``: the gate is on
    # REGISTRATION, so with this off the tools are never advertised at all.
    enable_admin_tools: bool = Field(default=False, alias="GRAPHKNOWS_ENABLE_ADMIN_TOOLS")

    # LLM (provider-neutral; litellm model strings)
    llm_model: str = Field(
        default="openrouter/deepseek/deepseek-v4-flash", alias="GRAPHKNOWS_LLM_MODEL"
    )
    llm_api_key: SecretStr = Field(default=SecretStr(""), alias="GRAPHKNOWS_LLM_API_KEY")
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
    embed_api_key: SecretStr = Field(default=SecretStr(""), alias="GRAPHKNOWS_EMBED_API_KEY")
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

    # Entity extraction.
    spacy_model: str = Field(default="en_core_web_lg", alias="GRAPHKNOWS_SPACY_MODEL")
    # The relex model and its three score floors. What each floor defends, and
    # the measurements behind the numbers, are at the use sites —
    # ingestion/extraction/entities/gliner_model.py and .../extractor.py.
    relex_model: str = Field(
        default="knowledgator/gliner-relex-large-v1.0", alias="GRAPHKNOWS_RELEX_MODEL"
    )
    relex_threshold: float = Field(default=0.35, alias="GRAPHKNOWS_RELEX_THRESHOLD")
    relex_relation_threshold: float = Field(
        default=0.3, alias="GRAPHKNOWS_RELEX_RELATION_THRESHOLD"
    )
    relex_adjacency: float = Field(default=0.6, alias="GRAPHKNOWS_RELEX_ADJACENCY")

    # Definition-described entity typing — a SECOND, smaller model run per line so
    # ontology definitions reach the extractor as label descriptions. Default off:
    # it is another ~0.5B model resident per process and a forward pass per line.
    # See ingestion/extraction/entities/typing_model.py.
    definition_typing: bool = Field(default=False, alias="GRAPHKNOWS_DEFINITION_TYPING")
    # base, not large. Measured on eight conversational lines against the same
    # 13-label CCO schema: base is 784 ms/line to large's 2574 (3.3x), types MORE
    # surfaces (19 vs 17), and wins the two clearest semantic calls outright —
    # "charity race" as an Act (0.93) where large says Organization (0.46), and
    # "fashion internship" as an Occupation Role (0.78) where large says
    # Organization (0.42). Large is better on a couple ("dance studio" as a
    # Facility) but not enough to buy at triple the ingest cost, and the
    # prototype's 22/26 typing result was measured on base too. Both are baked
    # into the image.
    typing_model: str = Field(default="fastino/gliner2-base-v1", alias="GRAPHKNOWS_TYPING_MODEL")
    # Device override for that model ("" = let the library choose).
    typing_device: str = Field(default="", alias="GRAPHKNOWS_TYPING_DEVICE")

    # Order the properties offered inside one (domain, range) group by chunk fit.
    # Ordering, never filtering — the offered set is identical. Default off until
    # its A/B arm decides. See ingestion/stm/ingest.py.
    relation_group_ranking: bool = Field(default=False, alias="GRAPHKNOWS_RELATION_GROUP_RANKING")

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
    # extraction in BOTH modes; classes are persisted as ONTOLOGY_CLASS nodes.
    # Properties ARE persisted too, as ONTOLOGY_PROPERTY vertices carrying the
    # ``functional`` flag: the flush-time relation consolidation in the storage
    # layer reads it to tombstone a superseded single-valued fact, and storage
    # sits below ``symbolic`` in the import-linter layer contract, so it cannot
    # import this loader to ask an in-memory OntologyTerm instead. Read via
    # ``ontology_source``.
    #
    # The default is the bundled Common Core Ontologies digest: a pre-parsed
    # JSON of 1401 classes and 224 object properties that loads WITHOUT rdflib
    # (which ships only in the ``assisted`` extra), so a bare `pip install
    # graphknows` gets the same ontology the container does, and both memory
    # modes load it through one identical code path.
    # ``python -m graphknows.symbolic.ontology.digest`` regenerates it from the TTL sources
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

    # Whether a stored CHUNK is the extraction WINDOW or the single turn.
    #
    # A lone turn is a poor retrieval unit: "She said yes" carries nothing an
    # embedding can match, and the turn that named her is a different chunk.
    # The extractor has read a 5-turn window since ``extract_window`` landed —
    # this makes the stored, embedded, retrieved unit that same window, so what
    # comes back carries the context extraction already depended on. Windows
    # overlap by ``extract_overlap``, so no adjacent pair of turns is split
    # across a boundary, and a turn appearing in two chunks is not double
    # evidence: MENTIONS and REL both MERGE.
    #
    # False restores one-chunk-per-turn exactly, which is what makes it the
    # control arm of an A/B rather than an approximation of one.
    window_chunks: bool = Field(default=True, alias="GRAPHKNOWS_WINDOW_CHUNKS")

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

    # Saturation ceiling for the tag-driven ontology channel
    # (channels/ontology.py). A tag covering more than this fraction of a
    # session's chunks is dropped from the join: like the conversation
    # participants (see the entity-saturation drop), a generic class such as
    # ``Person`` in a personal-conversation corpus matches nearly every chunk,
    # so returning its chunks floods recall with the whole conversation and
    # carries no discriminative signal.
    class_saturation_ratio: float = Field(default=0.5, alias="GRAPHKNOWS_CLASS_SATURATION_RATIO")

    # Pool shape and post-cut expansion (retrieval/retriever.py): retrieve wide,
    # return narrow, and — off by default — stitch each result's seq-adjacent
    # siblings back in for long-document corpora.
    pool_factor: int = Field(default=4, alias="GRAPHKNOWS_POOL_FACTOR")
    min_pool: int = Field(default=100, alias="GRAPHKNOWS_MIN_POOL")
    neighbor_radius: int = Field(default=0, alias="GRAPHKNOWS_NEIGHBOR_RADIUS")

    # The structured fact sheet handed to the generator, and its size cap.
    fact_context: bool = Field(default=True, alias="GRAPHKNOWS_FACT_CONTEXT")
    max_facts: int = Field(default=40, alias="GRAPHKNOWS_MAX_FACTS")

    # The LLM decoder's knobs. Only read on the ``llm`` decoder; inert under the
    # local stack, which has its own (relex, verifier, frames) switches above.
    #
    # The three section switches turn off a part of the ONE structured call —
    # a section the model is not asked for is a section it cannot get wrong, and
    # switching each off in turn is how the sections are ablated (FR-016).
    decode_entities: bool = Field(default=True, alias="GRAPHKNOWS_DECODE_ENTITIES")
    decode_relations: bool = Field(default=True, alias="GRAPHKNOWS_DECODE_RELATIONS")
    decode_frames: bool = Field(default=True, alias="GRAPHKNOWS_DECODE_FRAMES")
    # Relation confidence gate. 0 disables it, which IS the fourth ablation arm
    # (FR-014) — not a special case in the code.
    decode_confidence_min: float = Field(
        default=0.5, ge=0.0, le=1.0, alias="GRAPHKNOWS_DECODE_CONFIDENCE_MIN"
    )
    # Per-chunk retry budget and the exponential-with-full-jitter wait it spends
    # (FR-020). ``Retry-After`` from the provider wins over the computed wait.
    decode_attempts: int = Field(default=3, ge=1, alias="GRAPHKNOWS_DECODE_ATTEMPTS")
    decode_backoff_base_s: float = Field(
        default=1.0, ge=0.0, alias="GRAPHKNOWS_DECODE_BACKOFF_BASE_S"
    )
    decode_backoff_cap_s: float = Field(
        default=30.0, ge=0.0, alias="GRAPHKNOWS_DECODE_BACKOFF_CAP_S"
    )
    decode_honour_retry_after: bool = Field(
        default=True, alias="GRAPHKNOWS_DECODE_HONOUR_RETRY_AFTER"
    )
    # Bounded fan-out of the per-chunk calls in ``extract_batch``: chunks are
    # independent, so they go out concurrently, but not all at once — this is the
    # ceiling that keeps a large batch from tripping provider rate limits.
    decode_concurrency: int = Field(default=8, ge=1, alias="GRAPHKNOWS_DECODE_CONCURRENCY")

    @field_validator("topics", "enable_dspy_relations", mode="before")
    @classmethod
    def _empty_means_unset(cls, value: Any) -> Any:
        """Treat an empty env var as "not set", so it falls back to the preset.

        ``GRAPHKNOWS_TOPICS=`` is how docker-compose and shell wrappers spell an
        absent value; without this it reaches pydantic as ``""`` and fails enum
        validation, taking the whole process down at construction.
        """
        return None if isinstance(value, str) and not value.strip() else value

    @field_validator("enable_dspy_relations", mode="after")
    @classmethod
    def _retired_relations_knob(cls, value: bool | None) -> None:
        """Warn once per construction, then drop the retired knob's value.

        Returning ``None`` is what makes "ignored" structural: the field can
        never hold a value, so no consumer can start honouring it again.
        """
        if value is not None:
            warnings.warn(
                "GRAPHKNOWS_DSPY_RELATIONS / enable_dspy_relations is retired and ignored; "
                "the decoder follows Memory(mode=...) alone.",
                DeprecationWarning,
                stacklevel=2,
            )
        return None

    @property
    def topic_mode(self) -> TopicMode:
        """Resolved topic mode — the explicit setting, else the mode's preset."""
        return self.topics if self.topics is not None else _PRESETS[self.mode].topics

    @property
    def decoder(self) -> Decoder:
        """The resolved decoder — read-only, and a function of the mode alone.

        There is no per-capability override: under a replacement decoder an
        "LLM relations" knob names nothing, so the mode is the whole choice.
        """
        return _PRESETS[self.mode].decoder

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

        Defaults to the bundled personal profile — a separate digest from
        :attr:`ontology_source`, which keeps typing entities with CCO — so a
        deployment can swap relations back to CCO's object properties
        (``GRAPHKNOWS_RELATION_ONTOLOGY=cco``) without replacing the class
        vocabulary that types entities.
        """
        chosen = self.relation_ontology.strip()
        if not chosen:
            return _BUNDLED_RELATION_PROFILE
        # A bare vocabulary name resolves to its bundled asset; anything else is
        # taken as a path, so an explicit file always wins over the shorthand.
        return _RELATION_VOCABULARIES.get(chosen.casefold(), chosen)

    @property
    def uses_llm(self) -> bool:
        """Whether any resolved capability makes network LLM calls."""
        return self.decoder is Decoder.llm or self.topic_mode is TopicMode.llm

    def check_production_secrets(self) -> None:
        """Raise if running in production with unsafe defaults.

        Runs automatically at construction (see :meth:`_validate_secrets`); also
        callable directly as the strict entry point.

        Raises:
            ConfigurationError: When production env is detected with the default
                password, or with an LLM-using capability enabled but no API key.
        """
        if self.graphknows_env is not Env.production:
            return
        if self.arcadedb_password.get_secret_value() == "changeme":
            raise ConfigurationError(
                "GRAPHKNOWS_ARCADEDB_PASSWORD is still the development default."
            )
        # Keyed off the resolved capabilities, not the mode label.
        if self.uses_llm and not self.llm_api_key:
            raise ConfigurationError(
                "LLM-backed capabilities (the LLM decoder, LLM topics) "
                "in production require GRAPHKNOWS_LLM_API_KEY."
            )

    @model_validator(mode="after")
    def _validate_secrets(self) -> GraphKnowsSettings:
        """Fail fast at construction when production secrets are unsafe."""
        self.check_production_secrets()
        return self


@lru_cache(maxsize=1)
def get_settings() -> GraphKnowsSettings:
    """The process's settings, constructed once.

    Construction costs ~140 ms (env + ``.env`` parsing, validation), so hot paths
    read it through here rather than rebuilding per message or per chunk. Code
    that is *handed* a settings instance keeps using that one: an explicit
    ``Memory(settings=s)`` must win over the environment.

    Tests that patch the environment call ``get_settings.cache_clear()``; the
    autouse fixture in ``tests/conftest.py`` does it for every test.
    """
    return GraphKnowsSettings()

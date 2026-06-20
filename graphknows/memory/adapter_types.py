"""Canonical, host-agnostic record types for a generic agent memory adapter.

These types form the contract layer for memory adapters. They contain no
host-specific fields and no framework-specific terminology.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Protocol, runtime_checkable


class MemoryClass(StrEnum):
    """Classification of memory records by semantic role."""

    raw_evidence = "raw_evidence"
    summary = "summary"
    state = "state"
    episode = "episode"
    fact = "fact"
    other = "other"


@dataclass(frozen=True)
class ArtifactRecord:
    """Represents a file or code artifact ingested into memory.

    Attributes:
        artifact_id: Caller-assigned stable identifier.
        path: File path or URL.
        content: Full text content.
        mime_type: MIME type, e.g. "text/x-python" or "text/markdown".
        memory_class: Semantic classification. Defaults to
            :attr:`MemoryClass.raw_evidence`.
        metadata: Optional caller-provided metadata.
    """

    artifact_id: str
    path: str
    content: str
    mime_type: str
    memory_class: MemoryClass = MemoryClass.raw_evidence
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TurnRecord:
    """Represents one complete agent conversation turn.

    Attributes:
        turn_id: Unique turn identifier.
        session_id: Parent session identifier.
        role: Turn role, e.g. "user", "assistant", "system", "tool".
        content: Cleaned text content without raw tool outputs inline.
        memory_class: Semantic classification. Defaults to
            :attr:`MemoryClass.raw_evidence`.
        metadata: Optional caller-provided metadata.
    """

    turn_id: str
    session_id: str
    role: str
    content: str
    memory_class: MemoryClass = MemoryClass.raw_evidence
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RecallRequest:
    """Parameters for a memory recall query.

    Attributes:
        query: Free-text query.
        session_id: Scope to session if provided; ``None`` for global recall.
        top_k: Maximum number of results to return.
        prefer_classes: Classes to boost during retrieval.
        include_raw: Whether to include :attr:`MemoryClass.raw_evidence` in
            results.
    """

    query: str
    session_id: str | None = None
    top_k: int = 10
    prefer_classes: list[MemoryClass] = field(
        default_factory=lambda: [
            MemoryClass.state,
            MemoryClass.episode,
            MemoryClass.fact,
            MemoryClass.summary,
        ]
    )
    include_raw: bool = True


@dataclass(frozen=True)
class PromoteOptions:
    """Controls STM-to-LTM promotion behavior.

    Attributes:
        session_id: Session to promote.
        run_gds: Run GDS enrichment after promotion.
    """

    session_id: str
    run_gds: bool = True


@dataclass(frozen=True)
class AdapterCapabilities:
    """Describes what a concrete adapter instance supports.

    Attributes:
        supports_stm: Whether short-term memory is supported.
        supports_ltm: Whether long-term memory is supported.
        supports_dialog_graph: Whether dialog graph storage is supported.
        supports_artifact_ingest: Whether artifact ingestion is supported.
        supports_recall: Whether recall queries are supported.
        supports_promote: Whether STM-to-LTM promotion is supported.
    """

    supports_stm: bool
    supports_ltm: bool
    supports_dialog_graph: bool
    supports_artifact_ingest: bool
    supports_recall: bool
    supports_promote: bool


# ---------------------------------------------------------------------------
# Response types for GraphknowsMemoryAdapter
# ---------------------------------------------------------------------------


@dataclass
class RecallHit:
    """A single hit returned by a memory recall query."""

    score: float
    text: str
    session_id: str
    entities: list[str] = field(default_factory=list)
    source_type: str = ""


@dataclass
class FlushResult:
    """Result of flushing a session from STM to LTM."""

    nodes: int
    edges: int
    chunks: int
    turns: int
    elapsed_s: float
    errors: list[str] = field(default_factory=list)


@dataclass
class MemoryStats:
    """Aggregate node/chunk counts for STM and LTM."""

    stm: dict[str, Any] = field(default_factory=dict)
    ltm: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class AgentMemoryAdapter(Protocol):
    """Stable interface between an agent runtime and Graphknows memory.

    Implementations must be async-safe and session-scoped.
    All methods are async so they can be used in any async agent loop.

    The ``messages`` / ``options`` / ``scope`` parameters accept the types from
    ``graphknows.memory.models`` — typed as ``Any`` here to avoid circular imports.
    """

    async def ingest(self, messages: Any, options: Any) -> Any:
        """Ingest messages into STM."""
        ...

    async def recall(
        self,
        query: str,
        scope: Any,
        top_k: int = 5,
        recall_scope: str = "stm",
    ) -> list[RecallHit]:
        """Recall relevant chunks from memory for the given query."""
        ...

    async def flush(self, scope: Any) -> FlushResult:
        """Flush a session from STM to LTM."""
        ...

    async def stats(self, scope: Any | None = None) -> MemoryStats:
        """Return node/chunk counts for STM and LTM."""
        ...

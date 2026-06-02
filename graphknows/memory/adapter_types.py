"""Canonical, host-agnostic record types for a generic agent memory adapter.

These types form the contract layer for memory adapters. They contain no
host-specific fields and no framework-specific terminology.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class MemoryClass(str, Enum):  # noqa: UP042
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
class ToolCallRecord:
    """Represents one tool invocation during an agent turn.

    Attributes:
        tool_id: Unique call identifier.
        tool_name: Tool name, e.g. "terminal", "read_file", "web_search".
        input_summary: Human-readable summary of the input.
        output_summary: Human-readable summary of the output.
        exit_code: Exit code for shell/process tools; ``None`` for non-shell
            tools.
        error: First meaningful error line; ``None`` on success.
        memory_class: Semantic classification. Defaults to
            :attr:`MemoryClass.raw_evidence`.
        metadata: Optional caller-provided metadata.
    """

    tool_id: str
    tool_name: str
    input_summary: str
    output_summary: str
    exit_code: int | None = None
    error: str | None = None
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
        tool_calls: Tool calls made in this turn.
        memory_class: Semantic classification. Defaults to
            :attr:`MemoryClass.raw_evidence`.
        metadata: Optional caller-provided metadata.
    """

    turn_id: str
    session_id: str
    role: str
    content: str
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    memory_class: MemoryClass = MemoryClass.raw_evidence
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ParticipantRecord:
    """Represents a participant in a dialog.

    Attributes:
        participant_id: Unique participant identifier.
        role: Participant role, e.g. "user", "assistant", "system".
        display_name: Optional display name.
        metadata: Optional caller-provided metadata.
    """

    participant_id: str
    role: str
    display_name: str | None = None
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
        run_topics: Deprecated compatibility flag; promotion always runs topic
            summarization.
    """

    session_id: str
    run_gds: bool = True
    run_topics: bool = True


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

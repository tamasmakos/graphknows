"""IngestOptions: typed options object for memory ingestion."""

from __future__ import annotations

from pydantic import BaseModel, Field

from graphknows.models.memory_class import MemoryClass
from graphknows.models.scope import MemoryScope


class IngestOptions(BaseModel, frozen=True):
    """Options controlling how messages are ingested into memory.

    Attributes:
        scope: Identity triple (user_id / agent_id / run_id).
        metadata: Ingest-time hints read by the write path. Only ``timestamp``,
            ``anchor_date`` and ``memory_class`` are stored — ``timestamp`` or
            ``anchor_date`` dates the chunk and anchors relative-date
            resolution, ``memory_class`` classifies it. Every other key is
            dropped.
        infer: When True, run NER extraction and build entity graph nodes.
            When False, embed only — no NER, no STM graph writes. Useful for
            tool output, system prompts, and high-volume ephemeral content.
        memory_class: Semantic classification of the ingested content.
        turn_index_offset: Starting turn index for appending to existing dialogs.
    """

    scope: MemoryScope
    metadata: dict[str, object] = Field(default_factory=dict)
    infer: bool = True
    memory_class: MemoryClass = MemoryClass.raw_evidence
    turn_index_offset: int = 0

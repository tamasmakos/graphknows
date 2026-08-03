"""Cross-cutting domain models — the vocabulary shared by every layer.

Used by ingestion, retrieval and every endpoint. Layer-neutral: these depend on
nothing else in the package.
"""

from graphknows.models.hit import (
    MAX_CONTEXT_CHARS,
    Hit,
    RenderedMemories,
    date_in,
    render_memories,
)
from graphknows.models.ingest import IngestResult
from graphknows.models.ingest_options import IngestOptions
from graphknows.models.memory_class import MemoryClass
from graphknows.models.message import Message, MessageInput, Role, normalize_messages
from graphknows.models.scope import MemoryScope

__all__ = [
    "MAX_CONTEXT_CHARS",
    "Hit",
    "IngestOptions",
    "IngestResult",
    "MemoryClass",
    "MemoryScope",
    "Message",
    "MessageInput",
    "RenderedMemories",
    "Role",
    "date_in",
    "normalize_messages",
    "render_memories",
]

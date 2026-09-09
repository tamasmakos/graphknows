"""Cross-cutting domain models — the vocabulary shared by every layer.

Used by ingestion, retrieval and every endpoint. Layer-neutral: these depend on
nothing else in the package.
"""

from graphknows.models.fact import (
    Fact,
    FactState,
    Mention,
    Modality,
    Polarity,
    Validity,
)
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
from graphknows.models.source import Source
from graphknows.models.symbols import ConceptRef, PredicateRef, SymbolKind

__all__ = [
    "MAX_CONTEXT_CHARS",
    "ConceptRef",
    "Fact",
    "FactState",
    "Hit",
    "IngestOptions",
    "IngestResult",
    "MemoryClass",
    "MemoryScope",
    "Mention",
    "Message",
    "MessageInput",
    "Modality",
    "Polarity",
    "PredicateRef",
    "RenderedMemories",
    "Role",
    "Source",
    "SymbolKind",
    "Validity",
    "date_in",
    "normalize_messages",
    "render_memories",
]

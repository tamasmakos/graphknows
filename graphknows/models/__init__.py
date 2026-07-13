"""Cross-cutting domain models — the vocabulary shared by ingestion, retrieval,
and every endpoint. Layer-neutral: these depend on nothing else in the package.
"""

from graphknows.models.evidence import EvidenceItem
from graphknows.models.ingest import IngestRequest, IngestResult
from graphknows.models.ingest_options import IngestOptions
from graphknows.models.memory_class import MemoryClass
from graphknows.models.message import Message, MessageInput, Role, normalize_messages
from graphknows.models.scope import MemoryScope

__all__ = [
    "EvidenceItem",
    "IngestOptions",
    "IngestRequest",
    "IngestResult",
    "MemoryClass",
    "MemoryScope",
    "Message",
    "MessageInput",
    "Role",
    "normalize_messages",
]

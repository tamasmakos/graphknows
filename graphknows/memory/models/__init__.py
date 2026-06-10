"""Memory operation DTOs."""

from graphknows.memory.adapter_types import MemoryClass
from graphknows.memory.models.evidence import EvidenceItem
from graphknows.memory.models.ingest import IngestRequest, IngestResult
from graphknows.memory.models.ingest_options import IngestOptions
from graphknows.memory.models.message import Message, MessageInput, Role, normalize_messages
from graphknows.memory.models.promotion import PromotionRecord, PromotionRequest
from graphknows.memory.models.recall import RecallQuery, RecallResult
from graphknows.memory.models.scope import MemoryScope

__all__ = [
    "EvidenceItem",
    "IngestOptions",
    "IngestRequest",
    "IngestResult",
    "MemoryClass",
    "MemoryScope",
    "Message",
    "MessageInput",
    "PromotionRecord",
    "PromotionRequest",
    "RecallQuery",
    "RecallResult",
    "Role",
    "normalize_messages",
]

"""Memory operation DTOs."""

from graphknows.memory.adapter_types import MemoryClass
from graphknows.memory.models.contracts import (
    CommunityAssignment,
    ExtractedEntity,
    ExtractedRelation,
    ExtractionResult,
    MemoryRecord,
    PromotionAudit,
    RetrievalHit,
    SessionSnapshot,
)
from graphknows.memory.models.evidence import EvidenceItem
from graphknows.memory.models.ingest import IngestRequest, IngestResult
from graphknows.memory.models.ingest_options import IngestOptions
from graphknows.memory.models.message import Message, MessageInput, Role, normalize_messages
from graphknows.memory.models.promotion import PromotionPolicy, PromotionRecord, PromotionRequest
from graphknows.memory.models.recall import RecallQuery, RecallResult
from graphknows.memory.models.scope import MemoryScope

__all__ = [
    "CommunityAssignment",
    "EvidenceItem",
    "ExtractedEntity",
    "ExtractedRelation",
    "ExtractionResult",
    "IngestOptions",
    "IngestRequest",
    "IngestResult",
    "MemoryRecord",
    "MemoryClass",
    "MemoryScope",
    "Message",
    "MessageInput",
    "PromotionAudit",
    "PromotionPolicy",
    "PromotionRecord",
    "PromotionRequest",
    "RecallQuery",
    "RecallResult",
    "RetrievalHit",
    "Role",
    "SessionSnapshot",
    "normalize_messages",
]

"""Memory operation DTOs."""
from graphknows.memory.models.evidence import EvidenceItem
from graphknows.memory.models.ingest import IngestRequest, IngestResult
from graphknows.memory.models.promotion import PromotionRecord, PromotionRequest
from graphknows.memory.models.recall import RecallQuery, RecallResult
from graphknows.memory.adapter_types import MemoryClass

__all__ = [
    "IngestRequest",
    "IngestResult",
    "RecallResult",
    "RecallQuery",
    "PromotionRequest",
    "PromotionRecord",
    "EvidenceItem",
    "MemoryClass",
]

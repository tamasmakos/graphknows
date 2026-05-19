"""Data models for summarization tasks."""

from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Tuple

@dataclass
class SummarizationTask:
    """Represents a summarization unit for a community or subcommunity."""
    task_id: str
    community_id: int
    subcommunity_id: Optional[int]
    is_topic: bool
    
    # Context Data
    chunk_texts: List[str]
    entities: List[Dict[str, Any]]  # format: {'name': str, 'id': str, 'degree': int}
    relationships: List[Tuple[str, str, str]]  # format: (source, type, target)
    
    # Metadata
    chunk_ids: List[str]
    entity_ids: List[str]

    # Optional / Default fields
    sub_summaries: Optional[List[Dict[str, str]]] = None  # for Topics: [{'id': str, 'summary': str}]

    # Output
    title: Optional[str] = None
    summary: Optional[str] = None
    findings: Optional[List[Dict[str, Any]]] = None

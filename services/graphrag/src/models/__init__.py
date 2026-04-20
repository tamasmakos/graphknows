"""
Citation model and AgentResponse for structured RAG answers.
"""

from __future__ import annotations

from typing import Any
from pydantic import BaseModel


class Citation(BaseModel):
    chunk_id: str
    doc_id: str
    doc_title: str
    heading_path: list[str] = []
    text_excerpt: str
    score: float


class AgentResponse(BaseModel):
    answer: str
    citations: list[Citation] = []
    graph_data: dict[str, Any] = {"nodes": [], "edges": []}
    execution_time: float = 0.0
    conversation_id: str = ""

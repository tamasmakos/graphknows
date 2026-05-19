"""Pydantic request/response models for the agent HTTP API."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, model_validator


class ChatRequest(BaseModel):
    """Request body for POST /chat endpoint."""

    message: str
    session_id: Optional[str] = None

    @model_validator(mode="after")
    def _check_message(self) -> "ChatRequest":
        if not self.message.strip():
            raise ValueError("message must not be empty")
        return self


class GraphResponse(BaseModel):
    """Response model for graph endpoints."""

    nodes: List[Dict[str, Any]]
    edges: List[Dict[str, Any]]


class ChatResponse(BaseModel):
    """Response model for POST /chat endpoint."""

    answer: str
    session_id: str
    graph: GraphResponse
    reasoning_steps: list = []
    sources: list = []


class QueryRequest(BaseModel):
    """Request body for POST /sessions/{id}/query endpoint."""

    message: str

    @model_validator(mode="after")
    def _check_message(self) -> "QueryRequest":
        if not self.message.strip():
            raise ValueError("message must not be empty")
        return self

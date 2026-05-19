"""Pydantic v2 models for the Graphknows Search Service."""

from __future__ import annotations

from typing import Optional
from pydantic import BaseModel


class SearchQuery(BaseModel):
    query: str
    sources: Optional[list[str]] = None  # None = all sources
    date_from: Optional[str] = None  # ISO date YYYY-MM-DD
    date_to: Optional[str] = None
    limit: int = 20


class SearchResult(BaseModel):
    id: str
    source: str
    title: str
    date: Optional[str] = None
    url: Optional[str] = None
    excerpt: Optional[str] = None
    already_ingested: bool = False


class ScrapeRequest(BaseModel):
    document_ids: list[str]
    source: str


class ScrapeJob(BaseModel):
    job_id: str
    status: str  # queued | running | complete | failed
    created_at: str
    progress: float = 0.0
    result: Optional[list[dict]] = None
    error: Optional[str] = None


class Document(BaseModel):
    id: str
    source: str
    title: str
    text: str
    metadata: dict = {}
    ingested_at: str

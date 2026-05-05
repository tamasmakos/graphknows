"""Pipeline-level request/response models shared between search, graphgen, and agent."""
from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


class Document(BaseModel):
    """A source document to be ingested by the graphgen pipeline.

    Attributes:
        id: Unique document identifier (URL or hash).
        title: Document title.
        text: Full document text.
        source: Source adapter name (e.g. "ep_debates", "eurlex").
        metadata: Additional source-specific metadata.
    """

    id: str
    title: str
    text: str
    source: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class BuildRequest(BaseModel):
    """Request body for POST /build on the graphgen service.

    Attributes:
        documents: List of documents to process through the KG pipeline.
        run_node2vec: Whether to run Node2Vec after graph construction.
        run_leiden: Whether to run Leiden community detection.
    """

    documents: list[Document]
    run_node2vec: bool = True
    run_leiden: bool = True


class BuildJob(BaseModel):
    """Job status response for a graphgen build job.

    Attributes:
        job_id: Unique job identifier.
        status: Current job status.
        documents_total: Total documents queued.
        documents_done: Documents processed so far.
        error: Error message if status is "failed".
    """

    job_id: str
    status: Literal["queued", "running", "complete", "failed"]
    documents_total: int = 0
    documents_done: int = 0
    error: Optional[str] = None


class SearchQuery(BaseModel):
    """Query body for POST /search on the search service.

    Attributes:
        q: The raw query string.
        sources: Adapter names to query (empty means all).
        limit: Max results per source.
    """

    q: str
    sources: list[str] = Field(default_factory=list)
    limit: int = 10


class SearchResult(BaseModel):
    """A single search result from the search service.

    Attributes:
        doc_id: Source document identifier.
        title: Document title.
        snippet: Relevant excerpt.
        source: Source adapter name.
        score: Relevance score (higher is better).
    """

    doc_id: str
    title: str
    snippet: str
    source: str
    score: float = 0.0

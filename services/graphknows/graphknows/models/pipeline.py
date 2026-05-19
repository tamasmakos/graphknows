"""Pipeline-level request and response models.

Example:
    from graphknows.models.pipeline import Document, BuildRequest, BuildJob
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field

from graphknows.models.base import GraphKnowsBaseModel, ImmutableModel


class Document(ImmutableModel):
    """A source document to be ingested by the graphgen pipeline.

    Attributes:
        id: Unique document identifier (URL or hash).
        title: Document title.
        text: Full document text.
        source: Source adapter name (e.g. ``"ep_debates"``, ``"eurlex"``).
        metadata: Additional source-specific metadata.
    """

    id: str
    title: str
    text: str
    source: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class BuildRequest(ImmutableModel):
    """Request body for triggering a graphgen pipeline run.

    Attributes:
        documents: List of documents to process.
        run_node2vec: Whether to run Node2Vec after graph construction.
        run_leiden: Whether to run Leiden community detection.
    """

    documents: list[Document]
    run_node2vec: bool = True
    run_leiden: bool = True


class BuildJob(GraphKnowsBaseModel):
    """Mutable job status for a graphgen build run.

    Status transitions: queued → running → complete | failed.

    Attributes:
        job_id: Unique job identifier.
        status: Current job status.
        documents_total: Total documents queued.
        documents_done: Documents processed so far.
        error: Error message when status is ``"failed"``.
    """

    job_id: str
    status: Literal["queued", "running", "complete", "failed"]
    documents_total: int = 0
    documents_done: int = 0
    error: str | None = None


class SearchQuery(ImmutableModel):
    """Query body for a hybrid search request.

    Attributes:
        q: Raw query string.
        sources: Adapter names to query (empty means all).
        limit: Maximum results per source.
    """

    q: str
    sources: list[str] = Field(default_factory=list)
    limit: int = 10


class SearchResult(ImmutableModel):
    """A single search result from the retrieval layer.

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

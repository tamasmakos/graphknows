"""Async HTTP client for the Graphknows Search Service."""
from __future__ import annotations

import asyncio
import logging
import os

import httpx

from services.agent.graph.state import RawDoc
from services.search.models import ScrapeJob, ScrapeRequest, SearchQuery, SearchResult

logger = logging.getLogger(__name__)


class SearchServiceClient:
    """Async HTTP client wrapping the search service REST API.

    All calls go through HTTP — never imports adapters directly.
    """

    def __init__(
        self,
        base_url: str | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self._base_url = (base_url or os.environ.get(
            "SEARCH_SERVICE_URL", "http://localhost:8040"
        )).rstrip("/") + "/api/v1"
        self._client = http_client or httpx.AsyncClient(timeout=30.0)

    async def search(
        self,
        query: str,
        sources: list[str] | None = None,
        limit: int = 10,
    ) -> list[SearchResult]:
        """Call ``POST /search`` and return parsed ``SearchResult`` list."""
        body = SearchQuery(query=query, sources=sources, limit=limit)
        resp = await self._client.post(
            f"{self._base_url}/search",
            json=body.model_dump(exclude_none=True),
        )
        resp.raise_for_status()
        return [SearchResult(**item) for item in resp.json()]

    async def scrape(self, source: str, document_ids: list[str]) -> ScrapeJob:
        """Call ``POST /scrape`` to enqueue a background fetch job."""
        body = ScrapeRequest(source=source, document_ids=document_ids)
        resp = await self._client.post(
            f"{self._base_url}/scrape",
            json=body.model_dump(),
        )
        resp.raise_for_status()
        return ScrapeJob(**resp.json())

    async def poll_job(
        self,
        job_id: str,
        timeout: float = 30.0,
        interval: float = 1.0,
    ) -> ScrapeJob:
        """Poll ``GET /jobs/{job_id}`` until status is ``complete`` or ``failed``.

        Raises:
            TimeoutError: If the job does not finish within *timeout* seconds.
        """
        deadline = asyncio.get_event_loop().time() + timeout
        while True:
            resp = await self._client.get(f"{self._base_url}/jobs/{job_id}")
            resp.raise_for_status()
            job = ScrapeJob(**resp.json())
            if job.status in ("complete", "failed"):
                return job
            if asyncio.get_event_loop().time() >= deadline:
                raise TimeoutError(
                    f"Job {job_id} did not complete within {timeout}s "
                    f"(last status: {job.status})",
                )
            await asyncio.sleep(interval)

    async def fetch_documents(
        self,
        query: str,
        sources: list[str] | None = None,
        limit: int = 10,
    ) -> list[RawDoc]:
        """High-level helper: search → scrape → poll → return ``list[RawDoc]``."""
        results = await self.search(query, sources, limit)
        if not results:
            return []

        by_source: dict[str, list[str]] = {}
        for r in results:
            by_source.setdefault(r.source, []).append(r.id)

        raw_docs: list[RawDoc] = []
        for source, doc_ids in by_source.items():
            job = await self.scrape(source, doc_ids)
            job = await self.poll_job(job.job_id)
            if job.status == "complete" and job.result:
                for doc_dict in job.result:
                    raw_docs.append(RawDoc(
                        id=doc_dict.get("id", ""),
                        title=doc_dict.get("title", ""),
                        text=doc_dict.get("text", ""),
                        url=doc_dict.get("metadata", {}).get("url", ""),
                        source=doc_dict.get("source", source),
                        date=doc_dict.get("ingested_at", None),
                    ))
        return raw_docs

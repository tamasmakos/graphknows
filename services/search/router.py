"""FastAPI router — all /api/v1 endpoints for the Search Service."""

from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime, timezone
from typing import Any

import httpx
from fastapi import APIRouter, BackgroundTasks, HTTPException

from services.search.adapters.ep_debates import EPDebatesAdapter
from services.search.adapters.eurlex import EurLexAdapter
from services.search.adapters.uk_legislation import UKLegislationAdapter
from services.search.job_queue import enqueue, get_job, make_job, update_job
from services.search.models import (
    Document,
    ScrapeJob,
    ScrapeRequest,
    SearchQuery,
    SearchResult,
)

logger = logging.getLogger(__name__)
router = APIRouter()

# Adapter registry — instantiated once at import time
_ADAPTERS = {
    "ep_debates": EPDebatesAdapter(),
    "eurlex": EurLexAdapter(),
    "uk_legislation": UKLegislationAdapter(),
}


# ------------------------------------------------------------------ #
# Health / meta                                                         #
# ------------------------------------------------------------------ #

@router.get("/health")
async def health() -> dict:
    return {"status": "ok", "service": "graphknows-search"}


@router.get("/sources")
async def list_sources() -> list[dict]:
    return [
        {
            "id": adapter.source_id,
            "rate_limit": adapter.rate_limit,
        }
        for adapter in _ADAPTERS.values()
    ]


# ------------------------------------------------------------------ #
# Search                                                                #
# ------------------------------------------------------------------ #

@router.post("/search", response_model=list[SearchResult])
async def search(query: SearchQuery) -> list[SearchResult]:
    """
    Search across one or more sources.
    Pass `sources` to restrict; omit for all.
    """
    sources = query.sources or list(_ADAPTERS.keys())
    unknown = set(sources) - set(_ADAPTERS.keys())
    if unknown:
        raise HTTPException(status_code=400, detail=f"Unknown sources: {unknown}")

    tasks = [_ADAPTERS[src].search(query) for src in sources]
    results_per_source = await asyncio.gather(*tasks, return_exceptions=True)

    combined: list[SearchResult] = []
    for src, res in zip(sources, results_per_source):
        if isinstance(res, Exception):
            logger.warning("Adapter %s search failed: %s", src, res)
            continue
        combined.extend(res)

    # Sort by date descending (None dates go last), then slice to limit
    def _sort_key(r: SearchResult):
        return r.date or ""

    combined.sort(key=_sort_key, reverse=True)
    return combined[: query.limit]


# ------------------------------------------------------------------ #
# Scrape (async job)                                                    #
# ------------------------------------------------------------------ #

async def _run_scrape_job(job_id: str, req: ScrapeRequest) -> None:
    """Background task: fetch each requested document and store result."""
    adapter = _ADAPTERS.get(req.source)
    if adapter is None:
        update_job(job_id, status="failed", error=f"Unknown source: {req.source}")
        return

    update_job(job_id, status="running")
    docs: list[dict] = []
    total = len(req.document_ids)

    for i, doc_id in enumerate(req.document_ids, start=1):
        try:
            doc: Document = await adapter.fetch(doc_id)
            docs.append(doc.model_dump())
        except Exception as exc:
            logger.warning("Failed to fetch %s/%s: %s", req.source, doc_id, exc)
        update_job(job_id, progress=round(i / total, 2))

    update_job(
        job_id,
        status="complete",
        progress=1.0,
        result=docs,
    )

    # Fire-and-forget: send each scraped document to graphgen for KG ingestion.
    # Uses a fresh client per job — no blocking, errors are logged only.
    graphgen_url = os.environ.get("GRAPHGEN_URL", "http://graphgen:8003")
    async with httpx.AsyncClient(timeout=10.0) as client:
        for doc in docs:
            try:
                await client.post(f"{graphgen_url}/ingest", json=doc)
            except Exception as exc:
                logger.warning("graphgen ingest fire-and-forget failed for %s: %s", doc.get("id"), exc)


@router.post("/scrape", response_model=ScrapeJob)
async def scrape(req: ScrapeRequest, background_tasks: BackgroundTasks) -> ScrapeJob:
    if req.source not in _ADAPTERS:
        raise HTTPException(status_code=400, detail=f"Unknown source: {req.source!r}")
    if not req.document_ids:
        raise HTTPException(status_code=422, detail="document_ids must not be empty")

    job = make_job()
    await enqueue(job)
    background_tasks.add_task(_run_scrape_job, job.job_id, req)
    return job


@router.get("/jobs/{job_id}", response_model=ScrapeJob)
async def get_job_status(job_id: str) -> ScrapeJob:
    data = get_job(job_id)
    if data is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id!r} not found")
    return ScrapeJob(**data)

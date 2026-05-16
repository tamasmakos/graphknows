"""FastAPI router — all /api/v1 endpoints for the Search Service."""

from __future__ import annotations

import asyncio
import logging
import os

import httpx
from fastapi import APIRouter, BackgroundTasks, HTTPException

from .connector_registry import get_registry
from .job_queue import enqueue, get_job, make_job, update_job
from .models import (
    Document,
    ScrapeJob,
    ScrapeRequest,
    SearchQuery,
    SearchResult,
)

logger = logging.getLogger(__name__)
router = APIRouter()

# Registry is loaded once at import time from source_manifest.yaml
_registry = get_registry()


# ------------------------------------------------------------------ #
# Health / meta                                                         #
# ------------------------------------------------------------------ #

@router.get("/health")
async def health() -> dict:
    return {"status": "ok", "service": "graphknows-search"}


@router.get("/sources")
async def list_sources() -> list[dict]:
    entries = _registry.list_entries()
    return [
        {
            "id": e.id,
            "name": e.name,
            "description": e.description.strip(),
            "jurisdictions": e.jurisdictions,
            "doc_types": e.doc_types,
            "temporal_coverage": e.temporal_coverage,
            "language": e.language,
            "rate_limit_rps": e.rate_limit_rps,
            "requires_auth": e.requires_auth,
        }
        for e in entries
    ]


# ------------------------------------------------------------------ #
# Search                                                                #
# ------------------------------------------------------------------ #

@router.post("/search", response_model=list[SearchResult])
async def search(query: SearchQuery) -> list[SearchResult]:
    """Search across one or more sources. Pass `sources` to restrict; omit for all."""
    available = set(_registry.list_ids())
    sources = query.sources or list(available)
    unknown = set(sources) - available
    if unknown:
        raise HTTPException(status_code=400, detail=f"Unknown sources: {sorted(unknown)}")

    connectors = [_registry.get(src) for src in sources]
    tasks = [c.search(query) for c in connectors if c is not None]
    results_per_source = await asyncio.gather(*tasks, return_exceptions=True)

    combined: list[SearchResult] = []
    for src, res in zip(sources, results_per_source):
        if isinstance(res, Exception):
            logger.warning("Connector %s search failed: %s", src, res)
            continue
        combined.extend(res)

    combined.sort(key=lambda r: r.date or "", reverse=True)
    return combined[: query.limit]


# ------------------------------------------------------------------ #
# Scrape (async job)                                                    #
# ------------------------------------------------------------------ #

async def _run_scrape_job(job_id: str, req: ScrapeRequest) -> None:
    """Background task: fetch each requested document and store result."""
    connector = _registry.get(req.source)
    if connector is None:
        await update_job(job_id, status="failed", error=f"Unknown source: {req.source}")
        return

    await update_job(job_id, status="running")
    docs: list[dict] = []
    total = len(req.document_ids)

    for i, doc_id in enumerate(req.document_ids, start=1):
        try:
            doc: Document = await connector.fetch(doc_id)
            docs.append(doc.model_dump())
        except Exception as exc:
            logger.warning("Failed to fetch %s/%s: %s", req.source, doc_id, exc)
        await update_job(job_id, progress=round(i / total, 2))

    await update_job(
        job_id,
        status="complete",
        progress=1.0,
        result=docs,
    )

    graphgen_url = os.environ.get("GRAPHGEN_URL", "http://graphgen:8003")
    async with httpx.AsyncClient(timeout=10.0) as client:
        for doc in docs:
            try:
                await client.post(f"{graphgen_url}/ingest", json=doc)
            except Exception as exc:
                logger.warning(
                    "graphgen ingest failed for %s: %s", doc.get("id"), exc
                )


@router.post("/scrape", response_model=ScrapeJob)
async def scrape(req: ScrapeRequest, background_tasks: BackgroundTasks) -> ScrapeJob:
    if _registry.get(req.source) is None:
        raise HTTPException(status_code=400, detail=f"Unknown source: {req.source!r}")
    if not req.document_ids:
        raise HTTPException(status_code=422, detail="document_ids must not be empty")

    job = make_job()
    await enqueue(job)
    background_tasks.add_task(_run_scrape_job, job.job_id, req)
    return job


@router.get("/jobs/{job_id}", response_model=ScrapeJob)
async def get_job_status(job_id: str) -> ScrapeJob:
    data = await get_job(job_id)
    if data is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id!r} not found")
    return ScrapeJob(**data)


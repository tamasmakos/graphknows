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

# ── Reranker ─────────────────────────────────────────────────────────────────

_OPENROUTER_EMBED_MODEL = "openai/text-embedding-3-small"
_OPENROUTER_EMBED_URL = "https://openrouter.ai/api/v1/embeddings"


def _embed_via_openrouter(texts: list[str]) -> list[list[float]] | None:
    """Call OpenRouter embeddings API. Returns list of float vectors or None on failure."""
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        return None
    try:
        with httpx.Client(timeout=15.0) as client:
            resp = client.post(
                _OPENROUTER_EMBED_URL,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={"model": _OPENROUTER_EMBED_MODEL, "input": texts},
            )
            resp.raise_for_status()
            data = resp.json()["data"]
            return [item["embedding"] for item in sorted(data, key=lambda x: x["index"])]
    except Exception as exc:  # noqa: BLE001
        logger.warning("OpenRouter embedding call failed, falling back to Jaccard: %s", exc)
        return None


def _cosine_sim(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two equal-length float vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _jaccard_score(query: str, text: str) -> float:
    """Simple keyword overlap (Jaccard on lowercased unigrams)."""
    q_tokens = set(query.lower().split())
    t_tokens = set(text.lower().split())
    if not q_tokens or not t_tokens:
        return 0.0
    intersection = q_tokens & t_tokens
    union = q_tokens | t_tokens
    return len(intersection) / len(union)


def _rerank(query: str, results: list[SearchResult], min_score: float = 0.15) -> list[SearchResult]:
    """Rerank results by relevance to query; drop docs with score < min_score."""
    if not results:
        return results

    texts = [f"{r.title} {r.excerpt or ''}" for r in results]

    embs = _embed_via_openrouter([query] + texts)
    if embs is not None:
        q_emb = embs[0]
        scores = [_cosine_sim(q_emb, e) for e in embs[1:]]
    else:
        scores = [_jaccard_score(query, t) for t in texts]

    scored = [(r, s) for r, s in zip(results, scores) if s >= min_score]
    scored.sort(key=lambda x: -x[1])
    return [r for r, _ in scored]


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
    # Overfetch (up to 4× limit) to give reranking a meaningful candidate pool
    _RERANK_POOL_FACTOR = 4
    _RERANK_MIN_POOL = 20
    fetch_limit = max(query.limit * _RERANK_POOL_FACTOR, _RERANK_MIN_POOL)
    fetch_query = query.model_copy(update={"limit": fetch_limit})
    tasks = [c.search(fetch_query) for c in connectors if c is not None]
    results_per_source = await asyncio.gather(*tasks, return_exceptions=True)

    combined: list[SearchResult] = []
    for src, res in zip(sources, results_per_source):
        if isinstance(res, Exception):
            logger.warning("Connector %s search failed: %s", src, res)
            continue
        combined.extend(res)

    # Rerank by cosine similarity (or keyword fallback); drop off-topic docs
    reranked = _rerank(query.query, combined)

    # If reranking dropped everything (all below threshold), fall back to date-sorted
    if not reranked and combined:
        logger.info("Reranking dropped all results; falling back to date-sorted results")
        combined.sort(key=lambda r: r.date or "", reverse=True)
        return combined[: query.limit]

    return reranked[: query.limit]


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
                logger.warning("graphgen ingest failed for %s: %s", doc.get("id"), exc)


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

"""Async in-memory job queue for scrape tasks."""

from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timezone

from .models import ScrapeJob

_jobs: dict[str, dict] = {}
_queue: asyncio.Queue = asyncio.Queue()


async def enqueue(job: ScrapeJob) -> str:
    _jobs[job.job_id] = job.model_dump()
    await _queue.put(job.job_id)
    return job.job_id


def get_job(job_id: str) -> dict | None:
    return _jobs.get(job_id)


def update_job(job_id: str, **kwargs) -> None:
    if job_id in _jobs:
        _jobs[job_id].update(kwargs)


def make_job() -> ScrapeJob:
    """Factory: create a fresh queued ScrapeJob."""
    return ScrapeJob(
        job_id=str(uuid.uuid4()),
        status="queued",
        created_at=datetime.now(tz=timezone.utc).isoformat(),
    )

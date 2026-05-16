"""Async SQLite-backed job queue for scrape tasks.

Replaces the volatile in-memory queue with a persistent aiosqlite store
so jobs survive service restarts. Public API is identical to the previous
in-memory implementation (all async).
"""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

import aiosqlite

from .models import ScrapeJob

_DB_PATH = Path(os.environ.get("JOB_DB_PATH", "/data/jobs.db"))
_INIT_LOCK: bool = False


async def _ensure_db() -> None:
    global _INIT_LOCK
    if _INIT_LOCK:
        return
    _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    async with aiosqlite.connect(_DB_PATH) as db:
        await db.execute(
            """
            CREATE TABLE IF NOT EXISTS jobs (
                job_id     TEXT PRIMARY KEY,
                status     TEXT NOT NULL,
                created_at TEXT NOT NULL,
                progress   REAL DEFAULT 0.0,
                result     TEXT,
                error      TEXT
            )
            """
        )
        await db.commit()
    _INIT_LOCK = True


async def enqueue(job: ScrapeJob) -> str:
    await _ensure_db()
    async with aiosqlite.connect(_DB_PATH) as db:
        await db.execute(
            "INSERT OR REPLACE INTO jobs (job_id, status, created_at, progress) VALUES (?, ?, ?, ?)",
            (job.job_id, job.status, job.created_at, job.progress or 0.0),
        )
        await db.commit()
    return job.job_id


async def get_job(job_id: str) -> dict | None:
    await _ensure_db()
    async with aiosqlite.connect(_DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)) as cur:
            row = await cur.fetchone()
    if row is None:
        return None
    data = dict(row)
    if data.get("result"):
        try:
            data["result"] = json.loads(data["result"])
        except Exception:
            pass
    return data


async def update_job(job_id: str, **kwargs) -> None:
    if not kwargs:
        return
    await _ensure_db()
    if "result" in kwargs and kwargs["result"] is not None:
        kwargs["result"] = json.dumps(kwargs["result"])
    cols = ", ".join(f"{k} = ?" for k in kwargs)
    vals = list(kwargs.values()) + [job_id]
    async with aiosqlite.connect(_DB_PATH) as db:
        await db.execute(f"UPDATE jobs SET {cols} WHERE job_id = ?", vals)
        await db.commit()


def make_job() -> ScrapeJob:
    """Factory: create a fresh queued ScrapeJob."""
    return ScrapeJob(
        job_id=str(uuid.uuid4()),
        status="queued",
        created_at=datetime.now(tz=timezone.utc).isoformat(),
    )


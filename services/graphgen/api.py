"""GraphGen FastAPI REST wrapper.

Exposes the KnowledgePipeline as an async HTTP service.

Endpoints
---------
GET  /health           — liveness probe
POST /build            — enqueue a pipeline run (returns immediately with job_id)
GET  /jobs/{job_id}    — poll job status / retrieve result

Usage (development)::

    PYTHONPATH=/path/to/repo/services/graphgen \\
        uvicorn graphgen.api:app --host 0.0.0.0 --port 8099 --reload

Or from the repo root with the installed package::

    uvicorn graphgen.api:app --port 8099
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import BackgroundTasks, FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(
    title="Graphknows GraphGen",
    version="1.0.0",
    description="REST API for the GraphGen knowledge-graph pipeline.",
)

# In-memory job store (sufficient for single-process deployments).
# Replace with Redis / a DB for multi-worker deployments.
_jobs: Dict[str, Dict[str, Any]] = {}


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class BuildRequest(BaseModel):
    """Parameters for a pipeline build job."""

    input_files: List[str] = []
    """File paths (inside the container / host) or a directory to process.
    When empty the directory configured via env / config.yaml is used."""

    config_override: Optional[Dict[str, Any]] = None
    """Optional flat key-value overrides applied on top of the loaded
    PipelineSettings before the pipeline starts."""

    output_dir: Optional[str] = None
    """Override the output directory for this run."""


class BuildJob(BaseModel):
    """Representation of a pipeline job."""

    job_id: str
    status: str  # queued | running | complete | failed
    created_at: str
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health", summary="Liveness probe")
async def health() -> Dict[str, str]:
    """Return a simple OK payload so load-balancers and k8s probes work."""
    return {"status": "ok", "service": "graphknows-graphgen"}


@app.post("/build", response_model=BuildJob, status_code=202, summary="Start a pipeline build")
async def build(req: BuildRequest, background_tasks: BackgroundTasks) -> BuildJob:
    """Enqueue a new knowledge-graph pipeline run.

    The pipeline runs asynchronously in the background.  Poll ``GET
    /jobs/{job_id}`` to check progress and retrieve the result once the status
    transitions to ``complete`` or ``failed``.
    """
    job_id = f"job_{uuid.uuid4().hex[:8]}"
    now = datetime.now(timezone.utc).isoformat()
    _jobs[job_id] = {"status": "queued", "created_at": now}

    background_tasks.add_task(_run_build, job_id, req)

    return BuildJob(job_id=job_id, status="queued", created_at=now)


@app.get("/jobs/{job_id}", response_model=BuildJob, summary="Poll job status")
async def get_job(job_id: str) -> BuildJob:
    """Return the current state of a previously submitted build job."""
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
    j = _jobs[job_id]
    return BuildJob(job_id=job_id, **j)


# ---------------------------------------------------------------------------
# Background worker
# ---------------------------------------------------------------------------


async def _run_build(job_id: str, req: BuildRequest) -> None:
    """Execute the pipeline in a thread pool and update the job store."""
    _jobs[job_id]["status"] = "running"
    try:
        from graphgen.orchestrator import run_pipeline  # noqa: PLC0415

        # run_pipeline is an async coroutine; run it directly.
        result = await run_pipeline(
            input_files=req.input_files,
            config_override=req.config_override or {},
            output_dir=req.output_dir,
        )
        _jobs[job_id].update({"status": "complete", "result": result})
    except Exception as exc:  # noqa: BLE001
        _jobs[job_id].update({"status": "failed", "error": str(exc)})

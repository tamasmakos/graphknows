"""GraphGen FastAPI REST wrapper.

Exposes the KnowledgePipeline as an async HTTP service.

Endpoints
---------
GET  /health           — liveness probe
POST /build            — enqueue a pipeline run (returns immediately with job_id)
GET  /jobs/{job_id}    — poll job status / retrieve result

Usage (development)::

    PYTHONPATH=/path/to/repo/services/graphknows \
        uvicorn graphgen.api:app --host 0.0.0.0 --port 8099 --reload

Or from the repo root with the installed package::

    uvicorn graphgen.api:app --port 8099
"""

from __future__ import annotations

import asyncio
import logging
import os
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# Load .env early so InfrastructureSettings and run_pipeline both see it.
# python-dotenv override=False means process env always wins over .env values.
try:
    from dotenv import load_dotenv
    _env_path = Path(__file__).parents[2] / ".env"  # services/graphknows/../../.env
    if not _env_path.exists():
        _env_path = Path("/root/graphknows/.env")
    load_dotenv(_env_path, override=True)
except ImportError:
    pass  # dotenv not installed — rely on env vars being set externally

from fastapi import BackgroundTasks, FastAPI, HTTPException
from graphknows.shared.config import BaseServiceSettings
from graphknows.shared.models.pipeline import Document
from pydantic import BaseModel, Field
from pythonjsonlogger import jsonlogger

log = logging.getLogger(__name__)


def _configure_logging() -> None:
    """Configure root logger for structured JSON output."""
    handler = logging.StreamHandler()
    handler.setFormatter(
        jsonlogger.JsonFormatter("%(asctime)s %(name)s %(levelname)s %(message)s")
    )
    logging.root.handlers = [handler]
    logging.root.setLevel(logging.INFO)

try:
    from graphgen.orchestrator import run_pipeline
except ImportError as exc:
    raise RuntimeError("graphgen orchestrator could not be imported") from exc

_configure_logging()

settings = GraphGenSettings() if False else None  # instantiated in lifespan below


@asynccontextmanager
async def _lifespan(application: FastAPI):  # noqa: ARG001
    """Configure logging and validate secrets on startup."""
    global settings
    settings = GraphGenSettings()
    settings.check_production_secrets()
    log.info("GraphGen service starting", extra={"env": settings.graphknows_env})
    yield


app = FastAPI(
    title="Graphknows GraphGen",
    version="1.0.0",
    description="REST API for the GraphGen knowledge-graph pipeline.",
    lifespan=_lifespan,
)


class GraphGenSettings(BaseServiceSettings):
    """GraphGen service settings."""

    graphgen_host: str = Field(default="0.0.0.0", alias="GRAPHGEN_HOST")
    graphgen_port: int = Field(default=8003, alias="GRAPHGEN_PORT")


# In-memory job store (sufficient for single-process deployments).
# Replace with Redis / a DB for multi-worker deployments.
_jobs: Dict[str, Dict[str, Any]] = {}


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------
#
# NOTE: We keep local BuildRequest / BuildJob instead of using the shared-py
# BuildJob because the shared model is designed for document-based ingestion
# (tracks documents_total / documents_done) whereas the graphgen REST API
# tracks pipeline runs with created_at and result payloads.  Unifying these
# two concepts is left for a future refactor.


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


@app.post("/ingest", response_model=BuildJob, status_code=202, summary="Ingest a document")
async def ingest_document(doc: Document, background_tasks: BackgroundTasks) -> BuildJob:
    """Accept a single document from the search service and queue it for KG ingestion.

    This is a fire-and-forget endpoint — returns a job_id immediately.
    The caller can poll GET /jobs/{job_id} for status.
    """
    job_id = f"job_{uuid.uuid4().hex[:8]}"
    now = datetime.now(timezone.utc).isoformat()
    _jobs[job_id] = {
        "status": "queued",
        "created_at": now,
        "documents_total": 1,
        "documents_done": 0,
    }
    background_tasks.add_task(_run_ingest, job_id, doc)
    return BuildJob(job_id=job_id, status="queued", created_at=now)


# ---------------------------------------------------------------------------
# Background worker
# ---------------------------------------------------------------------------


async def _run_ingest(job_id: str, doc: Document) -> None:
    """Process a single document ingestion in a thread pool and update the job store."""
    _jobs[job_id]["status"] = "running"
    try:
        # Convert Document to input the pipeline can use
        import tempfile
        import json

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(doc.model_dump(), f)
            tmp_path = f.name

        result = await run_pipeline(
            input_files=[tmp_path],
            config_override={},
            output_dir=None,
        )
        _jobs[job_id].update({"status": "complete", "result": result, "documents_done": 1})
    except Exception as exc:
        _jobs[job_id].update({"status": "failed", "error": str(exc)})


async def _run_build(job_id: str, req: BuildRequest) -> None:
    """Execute the pipeline in a thread pool and update the job store."""
    _jobs[job_id]["status"] = "running"
    try:
        # run_pipeline is an async coroutine imported at module level.
        result = await run_pipeline(
            input_files=req.input_files,
            config_override=req.config_override or {},
            output_dir=req.output_dir,
        )
        _jobs[job_id].update({"status": "complete", "result": result})
    except Exception as exc:  # noqa: BLE001
        _jobs[job_id].update({"status": "failed", "error": str(exc)})

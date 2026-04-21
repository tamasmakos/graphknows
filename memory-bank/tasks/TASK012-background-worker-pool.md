# TASK012 — Background Worker Pool + Job Store

**Status:** Pending  
**Added:** 2026-04-21  
**Updated:** 2026-04-21  
**Phase:** B — Ingestion Scale (requires TASK010 complete; TASK011 preferred)

---

## Original Request
`POST /documents` currently blocks the HTTP connection for the entire ingestion pipeline (parse → chunk → embed → write Neo4j). A 50-page PDF can hold the connection for 20–60 seconds, causing timeouts and poor UX. Replace with an async queue + worker pool inside the same process (no Redis/Celery), expose a `/jobs` endpoint family, and have `POST /documents` return immediately with a job ID.

---

## Thought Process
The simplest production-ready approach for a single-service, single-replica worker is:
- `asyncio.Queue` (bounded, e.g. maxsize=100) as the job inbox
- `asyncio.Semaphore` to cap concurrent pipeline runs (default=4, env-configurable)
- One or more `worker` coroutines started in lifespan via `asyncio.create_task`
- An in-memory `dict[job_id, JobRecord]` as the job store (survives restarts if we add TASK013's recovery step)
- SSE `/jobs/{id}/stream` endpoint for real-time progress

In-process is the right call here (per user's decision): no Redis, no Celery, no separate worker container. Scale is a few dozen concurrent docs at most.

### Why not FastAPI `BackgroundTasks`?
`BackgroundTasks` runs after the response is sent in the same thread pool. It doesn't support backpressure (no queue size limit), has no progress tracking, and can't be cancelled. The `asyncio.Queue` approach gives us all of those.

---

## Implementation Plan

### New Files
1. **`services/graphgen/src/kg/jobs/__init__.py`** — empty
2. **`services/graphgen/src/kg/jobs/models.py`**:
   - `JobStatus = Literal["queued", "processing", "done", "error"]`
   - `JobRecord(BaseModel)`: `job_id: str`, `doc_id: str | None`, `status: JobStatus`, `progress: float (0–1)`, `created_at: datetime`, `updated_at: datetime`, `error: str | None`, `result: dict | None`
3. **`services/graphgen/src/kg/jobs/registry.py`**:
   - `_registry: dict[str, JobRecord] = {}`
   - `create_job(doc_id) -> JobRecord`
   - `get_job(job_id) -> JobRecord | None`
   - `list_jobs() -> list[JobRecord]`
   - `update_job(job_id, **kwargs) -> JobRecord`
4. **`services/graphgen/src/kg/jobs/queue.py`**:
   - `job_queue: asyncio.Queue[str]` (stores `job_id`)
   - `worker_semaphore: asyncio.Semaphore`
   - `enqueue(job_id: str) -> None` — raises `QueueFullError` if full
5. **`services/graphgen/src/kg/jobs/worker.py`**:
   - `async def run_worker(n: int) -> None:` — infinite loop: `job_id = await queue.get()` → acquire semaphore → run pipeline → update registry → release semaphore → `queue.task_done()`
   - Progress callback: `lambda frac: update_job(job_id, progress=frac, updated_at=now())`

### Modified Files
6. **`services/graphgen/src/main.py`** (lifespan):
   - After `yield`, spawn `WORKER_CONCURRENCY` worker tasks: `tasks = [asyncio.create_task(run_worker(i)) for i in range(settings.worker_concurrency)]`
   - On shutdown: cancel tasks, drain queue.
7. **`services/graphgen/src/main.py`** (`POST /documents`):
   - Parse the upload and create a `Document` node (just metadata, no embedding yet).
   - Create a `JobRecord` via `create_job(doc_id)`.
   - Enqueue `job_id`.
   - Return `{"job_id": ..., "doc_id": ..., "status": "queued"}` with `HTTP 202`.
8. **`services/graphgen/src/kg/pipeline/core.py`**:
   - Refactor `run_pipeline()` to accept `(parsed_doc: ParsedDocument, progress_callback: Callable[[float], None]) -> PipelineResult`.
   - Emit progress at: `0.1` (parsed), `0.3` (chunks created), `0.6` (embeddings done), `0.9` (graph written), `1.0` (done).
   - Batch embeddings: 64 docs at a time.

### New Endpoints
9. **`GET /jobs`** — list all `JobRecord`s (optionally filter by `status=`)
10. **`GET /jobs/{job_id}`** — single job detail
11. **`GET /jobs/{job_id}/stream`** — SSE stream; emits `{"type":"progress","progress":0.3}` events until done or error; clients should close connection after `done` event.

---

## Progress Tracking

**Overall Status:** Not Started — 0%

### Subtasks
| ID | Description | Status | Updated | Notes |
|----|-------------|--------|---------|-------|
| 12.1 | Create kg/jobs/ module (models, registry, queue, worker) | Not Started | 2026-04-21 | |
| 12.2 | Refactor pipeline/core.py to accept progress_callback | Not Started | 2026-04-21 | |
| 12.3 | Batch embeddings 64/batch | Not Started | 2026-04-21 | |
| 12.4 | Wire worker tasks in lifespan (start + shutdown) | Not Started | 2026-04-21 | |
| 12.5 | Change POST /documents to 202 + job_id | Not Started | 2026-04-21 | |
| 12.6 | Add GET /jobs, GET /jobs/{id}, GET /jobs/{id}/stream | Not Started | 2026-04-21 | SSE for stream |
| 12.7 | Write unit tests for queue + registry | Not Started | 2026-04-21 | pytest, no Neo4j needed |
| 12.8 | Write integration test for full POST /documents → job done | Not Started | 2026-04-21 | testcontainers Neo4j |

## Progress Log
### 2026-04-21
- Task created from audit findings. Current blocking behavior confirmed: pipeline runs in-request.

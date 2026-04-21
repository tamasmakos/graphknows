# TASK013 — Idempotency + Dedup + Resume

**Status:** Pending  
**Added:** 2026-04-21  
**Updated:** 2026-04-21  
**Phase:** B — Ingestion Scale (requires TASK012 complete)

---

## Original Request
Currently re-ingesting the same file creates duplicate `Document` nodes and doubles `Chunk` + `Entity` sets. The system also has no concept of resumable jobs — if graphgen restarts mid-ingestion, the document stays in a "processing" state forever. Fix both: content-hash deduplication and startup recovery.

---

## Thought Process
Two separate problems:
1. **Dedup**: Use `sha256(raw_bytes)` as the MERGE key for `Document`. If the hash already exists and the doc is `status:complete`, return the existing job ID immediately. If `?force=true`, mark the existing doc for reprocessing and create a new job.
2. **Resume**: On startup, scan Neo4j for `Document` nodes with `status:processing`. These were interrupted by a crash. Set them to `status:error` with `resumable:true`. The `/jobs/{id}/retry` endpoint re-enqueues them.

Stable chunk IDs (`{doc_id}:{position:04d}`) let us MERGE on chunk position during re-ingestion rather than creating duplicates. Entity MERGE is already correct if `entity_id` is deterministic (name + type normalized).

---

## Implementation Plan

1. **`content_hash` field on Document node** — computed before any pipeline work:
   ```python
   content_hash = hashlib.sha256(raw_bytes).hexdigest()
   ```
   Add `content_hash` to `schema_bootstrap.py` UNIQUE constraint (or at minimum a single-property index).

2. **Dedup check in `POST /documents`**:
   ```python
   existing = await neo4j.find_document_by_hash(content_hash)
   if existing and existing.status == "complete" and not force:
       return {"job_id": existing.last_job_id, "doc_id": existing.doc_id, "status": "duplicate"}
   ```

3. **`?force=true` override** — skip dedup check; useful for testing and reprocessing after schema changes.

4. **Stable `chunk_id`**:
   ```python
   chunk_id = f"{doc_id}:{position:04d}"
   ```
   Use `MERGE (c:Chunk {chunk_id: $chunk_id})` so re-ingestion updates rather than duplicates.

5. **Startup recovery in lifespan** (runs before workers start):
   ```python
   async def recover_interrupted_jobs(driver):
       """Set interrupted documents back to error+resumable so they can be retried."""
       await driver.execute_query(
           "MATCH (d:Document {status:'processing'}) "
           "SET d.status='error', d.resumable=true, d.error='interrupted by restart'"
       )
   ```

6. **`POST /jobs/{id}/retry` endpoint**:
   - Checks `job.resumable == true`.
   - Resets `status` to `queued`.
   - Re-enqueues `job_id`.
   - Returns `{"job_id": id, "status": "queued"}`.

7. **Frontend visibility** (deferred to TASK019): show "duplicate" badge in the documents table; show "retry" button on errored jobs.

---

## Progress Tracking

**Overall Status:** Not Started — 0%

### Subtasks
| ID | Description | Status | Updated | Notes |
|----|-------------|--------|---------|-------|
| 13.1 | Add content_hash index to schema_bootstrap.py | Not Started | 2026-04-21 | Update TASK011 output |
| 13.2 | Compute sha256 before ingestion in POST /documents | Not Started | 2026-04-21 | |
| 13.3 | Dedup check + ?force=true override | Not Started | 2026-04-21 | |
| 13.4 | Stable chunk_id = {doc_id}:{position:04d} | Not Started | 2026-04-21 | |
| 13.5 | MERGE on chunk_id in pipeline/core.py | Not Started | 2026-04-21 | |
| 13.6 | Startup recovery coroutine in lifespan | Not Started | 2026-04-21 | |
| 13.7 | POST /jobs/{id}/retry endpoint | Not Started | 2026-04-21 | |
| 13.8 | Unit tests for dedup logic | Not Started | 2026-04-21 | |

## Progress Log
### 2026-04-21
- Task created. Duplicate-node problem confirmed during audit — no MERGE on doc hash, no stable chunk_id.

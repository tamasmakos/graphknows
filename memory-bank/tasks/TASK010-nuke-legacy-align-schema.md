# TASK010 — Nuke Legacy & Align Schema

**Status:** Pending  
**Added:** 2026-04-21  
**Updated:** 2026-04-21  
**Phase:** A — Cleanup & Unification (must be first)

---

## Original Request
Extensive audit (2026-04-21) revealed two competing codepaths in graphgen (modern `/documents` vs legacy `/run`+LifeLogParser), dead FalkorDB code across both services, a mismatched schema (DOCUMENT/CHUNK vs DAY/SEGMENT/EPISODE), a wrong embedding model default, and a missing runtime dep (pytesseract). This task eliminates all of it so Phase B–F can build on a single clean foundation.

---

## Thought Process
The audit revealed the core problem: tasks TASK002 ("delete dead code") and TASK006 ("Neo4j uploader") were only partially executed. The FalkorDB surface is deeper than expected — it persists in graphrag's `workflow/`, `infrastructure/`, and `services/` directories, and the graphgen `/run` endpoint still calls the stub `LifeLogParser`. Nothing in Phase B–F is safe to build until these competing paths are gone. This is a pure deletion + fixup task — zero new features.

Key constraint: `kg/graph/extraction.py` references DAY/SEGMENT/EPISODE/TOPIC in the `build_lexical_graph()` function called by `/run`. That entire function and its supporting code must be removed; the `/run` endpoint should be rewired to iterate over `input/` using `ParserRegistry` + `KnowledgePipeline.process_document()`. This aligns `/run` with `/documents POST` — one path, one schema.

---

## Implementation Plan

### graphgen deletions
1. Delete `services/graphgen/src/kg/graph/parsers/` (entire folder: `base.py`, `life.py`, `__init__.py`).
2. Delete `services/graphgen/src/kg/utils/health.py` (imports FalkorDB, unused).
3. Delete `services/graphgen/src/kg/graph/parsing.py` (stub: only contains `SegmentData` model).
4. Remove `falkordb` from `services/graphgen/requirements.txt` line 7.
5. In `services/graphgen/src/kg/graph/extraction.py`: remove `build_lexical_graph()`, `process_single_document_lexical()`, `add_segments_to_graph()` and all DAY/SEGMENT/EPISODE/TOPIC/SUBTOPIC references. Retain entity extraction, resolution, embeddings.
6. Rewrite `/run POST` in `services/graphgen/src/main.py` to: enumerate `input/` dir, call `get_parser(file_ext)` → `parse()` → enqueue to pipeline (or call `process_document` directly), return job list.

### graphgen fixes
7. Add `pytesseract` to `[project.dependencies]` in `services/graphgen/pyproject.toml`.
8. Change default embedding model from `all-MiniLM-L6-v2` to `BAAI/bge-small-en-v1.5` in `services/graphgen/src/kg/embeddings/model.py` (update the default value in `AppSettings` or the model singleton).

### graphrag deletions
9. Delete `services/graphrag/src/workflow/` (entire folder: `graph_workflow.py` + any step files).
10. Delete `services/graphrag/src/infrastructure/graph_db.py` (FalkorDBDB implementation).
11. Delete `services/graphrag/src/infrastructure/postgres_store.py` (unused by agent path — confirm no imports from `agent/tools.py` first).
12. Delete `services/graphrag/src/services/graph_retriever.py` and `services/graphrag/src/services/context_builder.py` (belong to old GraphWorkflow).
13. Remove `falkordb` and any `langchain-community` FalkorDB imports from `services/graphrag/pyproject.toml` and `requirements.txt`.

### graphrag fixes
14. Remove FalkorDB-related fields from `services/graphrag/src/common/config/settings.py` (e.g., `falkordb_host`, `falkordb_port`).
15. Confirm `services/graphrag/src/infrastructure/neo4j_driver.py` is the only DB client now — verify it's used in `main.py` lifespan.

### Documentation
16. Update `services/graphgen/src/kg/KG_README.md` — remove FalkorDB + DAY/SEGMENT/EPISODE references; describe actual DOCUMENT/CHUNK/ENTITY schema.
17. Update root `README.md` — fix port reference (web is 3000 or 3001, not 8010); remove FalkorDB mentions.
18. Update `services/graphrag/src/APP_README.md` — remove references to non-existent `llamaindex_agent.py`.

---

## Progress Tracking

**Overall Status:** Not Started — 0%

### Subtasks
| ID | Description | Status | Updated | Notes |
|----|-------------|--------|---------|-------|
| 10.1 | Delete kg/graph/parsers/ folder | Not Started | 2026-04-21 | 3 files |
| 10.2 | Delete kg/utils/health.py | Not Started | 2026-04-21 | FalkorDB import |
| 10.3 | Delete kg/graph/parsing.py | Not Started | 2026-04-21 | stub only |
| 10.4 | Remove falkordb from graphgen requirements.txt | Not Started | 2026-04-21 | |
| 10.5 | Strip DAY/SEGMENT/EPISODE from extraction.py | Not Started | 2026-04-21 | Keep entity extraction |
| 10.6 | Rewrite /run endpoint to use ParserRegistry | Not Started | 2026-04-21 | |
| 10.7 | Add pytesseract to graphgen pyproject.toml | Not Started | 2026-04-21 | Runtime crash fix |
| 10.8 | Fix embedding model default to bge-small-en-v1.5 | Not Started | 2026-04-21 | Same 384-dim |
| 10.9 | Delete graphrag workflow/ folder | Not Started | 2026-04-21 | GraphWorkflow + steps |
| 10.10 | Delete graphrag infrastructure/graph_db.py | Not Started | 2026-04-21 | FalkorDBDB |
| 10.11 | Delete graphrag infrastructure/postgres_store.py | Not Started | 2026-04-21 | Confirm no refs first |
| 10.12 | Delete graphrag services/graph_retriever.py + context_builder.py | Not Started | 2026-04-21 | |
| 10.13 | Remove FalkorDB deps from graphrag pyproject.toml | Not Started | 2026-04-21 | |
| 10.14 | Remove FalkorDB fields from graphrag settings.py | Not Started | 2026-04-21 | |
| 10.15 | Update KG_README.md | Not Started | 2026-04-21 | |
| 10.16 | Update root README.md (port fix + FalkorDB) | Not Started | 2026-04-21 | |
| 10.17 | Update APP_README.md | Not Started | 2026-04-21 | |

## Verification
- `grep -r "falkordb\|FalkorDB\|LifeLog\|SEGMENT\|EPISODE\|DAY\|TOPIC\|SUBTOPIC" services/` returns zero hits.
- `uv run python -c "from kg.parser.image import ImageParser"` in graphgen venv succeeds (pytesseract available).
- `POST /run` with a PDF in `input/` creates `Document → Chunk` nodes in Neo4j (not DAY/SEGMENT nodes).
- `POST /documents` still works unchanged.

## Progress Log
### 2026-04-21
- Task created based on comprehensive audit findings.
- Identified 17 subtasks across graphgen and graphrag.

# Progress

_Last updated: 2026-04-22_

## Overall Status
**Phase A (Cleanup & Unification) complete. Codebase is clean: single Neo4j codepath, no FalkorDB/pgvector/postgres, no legacy DAY/SEGMENT/EPISODE schema. Ready for Phase B.**

| Phase | Description | Status | % |
|-------|-------------|--------|---|
| Phase A | Cleanup & Unification (TASK010 + TASK011) | ✅ Complete | 100% |
| Phase B | Ingestion Scale (TASK012 + TASK013) | ⏳ Pending | 0% |
| Phase C | Agent Quality (TASK014 + TASK015 + TASK016) | ⏳ Pending | 0% |
| Phase D | Frontend Rebuild (TASK017–TASK021) | ⏳ Pending | 0% |
| Phase E | CI/CD + Tests (TASK022) | ⏳ Pending | 0% |
| Phase F | Observability + Evals (TASK023 + TASK024) | ⏳ Pending | 0% |

## What's Working (post-Phase A)
- **graphgen**: FastAPI on :8020 — 8 parsers (TXT/MD/PDF/DOCX/PPTX/XLSX/HTML/Image), GLiNER NER, Neo4j uploader, schema bootstrapped at startup
- **graphrag**: FastAPI on :8010 — LlamaIndex ReActAgent, vector+fulltext search tools, Neo4j only
- **MCP server**: Internal dev tool — rewired to `run_agent()` and raw Neo4j Cypher
- **docker-compose**: Neo4j + graphgen + graphrag + Langfuse (observability profile); pgvector removed

## Known Gaps (Phase B targets)
- No background job queue — `/run` is synchronous and blocking
- No idempotency — re-ingesting same document creates duplicates
- No `/jobs` endpoint for progress tracking

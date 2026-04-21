# Active Context

_Last updated: 2026-04-22_

## Current Focus
Phase A (Cleanup & Unification) is **complete**. Both TASK010 and TASK011 are done.

## What Was Just Completed

### TASK010 — Nuke Legacy & Align Schema ✅
- Deleted all FalkorDB/pgvector/LifeLogParser/DAY/SEGMENT/EPISODE code
- Rewired `/run` to use ParserRegistry
- Fixed embedding model to BAAI/bge-small-en-v1.5
- Updated KG_README.md, APP_README.md, root README.md

### TASK011 — Schema Bootstrap + MCP Alignment ✅
- Created `kg/neo4j/schema_bootstrap.py` — idempotent constraints + vector + fulltext indexes
- Wired bootstrap_schema into graphgen lifespan
- Added index verification (SHOW INDEXES) to graphrag lifespan
- Rewired MCP server: `kg_chat` → `run_agent()`, `kg_schema` → `create_driver()` + raw Cypher
- Removed postgres fields from graphrag settings and both requirements
- Moved pgvector service in docker-compose behind `legacy-pgvector` profile (effectively disabled)
- Updated `.env.example` — removed POSTGRES_* vars, added NEO4J_AUTH

## Architecture State (post-cleanup)
- **Vector store**: Neo4j Community 5.11+ (unlimited vector indexes)
- **Embedding model**: BAAI/bge-small-en-v1.5 (384-dim cosine)
- **NER backend**: GLiNER (default)
- **Schema**: Document→[:CONTAINS]→Chunk→[:MENTIONS]→Entity→[:RELATED_TO]→Entity
- **Indexes**: chunk_embedding, entity_embedding (vector); chunk_text_fulltext, entity_name_fulltext (fulltext)
- **Agent**: LlamaIndex ReActAgent via `run_agent()` / `stream_agent()`

## Next Up
Phase B — Ingestion Scale:
- **TASK012**: Background Worker Pool + Job Store
- **TASK013**: Idempotency, Dedup, Resume

# Progress

## Overall Status
**Phase: Planning complete → Implementation starting**

| Phase | Description | Status | % |
|-------|-------------|--------|---|
| Phase 1a | Root tooling scaffold (Turborepo, pnpm, turbo.json) | Not Started | 0% |
| Phase 1b | Resolve merge conflicts + delete dead code | Not Started | 0% |
| Phase 1c | docker-compose cleanup + dev overlay | Not Started | 0% |
| Phase 1d | .dockerignore, .env.example, .gitignore | Not Started | 0% |
| Phase 2 | DocumentParser abstraction (graphgen) | Not Started | 0% |
| Phase 3 | Schema.py + plugin system + Neo4j uploader | Not Started | 0% |
| Phase 4 | AgentWorkflow rewrite (graphrag) | Not Started | 0% |
| Phase 5 | Next.js 15 frontend (apps/web) | Not Started | 0% |
| Phase 6 | Dockerfile upgrades + CONTRIBUTING.md | Not Started | 0% |

## What Works (Current Codebase)
- `uv` workspace configured in root `pyproject.toml` (members: services/graphgen, services/graphrag).
- graphgen: full ETL pipeline runs end-to-end for CSV life-log inputs (not useful post-refactor but functional).
- graphrag: linear retrieval chain answers queries against FalkorDB.
- docker-compose.yaml brings up FalkorDB, pgvector, langfuse, graphgen, graphrag.
- Healthchecks exist on FalkorDB and pgvector in compose.

## What Doesn't Work / Is Broken
- 3 files have unresolved `<<<<<<< HEAD` merge conflict markers.
- No `.dockerignore` — Docker builds slow and may include secrets.
- `neo4j` service in compose is unused (no code targets it).
- `frontend/` is static HTML+D3 — not a Next.js app, no TS, no component library.
- Python BFF at `frontend/backend/` runs on port 8001 — not integrated with compose.
- No streaming (SSE) in graphrag responses.
- No structured citations in graphrag responses.
- No document ingestion UI (only file drop into Docker volume).

## What's Left to Build (Full List)
- [ ] Turborepo + pnpm workspace configuration
- [ ] `apps/web/` Next.js 15 app (all 4 views + BFF Route Handlers)
- [ ] `packages/ui/` shared component library
- [ ] `packages/types/` OpenAPI type generation
- [ ] graphgen: `kg/parser/` module (8 format parsers + HeadingAwareChunker)
- [ ] graphgen: `kg/schema.py` + `kg/plugins/` system
- [ ] graphgen: `kg/neo4j/` uploader (replaces `kg/falkordb/`)
- [ ] graphgen: new REST endpoints (`/documents`, `/analytics`, `/health`)
- [ ] graphrag: `agent/` module (workflow + 4 tools + decomposer + sufficiency)
- [ ] graphrag: `models/` (AgentResponse, Citation, ToolCall, ReasoningStep)
- [ ] graphrag: SSE streaming endpoint
- [ ] graphrag: Neo4j client + Cypher rewrite (DOCUMENT/CHUNK/ENTITY schema)
- [ ] graphrag: per-tool Langfuse spans
- [ ] Both Dockerfiles: multi-stage, non-root, pinned, HEALTHCHECK
- [ ] docker-compose.yaml: Neo4j, pinned images, healthchecks, `web` service
- [ ] docker-compose.dev.yaml: bind mounts + --reload overlays
- [ ] `.dockerignore`, `.env.example`, updated `.gitignore`
- [ ] `CONTRIBUTING.md` with 3 extension-point recipes

## Known Issues (to fix as part of refactor)
1. Merge conflict markers in graphgen/main.py, graphrag/main.py, frontend/static/script.js.
2. FalkorDB used as primary graph DB — needs full Neo4j migration.
3. `community/subcommunities.py` may reference SEGMENT/EPISODE — audit required.
4. Embeddings stored using `all-MiniLM-L6-v2`; will change to `BAAI/bge-small-en-v1.5` (same dim, requires re-embed of any existing data).
5. LangChain `LLMGraphTransformer` prompt is life-log specific — needs generic replacement.

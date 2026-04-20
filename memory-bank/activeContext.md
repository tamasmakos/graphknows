# Active Context

## Current Focus
**Planning complete. Beginning implementation.** The full architecture plan has been reviewed and approved. The locked decisions are confirmed. Task files have been created for all 6 phases. Implementation starts with Phase 1 (Monorepo Restructure).

## Locked Decisions
| Concern | Decision |
|---------|---------|
| Graph DB | Neo4j only (FalkorDB deleted) |
| BFF | Next.js Route Handlers in `apps/web/app/api/v1/` |
| Streaming | SSE via POST + `fetch`/`ReadableStream` |
| Legacy: keep | Community detection (Leiden), entity resolution, pruning, MCP server, GLiNER |
| Legacy: drop | `src/simulation/`, `LifeLogParser`, `DAY/SEGMENT/EPISODE/TOPIC/SUBTOPIC` schema |
| Compatibility | Clean break — no deprecation shims |
| Embedding model | `BAAI/bge-small-en-v1.5` (384-dim, same dim as MiniLM-L6) |
| Conversations | SQLite in Docker volume (web app owns it) |
| Auth | Stub middleware only (no auth in MVP) |

## Recent Changes
- Architecture plan drafted (all 6 phases, file-level change lists, interface contracts).
- Memory Bank initialized (projectbrief, productContext, techContext, systemPatterns, activeContext, progress, tasks/).

## Next Steps (in order)
1. **TASK001** — Phase 1a: Root tooling scaffold (Turborepo, pnpm workspace, `turbo.json`, `pnpm-workspace.yaml`, root `package.json`).
2. **TASK002** — Phase 1b: Resolve merge conflicts + delete dead code.
3. **TASK003** — Phase 1c: `docker-compose.yaml` cleanup + `docker-compose.dev.yaml` creation.
4. **TASK004** — Phase 1d: `.dockerignore`, `.env.example`, update `.gitignore`.
5. **TASK005** — Phase 2: graphgen DocumentParser abstraction.
6. **TASK006** — Phase 3: Schema.py + plugin system + Neo4j uploader.
7. **TASK007** — Phase 4: graphrag AgentWorkflow rewrite.
8. **TASK008** — Phase 5: Next.js 15 frontend (apps/web).
9. **TASK009** — Phase 6: Dockerfile upgrades + CONTRIBUTING.md.

## Active Decisions Under Consideration
- **Neo4j Community Edition vector index limit:** Neo4j Community 5.x supports unlimited vector indexes since 5.11 (the restriction was lifted). Confirmed safe.
- **pgvector stays** as the preferred CHUNK hybrid-search backend. Neo4j vector index used for ENTITY search. Both receive embeddings at ingest time.
- **Conversation history**: SQLite with `better-sqlite3` in web app. Volume mount `/data/conversations.sqlite`.

## Patterns to Follow Consistently
- Python service Pydantic models use `model_config = ConfigDict(extra='ignore')`.
- All FastAPI apps use lifespan context managers (not `@app.on_event`).
- Neo4j queries use parameterized Cypher only (no f-string interpolation into Cypher).
- All async Python code uses `async with driver.session() as session` — never share sessions.
- TypeScript: use `satisfies` not `as` for type assertions; avoid `any`.
- Route Handlers: always return typed `NextResponse` with explicit status codes.

# Tech Context

## Repository Structure (Actual — Audited 2026-04-21)
```
graphknows/
├── apps/
│   └── web/                        # Next.js 15 App Router frontend + BFF (partial)
├── services/
│   ├── graphgen/                   # ETL pipeline — DUAL PIPELINE, needs Phase A cleanup
│   └── graphrag/                   # Agentic RAG — DUAL WORKFLOW, needs Phase A cleanup
├── packages/
│   ├── ui/                         # ⚠️ PLACEHOLDER ONLY — no components yet
│   └── types/                      # ✅ Complete hand-written types (not yet auto-generated)
├── scripts/
│   └── generate-types.sh           # ⚠️ Requires running services — not CI-safe
├── turbo.json                      # ✅ Complete
├── pnpm-workspace.yaml             # ✅ Complete
├── pyproject.toml                  # ✅ uv workspace root
├── uv.lock
├── docker-compose.yaml             # ✅ Neo4j 5.26, pgvector, graphgen, graphrag, Langfuse, web
├── docker-compose.dev.yaml         # ✅ bind-mounts + --reload overlays
├── .env.example                    # ✅ Complete
└── .dockerignore                   # ✅ Complete (root only; no per-service)
```

## Technology Stack

### Python Services (graphgen + graphrag)
| Layer | Technology | Target | Actual (Audited) |
|-------|-----------|--------|------------------|
| Runtime | Python 3.11 | pinned | ✅ pinned |
| Web framework | FastAPI | both services | ✅ both |
| Package manager | uv + uv workspaces | | ✅ configured |
| Graph DB client | neo4j ≥5.x async | replaces falkordb | ✅ in graphgen; ✅ in graphrag; ⚠️ FalkorDB still in graphrag workflow/ |
| Vector store | pgvector (Postgres 16) | HNSW index | ⚠️ Present in infra, effectively bypassed in agent path |
| LLM orchestration | LangChain (graphgen) + LlamaIndex (graphrag) | | ✅ Both present |
| Entity extraction | GLiNER `urchade/gliner_medium-v2.1` + LLMGraphTransformer | | ✅ Working |
| Embeddings | `BAAI/bge-small-en-v1.5` (384-dim) | locked decision | ⚠️ Code still defaults to `all-MiniLM-L6-v2` — **fix in TASK010** |
| Community detection | leidenalg + igraph | | ✅ Working |
| Document parsing | pymupdf4llm, python-docx, python-pptx, openpyxl, trafilatura, pytesseract | | ✅ All implemented; ⚠️ `pytesseract` missing from pyproject.toml — **add in TASK010** |
| Chunking | heading-aware markdown chunker | | ✅ HeadingAwareChunker in `kg/parser/markdown.py` |
| Streaming | sse-starlette | SSE from FastAPI | ✅ graphrag /chat; ⚠️ missing events (reasoning/tool_call/tool_result/error) |
| Observability | Langfuse + OpenTelemetry | global + per-tool | ✅ global; ⚠️ no per-tool spans yet |
| MCP | FastMCP | agent tools | ⚠️ Wired to wrong workflow (FalkorDB GraphWorkflow) |
| Config | pydantic-settings | .env loading | ✅ Both services |
| NLP | spaCy `en_core_web_lg` | | ⚠️ Baked into graphgen Docker image — ~40MB dead weight (GLiNER replaces it). **Remove in TASK023** |

### TypeScript / Frontend (apps/web + packages)
| Layer | Technology | Target | Actual (Audited) |
|-------|-----------|--------|------------------|
| Framework | Next.js 15 (App Router) | standalone output | ✅ Running |
| Language | TypeScript 5.x | strict | ✅ strict mode |
| Package manager | pnpm 9.x | | ✅ pinned |
| Monorepo | Turborepo | | ✅ wired |
| UI components | shadcn/ui + Kibo UI | | ❌ **NOT installed — zero component library** |
| Styling | Tailwind CSS 3.x | | ✅ installed; ⚠️ mixed with raw `style={{var(--surface)}}` throughout |
| State management | TanStack Query v5 + Zustand | | ❌ not yet installed |
| Graph visualization | react-force-graph-2d | retain | ✅ installed and working |
| SSE consumption | fetch + ReadableStream | POST-based | ✅ working in /chat |
| Type generation | openapi-typescript | from OpenAPI specs | ⚠️ script exists but not CI-safe |
| Charts | recharts | analytics page | ❌ not yet installed |
| Themes | next-themes | light/dark toggle | ❌ not yet installed |
| Toast | sonner | global notifications | ❌ not yet installed |
| Conversation store | better-sqlite3 | SQLite volume | ❌ not yet installed |
| Table | @tanstack/react-table | documents DataTable | ❌ not yet installed |

### Infrastructure (docker-compose.yaml — Audited)
| Service | Image | Port | Healthcheck |
|---------|-------|------|-------------|
| neo4j | `neo4j:5.26-community` | 7474, 7687 | ✅ wget |
| pgvector | `pgvector/pgvector:pg16` | 5432 | ✅ pg_isready |
| graphgen | custom Python build | 8020 | ❌ missing in compose (✅ in Dockerfile) |
| graphrag | custom Python build | 8010 | ❌ missing in compose (✅ in Dockerfile) |
| langfuse | `langfuse/langfuse:2.95.0` | 3000 | ❌ missing |
| langfuse-db | `postgres:16.6-alpine` | internal | ✅ pg_isready |
| web | custom Next.js build | 3000 | ✅ /api/v1 check |

## Development Setup

### Prerequisites
- Docker Desktop (or Rancher Desktop)
- pnpm 9.x (`npm install -g pnpm`)
- uv (`curl -LsSf https://astral.sh/uv/install.sh | sh`)
- Node.js 20.x LTS

### Bootstrap
```bash
cp .env.example .env           # fill in LLM_API_KEY at minimum
pnpm install                   # installs all TS workspaces
uv sync                        # installs all Python workspaces
docker compose up -d           # starts infra + services
```

### Environment Variable Convention
```
LLM_*          # shared: provider (groq|openai), api keys, model names
GRAPHGEN_*     # ETL service config (concurrency, paths)
GRAPHRAG_*     # agent service config (max_iterations, reranker model)
WEB_*          # Next.js app config (NEXT_PUBLIC_* for client-side only)
NEO4J_*        # graph DB URI, user, password
POSTGRES_*     # pgvector connection (if retained)
LANGFUSE_*     # public key, secret key, host
```

## Dependency Issues Found (Audit 2026-04-21)

### graphgen
- `falkordb` still in `requirements.txt` (not in `pyproject.toml`) → dead. **Remove in TASK010.**
- `pytesseract` NOT in `pyproject.toml` but imported in `kg/parser/image.py` → runtime crash. **Add in TASK010.**
- `psycopg2-binary` + `pgvector` in deps but unused if pgvector is dropped. **Audit in TASK011.**
- spaCy baked into Dockerfile → ~40MB dead weight. **Remove in TASK023 (or TASK010 for quick win).**

### graphrag
- `falkordb` (Python) used by `workflow/graph_workflow.py` and `infrastructure/graph_db.py` → both being deleted in TASK010.
- `langchain-core` imported but only used by old GraphWorkflow → can likely drop after cleanup.
- Both LangChain + LlamaIndex in the same service → post-cleanup LlamaIndex only should remain.

### apps/web
Packages to add for Phase D:
```
shadcn/ui (via cli: pnpm dlx shadcn@latest init)
Kibo UI components (via registry: npx shadcn@latest add https://www.kibo-ui.com/registry/<component>.json)
next-themes
sonner
@tanstack/react-table
recharts
better-sqlite3 (+ @types/better-sqlite3)
```

## Technical Constraints
- **Neo4j Community 5.11+**: unlimited vector indexes per DB (restriction lifted). Confirmed safe.
- **GLiNER + SentenceTransformers**: require torch — Docker image ~2GB. GPU optional (auto-detect via `device` param).
- **SSE via fetch POST**: `EventSource` is GET-only; the frontend must use `fetch` + `ReadableStream` for POST-based SSE. Already implemented correctly in the chat page.
- **bge-small-en-v1.5 → same 384-dim as MiniLM-L6-v2**: Neo4j vector indexes will survive the model swap without re-creation. Existing stored embeddings will be semantically stale and need a re-embed of any previously ingested content.
- **react-force-graph-2d**: requires `dynamic(..., { ssr: false })` in Next.js due to canvas/browser dependency. Already applied in `GraphVisualizer.tsx`.
- **better-sqlite3**: native addon — needs `node-gyp` build step. Use in a Route Handler (server-only), never in client components.

## What Was Removed (or Needs Removing)
| Item | Status | Action |
|------|--------|--------|
| FalkorDB Docker service | ✅ Removed from compose | — |
| FalkorDB Python client (graphgen) | ⚠️ Still in requirements.txt | TASK010 |
| FalkorDB Python client (graphrag) | ⚠️ Still in workflow/ + infrastructure/ | TASK010 |
| `frontend/` static HTML+D3 | ✅ Replaced by apps/web | — |
| `frontend/backend/` FastAPI BFF | ✅ Replaced by Route Handlers | — |
| Merge conflict markers | ✅ Resolved | — |
| LifeLogParser + kg/graph/parsers/ | ⚠️ Still present | TASK010 |
| DAY/SEGMENT/EPISODE/TOPIC/SUBTOPIC schema in extraction.py | ⚠️ Still present in `/run` path | TASK010 |
| kg/utils/health.py (FalkorDB import) | ⚠️ Still present | TASK010 |
| kg/graph/parsing.py (stub) | ⚠️ Still present | TASK010 |
| workflow/graph_workflow.py (FalkorDB) | ⚠️ Still present in graphrag | TASK010 |
| infrastructure/graph_db.py (FalkorDBDB) | ⚠️ Still present in graphrag | TASK010 |
| services/graph_retriever.py + context_builder.py | ⚠️ Still present in graphrag | TASK010 |

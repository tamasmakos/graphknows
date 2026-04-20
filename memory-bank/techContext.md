# Tech Context

## Repository Structure (Target)
```
graphknows/
├── apps/
│   └── web/                        # Next.js 15 App Router frontend + BFF
├── services/
│   ├── graphgen/                   # ETL pipeline (FastAPI + Python)
│   └── graphrag/                   # Agentic RAG service (FastAPI + Python)
├── packages/
│   ├── ui/                         # Shared shadcn/ui + Kibo UI + Tailwind preset
│   └── types/                      # Auto-generated TypeScript types from OpenAPI
├── scripts/                        # Dev helpers (type gen, seed)
├── turbo.json
├── pnpm-workspace.yaml
├── pyproject.toml                  # uv workspace root
├── uv.lock
├── docker-compose.yaml
├── docker-compose.dev.yaml
├── .env.example
└── .dockerignore
```

## Technology Stack

### Python Services (graphgen + graphrag)
| Layer | Technology | Notes |
|-------|-----------|-------|
| Runtime | Python 3.11 | pinned |
| Web framework | FastAPI | both services |
| Package manager | uv + uv workspaces | replaces pip/poetry |
| Graph DB client | neo4j ≥5.x (async) | replaces falkordb |
| Vector store | pgvector (Postgres 16) + psycopg | HNSW index |
| LLM orchestration | LangChain (graphgen extraction) + LlamaIndex (graphrag agent) | |
| Entity extraction | GLiNER `urchade/gliner_medium-v2.1` + LLMGraphTransformer | |
| Embeddings | SentenceTransformers `BAAI/bge-small-en-v1.5` (384-dim) | upgraded from MiniLM |
| Community detection | leidenalg + igraph | |
| Document parsing | pymupdf4llm, python-docx, python-pptx, openpyxl, trafilatura, pytesseract | |
| Chunking | markdown-it-py (heading-aware) | |
| Streaming | sse-starlette | SSE from FastAPI |
| Observability | Langfuse (auto-instrument + custom spans) | |
| MCP | FastMCP | graphrag exposes agent tools |
| Config | pydantic-settings | .env loading |

### TypeScript / Frontend (apps/web + packages)
| Layer | Technology | Notes |
|-------|-----------|-------|
| Framework | Next.js 15 (App Router) | standalone output for Docker |
| Language | TypeScript 5.x | strict mode |
| Package manager | pnpm 9.x | |
| Monorepo orchestration | Turborepo | |
| UI components | shadcn/ui + Kibo UI component registry | |
| Styling | Tailwind CSS 3.x | |
| State management | TanStack Query v5 (server) + Zustand (UI) | |
| Graph visualization | react-force-graph-2d | existing, retained |
| SSE consumption | `fetch` + `ReadableStream` (POST-based SSE) | EventSource is GET-only |
| Type generation | openapi-typescript | from FastAPI OpenAPI specs |
| Charts | recharts | analytics page |

### Infrastructure
| Component | Image | Port |
|-----------|-------|------|
| Neo4j | `neo4j:5.21-community` | 7474 (HTTP), 7687 (Bolt) |
| pgvector | `pgvector/pgvector:pg16.4` | 5432 |
| Langfuse server | `langfuse/langfuse:2.76` | 3000 |
| Langfuse DB | `postgres:16.4` | internal |
| graphgen | custom (Python 3.11 slim) | 8020 |
| graphrag | custom (Python 3.11 slim) | 8010 |
| web | custom (Node 22.11 Alpine) | 3000 |

## Development Setup

### Prerequisites
- Docker Desktop (or Rancher Desktop)
- pnpm 9.x (`npm install -g pnpm`)
- uv (`curl -LsSf https://astral.sh/uv/install.sh | sh`)
- Node.js 22.x LTS

### Bootstrap
```bash
cp .env.example .env           # fill in LLM_API_KEY at minimum
pnpm install                   # installs all TS workspaces
uv sync                        # installs all Python workspaces
pnpm dev                       # starts infra containers + all services hot-reload
```

### Environment Variable Convention
```
LLM_*          # shared: provider, keys, model names
GRAPHGEN_*     # ETL service config
GRAPHRAG_*     # agent service config
WEB_*          # Next.js app config (NEXT_PUBLIC_* for client-side)
NEO4J_*        # graph DB
POSTGRES_*     # pgvector
LANGFUSE_*     # observability
```

## Technical Constraints
- Neo4j vector indexes require Neo4j 5.11+ (Community supports 1 vector index per DB — use Enterprise or separate indexes per label).
- GLiNER + SentenceTransformers need a Python env with torch; image will be ~2GB. GPU optional.
- SSE via `fetch` POST requires the client to handle `ReadableStream` manually (no `EventSource`).
- `all-MiniLM-L6-v2` → `BAAI/bge-small-en-v1.5`: same 384-dim → pgvector index is compatible; but existing stored embeddings will be semantically stale and require a one-time re-embed.

## Dependencies Being Removed
- `falkordb` (Python + Docker service) — replaced by `neo4j`
- `frontend/backend/` FastAPI BFF — replaced by Next.js Route Handlers
- `leidenalg`-based SEGMENT/EPISODE hierarchy — community detection kept but schema simplified

## Known Issues in Current Codebase
- Unresolved `<<<<<<< HEAD` merge conflicts in:
  - `frontend/static/script.js`
  - `services/graphgen/src/main.py`
  - `services/graphrag/src/main.py`
- No `.dockerignore` — Docker builds pull in `.git`, `node_modules`, `.venv`
- `docker-compose.yaml` includes Neo4j but no code uses it
- `frontend/static/` is static HTML+D3, not a modern SPA

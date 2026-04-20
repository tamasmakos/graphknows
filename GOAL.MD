## What We Are Building

**graphknows** is a production-ready **monorepo template** for building GraphRAG-powered applications: any domain that benefits from a knowledge graph + hybrid vector retrieval + a conversational frontend. The canonical use case is **civicsgraph** — a political intelligence platform — but the template is intentionally domain-agnostic so it can be cloned and adapted.

The template ships three layers:

1. **The Knowledge Graph** — graph ingestion pipeline: entity extraction, normalization, deduplication, embeddings, idempotent writes to Neo4j.
2. **The GraphRAG Service** — hybrid retrieval API: Neo4j graph traversal + pgvector semantic search + keyword search, fused with RRF, with full provenance back to source documents.
3. **The Web Frontend** — a Next.js 15 chatbot UI built on Kibo UI components: source-citing, stream-rendered, no direct database access.

The long-term vision for civicsgraph: become the institutional memory of modern democracy.

***

## Immediate Priority: Standardize the Monorepo Template

Before building civicsgraph features, **get the repo structure right**. The goal is a clean, cloneable template that any team can fork, swap in their domain data, and have a working GraphRAG product in hours — not weeks.

### Phase 0 — Infrastructure (now)
- [x] Smoke test: verify Neo4j, FalkorDB, pgvector connectivity (`smoke_test.py`)
- [x] Lock all deps via `uv.lock`; both Dockerfiles copy the lockfile
- [ ] Add Neo4j to `docker-compose.yaml` as the canonical graph DB (FalkorDB stays as a secondary/legacy target)
- [ ] Add `.dockerignore` at repo root and per-service
- [ ] Add healthchecks to application service containers (`graphgen`, `graphrag`)

### Phase 1 — Monorepo Restructure
Reshape the repo to the target structure below. No feature work until this is clean.

```text
graphknows/
├── apps/
│   ├── web/                ← Next.js 15 + Kibo UI (replaces frontend/)
│   └── api-gateway/        ← FastAPI BFF: aggregates graphgen + graphrag, handles auth
├── services/
│   ├── graphgen/           ← ingestion only (parsing, extraction, graph writes, embeddings)
│   └── graphrag/           ← query only (retrieval, RRF, answer assembly, citations)
├── packages/
│   ├── ui/                 ← shared shadcn/ui + Kibo UI components (pnpm workspace)
│   ├── types/              ← auto-generated TypeScript types via openapi-ts
│   ├── db-adapters/        ← Neo4j + pgvector adapters (no vendor assumptions in domain layer)
│   ├── retrieval/          ← RRF, hybrid retrieval pipeline, context assembly
│   └── evals/              ← benchmark questions, golden answers, regression harness
├── infra/
│   └── docker/             ← compose overrides for dev / ci / prod
├── turbo.json
├── pnpm-workspace.yaml
├── docker-compose.yaml
└── pyproject.toml          ← uv workspace root
```

**What gets removed:** `frontend/` (replaced by `apps/web`), `example_*.py` scripts (moved to `packages/evals`).

### Phase 2 — Frontend: Next.js 15 + Kibo UI
- Bootstrap `apps/web` with Next.js 15 App Router + TypeScript + Tailwind CSS
- Install Kibo UI into `packages/ui` via shadcn registry CLI:
  ```bash
  pnpm dlx shadcn@latest add https://kibo-ui.com/r/<component>.json
  ```
- Key components per view:

  | View | Kibo UI Component |
  |---|---|
  | Document Upload | `Dropzone` |
  | Pipeline Logs | `Code Block` (streamed) |
  | Graph Chat | AI chatbot interface + `Timeline` |
  | Graph Explorer | `Panel` layout + `react-force-graph-2d` |
  | Entity Analytics | `Table` + `Gantt` |
  | Settings | `Multi-step Form` |

- State: **Zustand** for client state, **TanStack Query v5** for server data
- Frontend **never** calls Neo4j or pgvector directly — all requests go through `apps/api-gateway`

### Phase 3 — API Gateway (BFF)
- Thin FastAPI service in `apps/api-gateway/`
- Aggregates `graphgen` (8020) and `graphrag` (8010) into a single OpenAPI surface
- Handles auth: `fastapi-users` for internal access control (roles: admin / viewer)
- Receives multipart uploads from Kibo Dropzone, writes to shared volume, triggers `graphgen`
- Next.js Server Actions call the BFF directly — API keys and internal service URLs never reach the browser

### Phase 4 — civicsgraph Domain (first real dataset)
Only after the template structure is solid:
- Ingest **US Congress** data (congress.gov API + GovTrack: bills, roll-call votes, transcripts)
- Graph schema: `Person`, `Party`, `Institution`, `Speech`, `Vote`, `Bill`, `Event`, `Document`
- Every node carries: source URL, document ID, scrape timestamp, confidence level
- EU Parliament and other jurisdictions deferred until US Congress graph is stable

***

## Technology Stack

| Layer | Technology |
|---|---|
| Graph database | **Neo4j 5** (primary) |
| Legacy graph DB | **FalkorDB** (retained, Cypher-compatible) |
| Vector database | **pgvector** (Postgres 16) |
| Graph query | **Cypher** |
| Backend services | **Python 3.11** (FastAPI) + **uv** workspaces |
| Monorepo orchestration | **Turborepo** + **pnpm workspaces** |
| Frontend framework | **Next.js 15** (App Router, RSC, Server Actions) |
| UI components | **Kibo UI** on top of **shadcn/ui** + Tailwind CSS |
| Type sync | **openapi-ts** (FastAPI → TypeScript types) |
| LLM | **Kimi K2** via OpenRouter (`moonshotai/kimi-k2`) |
| Graph retrieval | **GraphRAG** — graph topology + pgvector + keyword, fused with RRF |
| Observability | **Langfuse** |

***

## Architecture

### Service boundaries (hard rules)

- `graphgen` **writes** to the graph. `graphrag` **reads** from the graph. Never mix.
- `apps/api-gateway` is the only entry point for the frontend. No service URLs or DB credentials in the browser.
- No service encodes Neo4j- or pgvector-specific assumptions in its domain layer — all storage goes through `packages/db-adapters`.
- Prompt templates, retrieval settings, and RRF parameters live in config/package modules — never in route files.
- Every user-visible answer must carry: source URL, document ID, retrieval path, and supporting chunk.
- Upserts must be idempotent. Use `MERGE` in Cypher, not `CREATE`.
- API payloads and citation structures are stable contracts. Version them if they change.
- RRF is mandatory. Evaluate against a vector-only baseline before marking any retrieval feature done.

### Port map

| Service | Internal port | Host port |
|---|---|---|
| Neo4j Bolt | 7687 | 7687 |
| Neo4j HTTP | 7474 | 7474 |
| FalkorDB | 6379 | 6379 |
| pgvector | 5432 | 5435 |
| graphgen | 8000 | 8020 |
| graphrag | 8000 | 8010 |
| api-gateway | 8000 | 8000 |
| apps/web (dev) | 3000 | 3000 |
| Langfuse | 3000 | 3002 |

***

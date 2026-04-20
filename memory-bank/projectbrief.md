# Project Brief

## Project Name
**GraphKnows** — Open-Source Knowledge-Graph RAG Monorepo Template

## One-Liner
A production-quality, cloneable monorepo that any company can configure with their own documents and LLM API keys to have an internal knowledge-graph RAG system running in 30 minutes.

## Core Requirements
1. **Genuinely modular** — adding new node types, edge types, parsers, or agent tools touches only isolated extension points, never core logic.
2. **Python backend** — FastAPI for both `graphgen` (ETL) and `graphrag` (retrieval). Dependency management via `uv` workspaces.
3. **Modern frontend** — Next.js 15 App Router + TypeScript + Tailwind CSS + shadcn/ui + Kibo UI. No Vite/React legacy.
4. **Monorepo tooling** — Turborepo at root; pnpm workspaces for TypeScript; uv workspace for Python.
5. **Simplified graph schema** — strict `DOCUMENT → CHUNK → ENTITY` canonical shape with plugin extension system.
6. **Docker best practices** — multi-stage builds, non-root users, pinned base images, HEALTHCHECK, .dockerignore.

## Non-Negotiables
- Graph DB: **Neo4j only** (FalkorDB removed entirely)
- BFF layer: **Next.js Route Handlers** under `apps/web/app/api/v1/`
- Chat streaming: **SSE** (POST-based via `fetch` + `ReadableStream`)
- Clean break: no deprecation shims, no life-log / simulation code
- Keep: community detection (Leiden), entity resolution, pruning, MCP server, GLiNER hinting

## Success Criteria
- Fresh clone → `pnpm dev` → all services healthy in < 5 minutes
- Upload a document → `DOCUMENT{status:'complete'}` + connected CHUNK/ENTITY nodes in Neo4j
- Multi-hop chat query → SSE stream with ≥2 tool calls + structured citations
- Adding a new parser/node type/agent tool requires editing only 1 new file
- `docker compose build` → images pass `docker scout quickview` with no critical CVEs

## Scope Boundary (MVP)
- Single-tenant (no auth — stub middleware only)
- Single Neo4j instance (no clustering)
- English documents only (OCR: Tesseract default)
- Conversations stored in SQLite (single web container)

# Tasks Index

_Last updated: 2026-04-21 after full codebase audit._

---

## In Progress
_(none — Phase A is next)_

---

## Pending — Phase A: Cleanup & Unification (must be first)

- **[TASK010]** Nuke Legacy & Align Schema — delete FalkorDB/life-log/dead code across both services; fix embedding model to bge-small-en-v1.5; add pytesseract; unify `/run` to new parser system; update stale docs.
- **[TASK011]** Single Workflow + Schema Bootstrap + MCP Alignment — Neo4j constraints/vector/fulltext indexes at lifespan; rewire MCP to agent workflow; remove FalkorDB from graphrag; resolve pgvector + MCP scope questions.

## Pending — Phase B: Ingestion Scale

- **[TASK012]** Background Worker Pool + Job Store — asyncio.Queue, bounded Semaphore, JobRecord registry, per-document `process_document()` with ProgressCallback, `/jobs` SSE endpoints.
- **[TASK013]** Idempotency, Dedup, Resume — content_hash MERGE, stable chunk_id, `?force=true`, job retry on startup crash.

## Pending — Phase C: Agent Quality

- **[TASK014]** Proper AgentWorkflow + Full SSE Taxonomy — explicit decompose/iterate/synthesize phases; `reasoning | tool_call | tool_result | token | citation | subgraph | done | error` events; full Pydantic models (ToolCall, ReasoningStep, SubQuestion); per-tool Langfuse spans.
- **[TASK015]** Reranker + Hybrid Search — BGE reranker; hybrid vector+fulltext merge in search_chunks and search_entities; Langfuse span.
- **[TASK016]** Verifier Sub-Agent — self-critique pass after synthesize; one retry on unsupported claims; `verification` field in AgentResponse.

## Pending — Phase D: Frontend Rebuild

- **[TASK017]** Design System Foundation — `shadcn init`; Kibo UI registries (Dropzone, AI chat, Timeline, etc.); populate packages/ui primitives; next-themes; Sonner; replace all CSS-var inline styles.
- **[TASK018]** App Shell + Landing + Navigation — collapsible Sidebar, TopBar, CommandPalette (⌘K); landing page at `/`; route group `(app)/`; mobile Sheet sidebar.
- **[TASK019]** Documents View Rebuild — Kibo Dropzone with per-file job-SSE progress; TanStack DataTable; document detail Sheet (chunks/entities/ingestion log).
- **[TASK020]** Chat View Rebuild — Kibo AI Input + AI Message; inline citation HoverCards `[n]`; Reasoning Timeline; SQLite conversation persistence; error states with Sonner.
- **[TASK021]** Graph Explorer + Analytics Rebuild — label/doc filters; full-screen force graph; inspector Sheet with neighbor expand; recharts bar/donut/area charts.

## Pending — Phase E: Observability, CI, Tests

- **[TASK022]** CI/CD + Tests — `.github/workflows/ci.yml`; testcontainers Neo4j integration tests; vitest component tests; Playwright E2E; 70/20/10 coverage ratio target.
- **[TASK023]** Observability + Evals — structlog; per-tool Langfuse spans (prerequisite done in TASK014); Ragas eval harness; Dependabot; spaCy removal; Node SHA-pin.

## Pending — Phase F: DevEx & Docs

- **[TASK024]** CONTRIBUTING + One-Command Dev + Docs — 4 extension recipes (parser/node/tool/view); `pnpm dev` one-liner; README fixes; `docs/ARCHITECTURE.md`; memory-bank final refresh.

---

## Completed

- **[TASK001]** Root Tooling Scaffold — Turborepo + pnpm-workspace.yaml + turbo.json + root package.json + package stubs. _Completed._
- **[TASK002]** Cleanup Dead Code — Merge conflicts resolved; `frontend/` deleted; `simulation/` removed; static BFF removed. ⚠️ **Incomplete**: FalkorDB code and life-log parsers still present in both services — carry-forward to TASK010.
- **[TASK003]** Docker Compose — Neo4j added, `web` service added, `docker-compose.dev.yaml` created. _Completed._
- **[TASK004]** .dockerignore / .env.example / .gitignore — All three created and complete. _Completed._
- **[TASK005]** DocumentParser Abstraction — `kg/parser/` with 8 format parsers + HeadingAwareChunker + ParserRegistry. _Completed._
- **[TASK006]** Schema + Plugin System + Neo4j Uploader — `kg/schema.py`, `kg/plugins/`, `kg/neo4j/` all implemented. ⚠️ **Incomplete**: no schema bootstrap at startup, dual-pipeline mismatch, embedding model not updated — carry-forward to TASK010/011.
- **[TASK007]** AgentWorkflow Rewrite — ReActAgent with 4 Neo4j tools, SSE /chat endpoint, basic models. ⚠️ **Incomplete**: dual workflows, incomplete SSE taxonomy, no reranker/verifier — carry-forward to TASK010/014/015/016.
- **[TASK008]** Next.js 15 Frontend — 4 routes, BFF Route Handlers, DocumentsPane, GraphVisualizer, packages/types. ⚠️ **Incomplete**: no Kibo UI/shadcn, packages/ui empty, pages minimal — carry-forward to TASK017-021.
- **[TASK009]** Dockerfiles + CONTRIBUTING.md — Multi-stage, non-root, HEALTHCHECK, CONTRIBUTING with recipes. _Completed._

## Abandoned
_(none)_

---

## Implementation Dependency Order

```
TASK010 ──► TASK011 ─┬─► TASK012 ──► TASK013
                     │
                     ├─► TASK014 ──► TASK015 ──► TASK016
                     │
                     └─► TASK017 ──► TASK018 ─┬─► TASK019
                                               ├─► TASK020  (also needs TASK012, TASK014)
                                               └─► TASK021

All Phase A–D complete ──► TASK022 ──► TASK023 ──► TASK024
```

**Critical path**: TASK010 → TASK011 → (TASK012 || TASK014 || TASK017 in parallel) → TASK022 → TASK024.

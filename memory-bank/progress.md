# Progress

## Overall Status
**Audit complete 2026-04-21. Phase A cleanup is the immediate blocker. Two competing codepaths exist in both services; the frontend has no design system installed.**

| Phase | Description | Status | % |
|-------|-------------|--------|---|
| Phase 1a | Root tooling scaffold (Turborepo, pnpm, turbo.json) | ✅ Complete | 100% |
| Phase 1b | Resolve merge conflicts + delete dead code | ⚠️ Partial | 50% |
| Phase 1c | docker-compose cleanup + dev overlay | ✅ Complete | 100% |
| Phase 1d | .dockerignore, .env.example, .gitignore | ✅ Complete | 100% |
| Phase 2 | DocumentParser abstraction (graphgen) | ✅ Complete | 100% |
| Phase 3 | Schema.py + plugin system + Neo4j uploader | ⚠️ Partial | 60% |
| Phase 4 | AgentWorkflow rewrite (graphrag) | ⚠️ Partial | 35% |
| Phase 5 | Next.js 15 frontend (apps/web) | ⚠️ Partial | 30% |
| Phase 6 | Dockerfile upgrades + CONTRIBUTING.md | ✅ Complete | 100% |
| **Phase A** | **Cleanup & Unification (TASK010–011)** | Not Started | 0% |
| **Phase B** | **Ingestion Scale (TASK012–013)** | Not Started | 0% |
| **Phase C** | **Agent Quality (TASK014–016)** | Not Started | 0% |
| **Phase D** | **Frontend Rebuild (TASK017–021)** | Not Started | 0% |
| **Phase E** | **CI/Tests/Observability (TASK022–023)** | Not Started | 0% |
| **Phase F** | **DevEx & Docs (TASK024)** | Not Started | 0% |

---

## What Actually Works (Audit Verified 2026-04-21)

### graphgen
- **8-format parser system** (`kg/parser/`): TXT, MD, PDF, DOCX, PPTX, XLSX, HTML, Image (OCR) — auto-discovery via `BaseParser.__init_subclass__`. Works.
- **Neo4j async driver, uploader, index creation** (`kg/neo4j/`). Works in isolation.
- **Declarative schema** (`kg/schema.py`): `NodeSpec`/`EdgeSpec`/`PropertySpec`. Works.
- **Plugin auto-discovery** (`kg/plugins/`): `GraphPlugin.__init_subclass__`, `TopicsPlugin`. Works.
- **Entity extraction**: GLiNER + LLMGraphTransformer pipeline (`kg/graph/extraction.py`). Works.
- **Semantic entity resolution** (`kg/graph/resolution.py`). Works.
- **Community detection**: Leiden via igraph (`kg/community/`). Works.
- **Summarization**: LLM-based titles + summaries (`kg/summarization/`). Works.
- **Pruning** (`kg/graph/pruning.py`). Works.
- **KnowledgePipeline orchestrator** (`kg/pipeline/core.py`) — 7 steps wired end-to-end. Works.
- **FastAPI endpoints**: `GET/POST /documents`, `GET /documents/{id}`, `DELETE /documents/{id}`, `GET /analytics`, `GET /health`, `POST /run`. Most work. `/run` calls broken legacy path.

### graphrag
- **FastAPI endpoints**: `POST /chat` (SSE), `POST /chat/sync`, `GET /schema`, `GET /health`, `GET /node-connections/{id}`. All work.
- **LlamaIndex ReActAgent** (`agent/workflow.py`): iterative loop, max_iterations=10, returns `{answer, citations, graph_data}`. Works.
- **4 Neo4j retrieval tools** (`agent/tools.py`): `search_chunks`, `get_entity_neighbours`, `get_document_context`, `search_entities` — all target correct DOCUMENT/CHUNK/ENTITY schema. Work if Neo4j has correct indexes.
- **Global Langfuse tracing** via `LlamaIndexInstrumentor`. Works when keys are set.
- **Groq/OpenAI LLM switching** and `BAAI/bge-small-en-v1.5` embeddings. Works.
- **MCP server** (FastMCP) — wired but calls wrong workflow (see issues).

### apps/web
- **Next.js 15 App Router** with TypeScript, Tailwind, 4 routes. Runs.
- **BFF Route Handlers** proxying to graphgen/graphrag with proper timeout + SSE passthrough. Work.
- **Chat page** (`/chat`): SSE streaming, message bubbles, citation tags, tool call display, live GraphVisualizer. Works end-to-end.
- **DocumentsPane**: drag-drop upload, document list, pipeline trigger, activity log, 5-second polling. Works.
- **GraphVisualizer**: react-force-graph-2d with node colour by label, responsive sizing. Works.
- **packages/types**: `ChatRequest`, `ChatResponse`, `Citation`, `SSEChunk`, `GraphNode`, `GraphEdge`, `GraphData` types. Complete and used.

### Infra
- `docker-compose.yaml` + `docker-compose.dev.yaml` — Neo4j, pgvector, graphgen, graphrag, Langfuse, web. Works.
- Multi-stage Dockerfiles for all 3 services (non-root, HEALTHCHECK, tini). Correct.
- Root `.dockerignore` + `.env.example`. Complete.
- `turbo.json` + `pnpm-workspace.yaml` + root `pyproject.toml` (uv workspace). Wired.
- Pre-commit hooks (ruff, trailing-whitespace, merge-conflict check). Configured.

---

## What Is Broken / Half-Built

### graphgen
| Issue | Location | Severity |
|-------|----------|----------|
| `/run` POST calls LifeLogParser (returns `[]`) | `main.py` + `kg/graph/parsers/life.py` | 🔴 Critical |
| Schema mismatch: `/run` creates DAY/SEGMENT/EPISODE; `/documents` creates DOCUMENT/CHUNK | `kg/graph/extraction.py` | 🔴 Critical |
| No Neo4j schema bootstrap at startup | missing `kg/neo4j/schema_bootstrap.py` | 🔴 Critical |
| `pytesseract` missing from `pyproject.toml` | `kg/parser/image.py` imports it | 🔴 Runtime crash |
| Embedding model still `all-MiniLM-L6-v2` | `kg/embeddings/model.py` | 🟡 Semantic mismatch |
| `/documents/{id}/reprocess` is a `# TODO` stub | `main.py` | 🟡 Missing |
| Legacy `kg/graph/parsers/` folder not deleted | 3 files with LifeLogParser | 🟡 Confusion |
| `falkordb` still in `requirements.txt` | `requirements.txt` line 7 | 🟡 Dead dep |
| `kg/utils/health.py` imports FalkorDB | `kg/utils/health.py` | 🟡 Dead code |
| `KG_README.md` mentions FalkorDB + old schema | `kg/KG_README.md` | 🔵 Docs |

### graphrag
| Issue | Location | Severity |
|-------|----------|----------|
| Two incompatible workflows in the same service | `agent/workflow.py` vs `workflow/graph_workflow.py` | 🔴 Critical |
| MCP wired to GraphWorkflow (FalkorDB) not agent (Neo4j) | `mcp/server.py` | 🔴 Wrong path |
| No Neo4j schema/index initialization in graphrag | `main.py` lifespan | 🔴 Agent tools fail without indexes |
| SSE taxonomy incomplete (missing reasoning/tool_call/tool_result/error) | `main.py` event stream | 🟡 Quality |
| No per-tool Langfuse spans (only global auto-instrument) | `agent/tools.py` | 🟡 Quality |
| No reranker (raw vector-only retrieval) | `agent/tools.py` | 🟡 Quality |
| No explicit query decomposition or sufficiency check | `agent/workflow.py` | 🟡 Quality |
| No verifier sub-agent | missing | 🟡 Quality |
| `ToolCall`, `ReasoningStep`, `SubQuestion` models missing | `models/__init__.py` | 🟡 Incomplete |
| Dead infrastructure: FalkorDBDB, postgres_store, graph_retriever, context_builder | `infrastructure/`, `services/` | 🟡 Clutter |
| `APP_README.md` references non-existent `llamaindex_agent.py` | docs | 🔵 Stale |

### apps/web + packages
| Issue | Location | Severity |
|-------|----------|----------|
| No shadcn/ui, no Kibo UI installed | `apps/web/package.json` | 🔴 Critical for quality |
| `packages/ui` is a hollow placeholder | `packages/ui/src/components/placeholder.ts` | 🔴 Empty |
| Mixed styling: Tailwind + CSS-var `style={{}}` throughout | `/documents`, `/graph`, `/analytics`, `layout.tsx` | 🟡 Inconsistent |
| No conversation persistence (SQLite) | `/chat/page.tsx` | 🟡 Missing |
| No dark/light theme toggle | `layout.tsx` | 🟡 Missing |
| `/documents/page.tsx` is redundant + minimal (uploads already in DocumentsPane) | `app/documents/page.tsx` | 🟡 Confusing UX |
| `/graph/page.tsx` shows three lists, no interactive explorer | `app/graph/page.tsx` | 🟡 Not useful |
| `/analytics/page.tsx` shows 4 numbers, no charts | `app/analytics/page.tsx` | 🟡 Minimal |
| `GraphVisualizer` node click is `console.log` only | `GraphVisualizer.tsx` | 🟡 No inspector |
| No loading skeletons, no toast system, no error boundaries | throughout | 🟡 UX polish |
| No landing page | missing | 🟡 Missing |

### Infra / DevEx
| Issue | Location | Severity |
|-------|----------|----------|
| `.github/workflows/` is empty — no CI | `.github/workflows/` | 🔴 Critical |
| Zero test files (pytest + vitest configured but empty) | `tests/`, `apps/web/` | 🔴 Critical |
| Node.js base not SHA-pinned (`node:18-alpine`) | `apps/web/Dockerfile` | 🟡 Supply chain |
| spaCy `en_core_web_lg` baked in graphgen image (~40MB bloat) | `services/graphgen/Dockerfile` | 🟡 Image size |
| `generate-types.sh` requires live services — can't run in CI | `scripts/generate-types.sh` | 🟡 DevEx |
| No Dependabot | missing | 🟡 Supply chain |
| README port reference 8010 should be 3001 (web) | `README.md` | 🔵 Docs |

---

## Remaining Build List (New Phase A–F)

### Phase A — Cleanup & Unification
- [ ] TASK010: Nuke all FalkorDB/life-log/legacy code; fix embedding model to bge-small-en-v1.5; add pytesseract to pyproject.toml; align `/run` to new parser; update stale docs.
- [ ] TASK011: Neo4j schema bootstrap (constraints + vector + fulltext indexes); rewire MCP to agent workflow; remove FalkorDB from graphrag settings; pick pgvector fate.

### Phase B — Ingestion Scale
- [ ] TASK012: In-process asyncio.Queue + worker pool; per-document `process_document()`; `/jobs` SSE endpoints.
- [ ] TASK013: Content-hash idempotency; stable chunk IDs; `?force=true`; job retry on startup crash.

### Phase C — Agent Quality
- [ ] TASK014: Explicit AgentWorkflow (decompose/iterate/synthesize) + full SSE taxonomy + per-tool Langfuse spans + Pydantic models (ToolCall, ReasoningStep, SubQuestion).
- [ ] TASK015: BGE reranker; hybrid search (vector + fulltext merge); reranker Langfuse span.
- [ ] TASK016: Verifier sub-agent (self-critique before return; one retry on unsupported claims).

### Phase D — Frontend Rebuild
- [ ] TASK017: shadcn init + Kibo UI registries + packages/ui primitives + next-themes + Sonner.
- [ ] TASK018: App shell (Sidebar, TopBar, CommandPalette) + landing page + route group.
- [ ] TASK019: Documents view (Kibo Dropzone + TanStack DataTable + job-SSE progress + detail Sheet).
- [ ] TASK020: Chat rebuild (Kibo AI chat + citation HoverCards + reasoning Timeline + SQLite conversations).
- [ ] TASK021: Graph Explorer (filters + inspector Sheet + neighbor expand) + Analytics (recharts charts).

### Phase E — CI, Tests, Observability
- [ ] TASK022: GitHub Actions CI; testcontainers integration tests; vitest component tests; Playwright E2E; target 70/20/10 ratio.
- [ ] TASK023: structlog; per-tool Langfuse spans; Ragas eval harness; Dependabot; spaCy removal; Node SHA-pin.

### Phase F — DevEx & Docs
- [ ] TASK024: CONTRIBUTING extension recipes (parser/node/tool/view); one-command dev; README fix; ARCHITECTURE.md; memory-bank refresh.

---

## Known Issues Carried Forward
1. Embedding model mismatch: code says `all-MiniLM-L6-v2`, locked decision says `bge-small-en-v1.5`. Fix in TASK010. Same dim (384) so no index migration needed.
2. pgvector role unclear: present in infra but bypassed now that Neo4j vectors are used. Decision needed before TASK011 (drop vs keep CHUNK-only).
3. spaCy `en_core_web_lg` pre-downloaded in graphgen Docker build — should be removed since GLiNER handles NER. Fix in TASK023 (or pull into TASK010 for quick win).
4. No integration tests exist — first test run will be full discovery. Plan for testcontainers Neo4j in TASK022.
5. MCP server: currently wired to wrong workflow AND the auth story is undefined. Clarify scope before TASK011.

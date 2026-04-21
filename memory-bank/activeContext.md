# Active Context

## Current Focus
**Extensive audit complete (2026-04-21). Planning Phase A–F drafted. TASK001–009 partially delivered — significant gaps remain.** The codebase has two of everything and one of nothing finished. Before any new feature work, Phase A cleanup tasks (TASK010–011) must run to eliminate competing codepaths. Next active work item: **TASK010**.

## Audit Findings Summary (2026-04-21)

### graphgen — critical gaps
- **Dual pipeline entry points**: `/documents POST` uses modern `kg/parser/` (8 parsers, auto-discovery) but `/run POST` still calls `LifeLogParser` from the legacy `kg/graph/parsers/` system — which returns `[]`. Schema mismatch: new path creates `DOCUMENT→CHUNK`, old path creates `DAY→SEGMENT→EPISODE→CHUNK`.
- **Legacy cruft still present**: `kg/graph/parsers/` folder (3 files), `kg/utils/health.py` (FalkorDB imports), `kg/graph/parsing.py` (stub only), `falkordb` in `requirements.txt`.
- **`pytesseract` missing from `pyproject.toml`** — `kg/parser/image.py` imports it; runtime crash.
- **Embedding model still `all-MiniLM-L6-v2`** in `embeddings/model.py` despite locked decision for `BAAI/bge-small-en-v1.5`.
- **No Neo4j schema bootstrap** — no constraints, no vector indexes, no fulltext index created at startup.
- `/documents/{id}/reprocess` endpoint exists but is `# TODO` stub.

### graphrag — critical gaps
- **Two incompatible workflows**: `agent/workflow.py` (ReActAgent + Neo4j) vs `workflow/graph_workflow.py` (LlamaIndex Workflow + FalkorDB). MCP server wired to wrong one (GraphWorkflow).
- **Schema mismatch**: agent tools target `DOCUMENT/CHUNK/ENTITY`; GraphWorkflow targets `TOPIC/SUBTOPIC/ENTITY_CONCEPT`.
- **Dead infrastructure**: `infrastructure/graph_db.py` (FalkorDBDB), `infrastructure/postgres_store.py`, `services/graph_retriever.py`, `services/context_builder.py` — not used by agent path.
- **SSE taxonomy incomplete**: only emits `token | citation | graph | done`. Missing: `reasoning | tool_call | tool_result | error`.
- **Langfuse**: global OpenTelemetry enabled but no per-tool spans.
- **No reranker, no verifier sub-agent, no explicit decompose/sufficiency loop** — just `ReActAgent.achat()`.
- **`ToolCall`, `ReasoningStep`, `SubQuestion` models missing** from `models/__init__.py`.

### frontend — critical gaps
- **Zero component library**: no shadcn/ui, no Kibo UI. All hand-rolled raw Tailwind + CSS variables.
- **`packages/ui`** is a hollow placeholder (`export __placeholder = true`).
- `/documents`, `/graph`, `/analytics` pages are minimal stubs with inconsistent styling.
- No conversation persistence (messages lost on refresh). No toasts. No skeletons. No landing page. No light/dark toggle. No command palette.
- `GraphVisualizer` node click handler is `console.log` only — no inspector.

### infra/DevEx
- `.github/workflows/` is empty — no CI.
- Zero test files (pytest + vitest listed as deps but unused).
- Node.js base image not SHA-pinned (`node:18-alpine`).
- spaCy `en_core_web_lg` baked into graphgen Docker image (~40MB bloat, largely unused since GLiNER handles NER).
- `scripts/generate-types.sh` requires running services — cannot run in CI.

## Locked Decisions (confirmed post-audit)
| Concern | Decision |
|---------|---------|
| Graph DB | Neo4j only — nuke all FalkorDB code |
| Schema | `DOCUMENT→CHUNK→ENTITY` canonical, `community_id` as property on Entity (not TOPIC nodes) |
| BFF | Next.js Route Handlers in `apps/web/app/api/v1/` |
| Streaming | SSE via POST + `fetch`/`ReadableStream`; full event taxonomy (see systemPatterns §5) |
| Legacy: keep | Community detection (Leiden), entity resolution, pruning, MCP server, GLiNER |
| Legacy: drop | FalkorDB, GraphWorkflow, LifeLogParser, DAY/SEGMENT/EPISODE/TOPIC/SUBTOPIC, spaCy |
| Embedding model | `BAAI/bge-small-en-v1.5` (384-dim) — fix in TASK010 |
| Conversations | SQLite `better-sqlite3` in Docker volume `/data/conversations.sqlite` |
| Auth | Stub middleware only (no auth in MVP) |
| Ingestion scale | In-process `asyncio.Queue` + bounded worker pool (no Redis/Celery for MVP) |
| Agent quality | Explicit decompose → iterate(tool/reflect/sufficiency) → rerank (BGE) → synthesize → verifier |
| UI design system | shadcn/ui + Kibo UI component registry; Linear/Notion aesthetic; `next-themes` |
| pgvector | **Decision pending** — likely drop in favour of Neo4j vector indexes for both CHUNK+ENTITY |
| spaCy | **Drop** — GLiNER replaces NER; saves ~40MB in Docker image |

## Open Questions (answer before TASK011)
1. **pgvector**: drop entirely (Neo4j vectors cover CHUNK+ENTITY) / keep for CHUNK / keep both?
2. **MCP server**: keep public-facing (needs auth plan) / internal dev tool only?

## Next Steps (in order)
1. **TASK010** — Nuke legacy: delete FalkorDB, life-log parsers, GraphWorkflow, dead infrastructure; fix embedding model; add `pytesseract` to pyproject.toml.
2. **TASK011** — Neo4j schema bootstrap (constraints + vector + fulltext indexes in `lifespan`); rewire MCP to agent workflow; remove FalkorDB from graphrag settings.
3. **TASK012** — Background worker pool + job store (asyncio.Queue, SSE progress endpoints).
4. **TASK013** — Idempotency + content-hash dedup + retry.
5. **TASK014** — Proper AgentWorkflow rewrite (decompose/iterate/synthesize) + full SSE taxonomy.
6. **TASK015** — Reranker + hybrid search (BGE reranker + fulltext merge).
7. **TASK016** — Verifier sub-agent (self-critique pass before return).
8. **TASK017** — Design system: shadcn init + Kibo UI + packages/ui populate + next-themes.
9. **TASK018** — App shell, landing page, sidebar, command palette.
10. **TASK019** — Documents view rebuild (Kibo Dropzone + DataTable + detail Sheet).
11. **TASK020** — Chat rebuild (Kibo AI chat + reasoning timeline + citation hovers + SQLite).
12. **TASK021** — Graph Explorer + Analytics rebuild (recharts, inspector, neighbor expand).
13. **TASK022** — CI/CD + integration tests + component tests + E2E.
14. **TASK023** — Observability (structlog, Langfuse spans, Ragas evals, Dependabot).
15. **TASK024** — CONTRIBUTING, one-command dev, README fixes, architecture doc.

## Patterns to Follow Consistently
- Python service Pydantic models use `model_config = ConfigDict(extra='ignore')`.
- All FastAPI apps use lifespan context managers (not `@app.on_event`).
- Neo4j queries use **parameterized Cypher only** — no f-string or `.format()` into Cypher.
- All async Python uses `async with driver.session() as session` — never share sessions.
- TypeScript: use `satisfies` not `as`; avoid `any`; all Route Handlers return typed `NextResponse`.
- SSE event shape: `{ type: EventType, data: unknown, timestamp: string }`.
- Every tool in graphrag wraps body in a Langfuse span via decorator before returning.
- shadcn/Kibo components only in `apps/web` — no raw Tailwind utility-soup for interactive elements.

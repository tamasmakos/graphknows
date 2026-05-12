# Wiki Log

> Chronological record of all wiki actions. Append-only.
> Format: ## [YYYY-MM-DD] action | subject
> Rotate when >500 entries.

## [2026-05-12] restructure | wiki cleanup — flat concepts/ → typed folders
- Deleted concepts/ folder entirely (was a catch-all with 33 files)
- New structure: architecture/ adr/ product/ research/ ops/ entities/ raw/
- Deleted 12 noise files: 6 dated session notes (2026-05-06), graph-viz-diagnosis, test-results-frontend, kanban-plan, graphgen-architecture, graphrag-and-downstream, pipeline-stages
- Promoted 4 files to adr/: 001-langgraph-agent, 002-investigative-workflow, 003-services-refactor, 004-ontology-patch
- Moved 6 files to architecture/, 4 to research/, 5 to product/, 2 to ops/
- Rewrote SCHEMA.md: full structure definition, folder conventions, tag taxonomy, log format rules
- Rewrote index.md: table-per-folder layout, [[wikilinks]], page count 26

## [2026-05-12] pivot | deep-research-agent model
- Major product pivot: monolith shared KG → per-session deep research agent
- Archived 35 stale kanban cards from old model
- Created 7 new cards: product-vision-rewrite, deep-research-loop, dual-write, hybrid-retrieval, brief-node, session-lifecycle, pgvector-namespacing
- Rewrote: product-architecture-v2.md, project-roadmap.md
- Created: deep-research-agent-design.md, dual-write-hybrid-retrieval.md
- Updated: index.md

## [2026-05-10] create | concepts/investigative-agent-architecture-redesign-2026-05-10.md
- Captured the 3 macro-phase investigative workflow redesign: Scope → Investigate → Communicate
- Documented strict HTTP-only boundaries for search/graphgen, async session graph jobs, and Memgraph session ingest mode
- Defined the first implementation slice: search_client boundary before deeper workflow/stategraph changes

## [2026-05-03] create | entities/graphgen-project.md
- GraphGen project entity page: tech stack, key results, infrastructure

## [2026-05-03] create | concepts/graphgen-architecture.md
- Full pipeline architecture: DOC->SEGMENT->CHUNK->ENTITY->TOPIC, PipelineContext, modules

## [2026-05-03] create | concepts/pipeline-stages.md
- All 10 pipeline stages with rationale and parameters

## [2026-05-03] create | concepts/knowledge-graph-construction.md
- KG construction: ontology, scale-free topology, entity resolution, canonicalization

## [2026-05-03] ingest | raw/papers/graphgen-thesis-synthesis.md
- Immutable synthesis of thesis corpus, results, evaluation findings

## [2026-05-03] create | concepts/thesis-findings-and-learnings.md
- Core claim, empirical results table, what worked/didn't, open questions, academic framing

## [2026-05-03] create | concepts/graphrag-and-downstream.md
- GraphKnows GraphRAG agent, Neo4j+pgvector dual retrieval, pending task tracker

## [2026-05-03] create | concepts/business-plan.md
- Full business plan: problem, solution, customers, revenue model, competitive landscape

## [2026-05-03] create | concepts/market-analysis.md
- TAM/SAM/SOM, comparable SaaS pricing, EU regulatory demand, buyer personas

## [2026-05-03] create | queries/gtm-strategy.md
- Three-phase GTM strategy: EU Parliament -> national parliaments -> regulatory bodies

## [2026-05-03] create | concepts/project-roadmap.md
- Phase 0-3 roadmap with task IDs, dependencies, effort estimates

## [2026-05-03] create | concepts/technical-debt.md
- 10-item debt register: label collapse, community_id, pgvector, entity resolution, etc.

## [2026-05-03] create | concepts/administration.md
- Repo structure, PR workflow, config management, key commands, infrastructure

## [2026-05-03] create | concepts/kanban-plan.md
- Full Kanban board with in-progress, blocked, backlog, done columns

## [2026-05-03] update | index.md
- Consolidated canonical index after parallel wiki population

## [2026-05-04] create | concepts/product-architecture-v2.md
- Full product architecture redesign: search service (from scrapers), agent service (from graphrag), service decomposition, session-scoped context graph, frontend viz exports, market/academic validation synthesis


## [2026-05-04] update | concepts/product-architecture-v2.md
- Added Memgraph as session graph layer (replaces Redis/in-memory Neo4j); full technical rationale, Docker config, Cypher schema, MAGE algorithms

## [2026-05-04] update | concepts/graphrag-and-downstream.md
- Added Memgraph session graph section: architecture, lifecycle, Cypher differences

## [2026-05-04] create | kanban: t_f5f3022a infra-memgraph-docker — Memgraph in docker-compose for agent service

## [2026-05-05] update | frontend-v2-design
- PR #10 merged: apps/web/ (SPA), services/agent/main.py
- Agent at localhost:8002, full stack verified
- Public URL: https://exposure-steel-corpus-parents.trycloudflare.com


## [2026-05-04] create | concepts/architecture-reference.md
- Canonical technical reference: ASCII service map (frontend :8080 → agent :8002 → graphrag :8001 → Neo4j/pgvector + Memgraph :17687)
- Step-by-step 7-stage data flow for full query cycle
- Complete API contracts for all endpoints (graphrag /retrieve, agent /chat + /session/{id}/graph + DELETE /session/{id}, frontend proxy)
- All environment variables with defaults and descriptions (agent, graphrag, frontend)
- Exact commands to start all three services locally
- Cloudflare tunnel / ngrok launch via python3 apps/web/launch.py

## [2026-05-04] create | concepts/product-roadmap-v2.md
- Shipped: full registry of PRs #1–#10 with descriptions of what each delivers
- Testing wave: 5 stability gates (A: health, B: data integrity, C: RAGAS quality, D: resilience, E: frontend)
- Next features P1–P8 in priority order with acceptance criteria: corpus ingest, WebSocket streaming, named sessions, Neo4j write-back, Bundestag scraper, RAGAS eval, CI/CD, auth+rate-limiting
- Horizon features: multi-country KG federation, argument cards, embed API, public data API
- GTM beachhead: investigative journalists + policy researchers at $99/$199/$299/mo tiers

## [2026-05-04] update | index.md
- Added entries for architecture-reference and product-roadmap-v2 under Concepts — Architecture section

## [2026-05-05] create | test-results-frontend
- 3 bugs found and fixed during testing
- All 8 smoke tests now PASS
- Commits: f4741e2, 7b1b68a, 1a207a7

## [2026-05-05] wave-3 | multi-stream development iteration

### PRs merged:
- PR #11: feat/websocket-streaming — live ReAct step streaming; reasoning trace updates in real time
- PR #12: fix/frontend-ux-audit — 12-point UX audit: node rendering, reasoning_steps forwarded, elapsed timer, citation modal, edge labels, force tuning, node animations, reset button, error cards

### Architecture decisions:
- WebSocket endpoint: GET /ws/chat/:session_id on agent:8002, proxied through frontend:8080
- WS protocol: status -> step frames -> answer (HTTP /chat preserved as fallback)
- Ingestion redesign: BaseSource abstract class (in-progress on feat/base-source)
- EU horizontal scaling: 6 new adapters planned (EP resolutions, Council conclusions, EC comms, ECB, ECA, EP committees)

### Running services:
- graphrag :8001, agent :8002, frontend :8080
- cloudflare tunnel: https://exposure-steel-corpus-parents.trycloudflare.com

### Next:
- feat/base-source BaseSource + ParsedDocument
- feat/eu-source-adapters 6 new EU document adapters
- Named sessions, export suite, RAGAS eval

## [2026-05-05] create | concepts/eu-source-taxonomy.md
- 12 EU institutions mapped: EP, European Council, Council EU, Commission, EUR-Lex, ECB, ECA, CURIA, EPRS, CoR, EESC, Europol
- 25+ document types with PIV/difficulty/endpoint/volume ratings
- Adapter priority order: EP Resolutions, Council Conclusions, EC Press, CELLAR SPARQL, ECB, ECA, EPRS, CURIA, MEP votes, EP Committees

## [2026-05-05] wave-4 | ingestion horizontal scaling + UX features

### PRs committed to main:
- PR #13: feat/base-source — BaseSource abstract class + ParsedDocument dataclass + EPResolutions adapter + 16 tests
- PR #14: feat/eu-source-adapters — 5 EU adapters: Council Conclusions, EC Press, ECB Bulletins, ECA Reports, EPRS Briefings + 22 tests
- PR #15: feat/named-sessions — investigation sidebar, localStorage persistence, inline rename, session restore via /api/session/{id}/graph

### EU taxonomy written:
- 12 EU institutions, 25+ source types, PIV/difficulty ratings, endpoint URLs, adapter priority order
- See concepts/eu-source-taxonomy.md

### Architecture now:
- services/search/adapters/ has 9 adapters total (ep_debates, eurlex, uk_legislation, ep_resolutions, council_conclusions, ec_press, ecb_publications, eca_reports, eprs_briefings)
- All new adapters inherit BaseSource -> ParsedDocument -> markdown_to_chunks() pipeline
- frontend has named sessions sidebar with localStorage persistence

### Running:
- graphrag :8001, agent :8002, frontend :8080 (all healthy)
## [2026-05-05] create | concepts/graph-visualisation-diagnosis.md
- Full diagnostic: Neo4j schema, agent output shape, 5 root causes identified
- Fix plan: entity co-occurrence edges, topic nodes, chunk metadata, citation lookup, frontend layout
## [2026-05-05] create | services-refactor-phases-0-5 — all 5 phases complete, 54 tests pass, Google style enforced
## [2026-05-05] session-2 fixes | services-refactor-phases-0-5
- Fixed get_langchain_llm missing import in graphgen/extractors.py (root cause: Neo4j never populated)
- Fixed agent tool field aliasing: entity/entity_id/entity1_id → UPPER_SNAKE_CASE normalisation
- Fixed agent answer empty: orchestrator now accepts text|answer|content keys
- Added GRAPHRAG_URL/GRAPHGEN_URL/SEARCH_URL to .env (were missing — caused silent 8001 default)
- Fixed PYTHONPATH: /root/graphknows/services (not services/agent)
- Full SSE ReAct demo verified: 5-tool chain, 10 chunks retrieved, graceful Neo4j empty fallback
- Commits 3049d4a, c584958, 1c96d3a pushed to PR #10
- GraphGen build job_6f755304 running — will populate Neo4j with get_langchain_llm fix

## [2026-05-08] create | concepts/langgraph-agent-architecture-2026-05-08.md
- Architecture pivot: hand-rolled ReAct → LangGraph StateGraph
- Framework selection rationale: astream_events v2, ChatGroq, max_steps=6
- 4 tools defined: search_web, search_eu, graphgen, graphrag
- WS event schema v3: thought_token, tool_start, tool_result, answer_token, graph_update
- EPRS-style report format documented
- Dual graph tabs: Knowledge Graph + Lexical Graph
- All files to change listed

## [2026-05-08] bugs-fixed | live-search agent
- Removed GraphRAG tools from orchestrator tool registry (caused infinite loops)
- max_steps reduced 8→3 as emergency patch before LangGraph rewrite
- Fixed Step-N preamble leaking into answer text (both answer paths)
- Fixed graph_update WS payload shape (counts→full arrays)
- Added keyword fallback graph (regex noun phrases, no ML, fires immediately)
- RAM guard: skip GLiNER2 subprocess if <2 GB available, fall back to keyword graph
- graphgen_worker API aligned: extract_entities, correct label_space usage
- Tunnel URL: css-screensaver-seniors-finding.trycloudflare.com

## [2026-05-08] impl | langgraph-agent-v3 — fully working
- LangGraph StateGraph agent live on main branch (commit f1321ee)
- Python 3.13 compat fix: AgentState + add_messages at module level (not inside build_graph)
- should_continue() has no type hints (LangGraph introspects with get_type_hints)
- tool_choice="required" on first call forces search_eu before answering
- End-to-end test verified: 717 thought_token + 4 tool_start + 4 tool_result + 1 graph_update + 1 answer
- Keyword fallback graph fires (GLiNER2 OOM RAM guard: <2GB → fallback)
- KG: 1n/0e (fallback), Lexical: 4n/2e from doc terms
- Tunnel: css-screensaver-seniors-finding.trycloudflare.com

## 2026-05-10 — GLiNER2+DSPy graphgen pipeline restored; streaming partial graphs

### Changes
- **graphgen_worker.py** fully rewritten with correct PYTHONPATH (`/services/`, not `/services/graphgen/`)
  - GLiNER2 `knowledgator/gliner-multitask-large-v0.5` loads in ~12s, extracts typed entities
  - DSPy `GraphExtractorModule` with Groq llama-3.3-70b extracts `(source, relation, target, evidence, confidence)` triplets
  - Per-chunk pipeline: GLiNER2 → select_candidate_labels → DSPy; fallback per chunk if DSPy fails
  - Verified: `Ursula von der Leyen → signed → European Green Deal` (confidence 0.9)
- **langgraph_agent.py** redesigned for streaming UX:
  - `_emit_keyword_graph()` fires as `asyncio.create_task` after each search tool → first graph in ~8s
  - `_run_graphgen()` subprocess with 240s timeout for GLiNER2+DSPy
  - Entity names returned from graphgen tool result → LLM uses them in final report
  - `_session_entities` dict accumulates extracted names per session
- **index.html** live graph UX:
  - Auto-switch to KG tab on first `graph_update` event
  - Live badge: `⚡ 36n/108e — extracting…` → `✓ 60 entities / 180 relations`
  - `partial: true/false` field in graph_update distinguishes keyword vs GLiNER2 graph
- **router.py** thought_token batching at 350ms → 17 total WS frames (was 500+, was killing Cloudflare tunnel)

### Architecture decision: subprocess for GLiNER2
GLiNER2 loads in ~12s cold. Running it in-process inside the async agent would block the event loop. Subprocess isolation is the correct pattern. The 240s timeout accommodates 10 docs × ~4 chunks × GLiNER2 + DSPy per chunk.

### Verified E2E flow (localhost WS proxy)
- [4s] tool_start: search_eu + search_web (parallel)
- [8s] graph_update partial=True: 36n/108e (keyword, from EU docs)
- [9s] graph_update partial=True: 60n/180e (keyword, after web docs merged)
- [42s] graph_update partial=False: GLiNER2+DSPy final graph
- [44s] ANSWER with EPRS brief format citing entity names from graph
- 17 total WS frames

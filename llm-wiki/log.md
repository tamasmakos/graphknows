# Wiki Log

> Chronological record of all wiki actions. Append-only.
> Format: ## [YYYY-MM-DD] action | subject
> Rotate when >500 entries.

## [2026-05-03] create | Wiki initialized
- Domain: Political Intelligence & Knowledge Graph Research (GraphGen project)
- Structure: SCHEMA.md, index.md, log.md, raw/, entities/, concepts/, comparisons/, queries/

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

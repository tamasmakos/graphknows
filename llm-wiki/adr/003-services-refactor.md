---
title: Services Refactor — Phases 0–5
created: 2026-05-05
updated: 2026-05-05
type: concept
tags: [architecture, refactor, services, python, hexagonal]
sources: []
confidence: high
---

# Services Refactor — Phases 0–5

Complete restructure of the 4 graphknows microservices into hexagonal layouts.
Branch: `refactor/phase-1-graphrag`. All 54 tests pass, 5 skipped (live Memgraph).

---

## What Changed

### Phase 0 — Shared Foundation (`libs/python/graphknows-shared/`)
- Installable editable package `graphknows-shared 0.1.0`
- `constants.py`: EMBEDDING_MODEL, EntityLabel enum
- `models/entity.py`, `session.py`, `pipeline.py`: shared Pydantic models
- `config.py`: `BaseServiceSettings(BaseSettings)` with `check_production_secrets()`
- `python-json-logger` declared as dependency
- Root `pyproject.toml`: ruff `pydocstyle-convention = "google"` enforced

### Phase 1 — GraphRAG → Pure Retrieval
- **Deleted**: `agent.py`, `hybrid_agent.py`
- **Created**: `services/graphrag/retriever.py` — `HybridRetriever`
  - `retrieve(query, k)` → RRF fusion of pgvector + Neo4j
  - `get_path(a, b)` → BFS over Neo4j (`*1..5` hardcoded — no parameter support)
  - `get_chunks(entity_id)` → chunks from pgvector
  - `get_community(entity_id)` → Leiden community lookup
- **Refactored** `api.py`: 4 endpoints (`/retrieve`, `/path`, `/chunks`, `/community/{id}`)
- `NeoStore` has no context manager — never use `with neo_store as neo`

### Phase 2 — GraphGen REST (small delta)
- Fixed fragile intra-function import → module-level with startup error
- Added `POST /ingest` (accepts `Document` from shared-py)
- `GraphGenSettings(BaseServiceSettings)`
- `services/graphgen/Dockerfile` created

### Phase 3 — Agent Service Full Build
- `config.py`: `AgentSettings(BaseServiceSettings)` + MEMGRAPH_URI + downstream URLs
- `orchestrator.py`: `IterativeOrchestrator` — async ReAct loop (think→tool→observe→synthesize)
- `tools/`: `BaseTool` ABC + 6 concrete tools (retrieve, get_path, get_chunks, get_community, build, search)
- `session/manager.py`: `SessionGraphManager` (refactored from `session_graph.py`)
- `session/writeback.py`: `WriteBackService` — reads Memgraph, writes confirmed entities to Neo4j
- `router.py`: all HTTP+WS routes — `/sessions` CRUD, `/query` SSE, `/graph` D3 snapshot
- `main.py`: slim lifespan + `include_router`

### Phase 4 — Search Service
- `services/search/Dockerfile` (already existed from prior work)
- `router.py`: post-scrape fire-and-forget `POST graphgen/ingest` per document (httpx, errors logged)
- Silent stub adapters (council_conclusions, eca_reports, ecb_publications, eprs_briefings) → `raise NotImplementedError` with TODO docstring

### Phase 5 — Production Hardening
- Structured JSON logging (`_configure_logging()`) in all 4 services — `python-json-logger`
- `check_production_secrets()` called in all service lifespans
- `docker-compose.yaml`: `agent` (port 8030) + `search` (port 8040) services added
  - Agent depends on: neo4j (healthy), pgvector (healthy), memgraph (healthy), graphrag (started)
  - Docker service names used throughout (no hardcoded IPs in compose)

---

## Code Standards Enforced

All code follows [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html):
- Google-style docstrings (`pydocstyle-convention = "google"` in ruff)
- Type hints everywhere
- No bare `except`
- 100-character line limit
- This is now the law for all subagents and future work

---

## Key Constraints / Pitfalls

- `NeoStore` has **no context manager** — do not use `with NeoStore(...) as neo`
- Neo4j `shortestPath` does **not** accept `$param` for hop depth → hardcoded `*1..5` in `get_path()`
- OpenCode must run **foreground-only** with `LANG=C LC_ALL=C` prefix (background mode crashes)
- `GRAPHKNOWS_ENV=production` required to trigger secret validation
- `python-json-logger` must be installed in each service's pyproject.toml deps

---

## Test Status (2026-05-05)

| Suite | Status |
|---|---|
| test_graphrag_api.py | 11/11 pass |
| test_session_graph.py | 5/5 skipped (no live Memgraph) |
| All others | 38 pass |
| **Total** | **54 pass, 5 skip** |

---

## Bug Fixes Applied (2026-05-05 session 2)

| Bug | Root Cause | Fix |
|---|---|---|
| Agent tool 422s on `get_chunks`/`get_path` | LLM uses `entity`, `entity1`, `entity1_id` — wrong field names | Field aliasing + UPPER_SNAKE_CASE normalisation in each tool's `run()` |
| Agent answer always empty | LLM emits `{"answer": "..."}` instead of `{"text": "..."}` | Orchestrator accepts `text`\|`answer`\|`content` keys |
| Neo4j never populated | `get_langchain_llm` not imported in `extractors.py` | Added import from `graphgen.config.llm` |
| Agent using wrong GRAPHRAG_URL (8001 default) | URL vars missing from `.env` | Added `GRAPHRAG_URL`, `GRAPHGEN_URL`, `SEARCH_URL` to `.env` |
| Agent loading stale code after patch | PYTHONPATH was `/root/graphknows/services/agent` (wrong) | Corrected to `/root/graphknows/services`; `/tmp/restart_agent.sh` for hot-reload |

## Agent Demo Verified (2026-05-05)

Full SSE ReAct loop confirmed working end-to-end:
- 5-tool chain: `retrieve` → `get_community` → `get_chunks` → `search` → `get_path` → `answer`
- pgvector retrieval: 10 chunks, 4 entities returned on EP energy/Ukraine query
- Neo4j tools (`get_chunks`, `get_path`) return gracefully empty (Neo4j not yet populated)
- GraphGen build `job_6f755304` running with `get_langchain_llm` fix — will populate Neo4j

## Commits on PR #10

```
3049d4a fix(graphgen): add missing get_langchain_llm import in extractors.py
c584958 fix(agent): UPPER_SNAKE_CASE normalisation + empty entity guard
1c96d3a fix(agent): normalise LLM tool param field names; fix PYTHONPATH
```

## Related Pages
- [[architecture/reference]]
- [[architecture/service-agent]]
- [[architecture/product-architecture]]
- [[ops/technical-debt]]

## Deviation Note (2026-05-20)

The original ADR planned **4 microservices**: search / graphgen / graphrag / agent.

**What actually shipped:** 3 services — search / graphknows / agent.

`graphrag` was never deployed as a standalone service. `HybridRetriever` (the graphrag responsibility) was absorbed into `services/graphknows/graphknows/retrieval/`. `graphgen` was renamed `graphknows` and expanded to include retrieval, ontology, and MCP. The refactor's hexagonal layout and test suite (54 tests) remain valid for the current structure.

---
title: "Service Design — services/agent/"
created: 2026-05-04
updated: 2026-05-20
type: architecture
tags: [architecture, agent, memgraph, search, langgraph, service-design]
confidence: high
---

# Service Design — `services/agent/`

The agent service is the single application-facing backend. It owns session lifecycle, the 3-phase investigative workflow, dual-write (Memgraph + pgvector), hybrid retrieval, and final EPRS-style briefing generation.

---

## Responsibilities

- Receives chat requests from the frontend (HTTP `/chat` + WebSocket `/ws/{session_id}`)
- Creates and tracks investigation sessions
- Runs the 3-phase LangGraph workflow: `scope → investigate → communicate`
- Writes chunks to pgvector and entities/relations to the Memgraph session graph per iteration
- Streams progress events over WebSocket
- Returns final EPRS-style HTML brief + graph snapshot on `/chat`

---

## Directory Layout

```
services/agent/
  main.py                  # FastAPI app factory, lifespan, dependency wiring
  router.py                # Route definitions: /chat, /ws/{session_id}, /health
  config.py                # AgentSettings (Pydantic): GRAPHKNOWS_URL, SEARCH_URL,
                           #   MEMGRAPH_URI, PGVECTOR_DSN, etc.

  api/
    schemas.py             # Request/response Pydantic models for HTTP endpoints

  graph/
    workflow.py            # build_workflow() → compiled StateGraph (scope→investigate→communicate)
    state.py               # InvestigativeState TypedDict + SubQuestion/RawDoc/Chunk/Entity/Relation dataclasses
    events.py              # WS event constructors: search_start, docs_fetched, chunks_written,
                           #   entities_written, novelty_check, graph_update, brief_done, etc.
    stream_events.py       # astream_events v2 helpers for LangGraph streaming
    agent.py               # LangGraph agent wiring (tool → node binding)
    backend.py             # GraphBackend ABC — session graph abstraction
    html_report.py         # HTML brief renderer (EPRS format)
    viz.py                 # Graph snapshot serialiser (nodes/edges → JSON for frontend)

    backends/
      memgraph.py          # MemgraphBackend: per-session nodes/edges, bolt driver, MERGE queries
      neo4j_gds.py         # Neo4jGDSBackend: pluggable alternative for GDS-based reasoning

    nodes/
      scope.py             # scope_node: LLM decomposes topic into SubQuestions + source priorities
      investigate.py       # make_investigate_node(): search → chunk → embed → dual-write → novelty check
      communicate.py       # make_communicate_node(): hybrid retrieval → EPRS brief via LLM

  session/
    models.py              # Session Pydantic model: session_id, topic, status, created_at
    manager.py             # SessionGraphManager: per-session Memgraph graph, Neo4j fallback
    writeback.py           # WriteBackService: exports high-confidence entities to Neo4j post-session
    end.py                 # Session teardown: TTL expiry, Memgraph cleanup

  tools/
    base.py                # BaseTool ABC

    search/
      search.py            # SearchTool: parallel EU + Tavily search, BM25 re-rank
      client.py            # SearchServiceClient: HTTP client for services/search/
      web.py               # TavilySearchTool: Tavily API wrapper
      cellar.py            # CellarSPARQLTool: direct SPARQL queries to EU CELLAR endpoint
      router.py            # route_sources(): maps SubQuestion.source_priorities → connector IDs
      ranking.py           # BM25 + date-decay document ranker

    graph/
      build.py             # build_session_graph(): keyword + GLiNER2 entity graph construction
      extract.py           # Entity/relation extraction helpers
      client.py            # GraphServiceClient: HTTP client for services/graphknows/ (optional mode)

    extraction/
      in_process.py        # In-process GLiNER2 + DSPy extraction (no HTTP hop to graphknows)

  ontology/
    loader.py              # OntologyLoader: reads OWL/JSONLD assets
    router.py              # OntologyRouter: embedding-similarity → keyword → LLM classifier
    rel_extractor.py       # OntologyRelExtractor: typed ONTOLOGY_REL edges from DSPy
    assets/                # legal/, politics/, finance/, science/ ontology files + registry.json
```

---

## 3-Phase Workflow (workflow.py)

```
scope_node
    │
    ▼
investigate_node ◄──────────────────────┐
    │                                   │
    │  should_continue():               │
    │    dup_ratio > 0.6 → done         │
    │    iteration_count >= 12 → done   │
    └── continue ──────────────────────►┘
    │
    done
    │
    ▼
communicate_node
    │
    ▼
  END
```

**scope_node** — decomposes the research topic into `SubQuestion` objects, each with `source_priorities` (e.g. `["eprs_briefings", "ep_debates", "web"]`).

**investigate_node** — per iteration:
1. Runs `SearchTool` (EU connectors + Tavily) in parallel
2. Chunks and embeds results (BAAI/bge-small-en-v1.5)
3. Dual-writes: chunks → pgvector, entities/relations → Memgraph session graph
4. Computes novelty signal (`dup_ratio`) via cosine similarity on chunk embeddings
5. Emits `graph_update` WS event with current node/edge counts

**communicate_node** — retrieves top-K context via `HybridRetriever`, constructs `NodeFeatures` from the session graph, and generates the EPRS-style 7-section brief via an LLM (Groq / OpenAI-compatible).

---

## WebSocket Event Schema

| Event type | When fired | Key fields |
|---|---|---|
| `search_start` | Before each search iteration | `session_id`, `iteration`, `queries` |
| `docs_fetched` | After search returns | `session_id`, `doc_count`, `sources` |
| `chunks_written` | After pgvector write | `session_id`, `chunk_count` |
| `entities_written` | After Memgraph write | `session_id`, `entity_count`, `relation_count` |
| `novelty_check` | After dedup computation | `session_id`, `dup_ratio`, `decision` |
| `graph_update` | After each iteration write | `session_id`, `nodes`, `edges`, `partial` |
| `retrieval_done` | After hybrid retrieval | `session_id`, `chunk_count` |
| `brief_done` | After brief generation | `session_id`, `word_count` |

---

## Key Config (AgentSettings)

| Env var | Default | Purpose |
|---|---|---|
| `GRAPHKNOWS_URL` | `http://localhost:8025` | graphknows service (optional HTTP mode) |
| `SEARCH_URL` | `http://localhost:8000` | search service |
| `MEMGRAPH_URI` | `bolt://172.18.0.1:17687` | Memgraph session graph |
| `PGVECTOR_DSN` | `postgresql://postgres:***@172.18.0.1:5432/vectordb` | pgvector |
| `SESSION_GRAPH_BACKEND` | `memgraph` | `memgraph` or `neo4j_gds` |
| `GRAPHKNOWS_MODE` | `http` | `http` (call graphknows service) or `inprocess` |
| `LLM_PROVIDER` | `groq` | LLM provider |
| `LLM_MODEL` | `llama-3.3-70b-versatile` | LLM model name |

---

## Notes

- `orchestrator.py` exists in the repo as a **legacy file** from the pre-LangGraph ReAct era. It is not in the main request path. Do not modify or reference it for new features.
- `tools/extraction/in_process.py` calls the graphknows package directly (no HTTP), which is the default path. The `GRAPHKNOWS_MODE=http` path exists for future distributed deployment.
- Session graph is **ephemeral**: Memgraph nodes are namespaced by `session_id` and cleaned up on session end. `WriteBackService` optionally promotes high-confidence entities to the persistent Neo4j corpus.

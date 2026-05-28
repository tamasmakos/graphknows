---
description: "Read this at the beginning at every session"
applyTo: "**"
---

# graphknows — Project Description

## What It Is

**graphknows** is a framework-agnostic agentic memory system: a Python library and service that lets any AI agent ingest, store, and retrieve knowledge across three memory layers without coupling to any specific agent framework (LangChain, OpenAI Agents SDK, PydanticAI, etc.).

```python
from graphknows import GraphKnows

async with GraphKnows.from_env() as gk:
    tools = gk.tools          # list[GraphKnowsTool] — name + description + JSON Schema + async callable
    ctx   = await gk.retriever.retrieve("my query", session_id="s1")
```

Agent frameworks wrap the tools in one line at the call site. No framework is imported by this package.

---

## Three-Layer Memory Architecture

| Layer | Store | Technology | Role |
|-------|-------|-----------|------|
| **Long-term (LTM)** | `Neo4jStore` + `ltm_*` pgvector tables | Neo4j + pgvector | Durable knowledge graph + durable vectors after promotion/flush |
| **Short-term (STM)** | `MemgraphStore` + `stm_*` pgvector tables | Memgraph (Bolt) + pgvector | Live session evidence graph + session-scoped chunk/turn vectors |
| **Semantic** | `PgVectorStore` | pgvector (asyncpg) | ANN recall over STM/LTM chunks, turns, and topics |

Sessions can be flushed explicitly and may auto-merge after `SESSION_TTL_MINUTES` of inactivity. Promotion moves both graph structure and vector evidence from STM into LTM.

---

## Core Mental Model

`graphknows` is not a single linear system. It is two coupled pipelines that share retrieval and stores:

1. **Standalone corpus ingestion** (`graphgen`) turns external documents into durable graph structure.
2. **Session memory ingestion** (`memory`) writes live agent interaction into STM first, then later promotes that session into LTM via `flush`.

If there were no STM/LTM split, the standalone `graphgen` pipeline would be the whole story. In the actual architecture, raw session ingestion and durable topic formation are separated:

- **STM capture** is fast, local, and session-scoped.
- **`flush`** is the STM → LTM boundary where evidence is resolved, promoted, enriched, optionally summarized into topics, and optionally cleared from STM.
- **Topic summarization is a flush-time concern**, not a raw STM-ingest concern.
- **Feature engineering also lives on that promotion boundary**: entity resolution, relation extraction, optional GDS metrics, community IDs, and topic materialization are what turn ephemeral session evidence into durable LTM structure.


## Standalone Graphgen Pipeline

This is the generic document-to-knowledge-graph pipeline. It is the right mental model for direct corpus ingestion, but it is **not** the full story for session memory promotion.

1. **Lexical graph** — chunk text and build the initial evidence graph.
2. **Extraction** — extract entities and relations from chunks.
3. **Enrichment** — semantic resolution, embeddings, and evaluation.
4. **Communities** — detect communities and compute graph structure signals.
5. **Topic analysis** — score partitions and topic quality.
6. **Pruning** — clean noisy graph structure.
7. **Upload** — persist durable graph state with `SEGMENT` provenance.

Entry: `graphgen.orchestrator.run_pipeline(input_files, config_override, output_dir)`

In the STM/LTM architecture, those same ideas still matter, but **session topic summaries are not created during raw STM ingest**. They are created during `flush`, after session evidence has been promoted into LTM.

---

## STM → LTM Flush Pipeline

`graphknows.memory.flush.SessionFlusher` is the actual promotion path for agent/session memory.

1. **Read STM subgraph** from Memgraph via `read_session_subgraph(session_id)`.
2. **Resolve duplicate entities** inside the session by canonical lowercase name and highest confidence.
3. **MERGE the evidence graph into Neo4j**: `SESSION -> FILE -> CHUNK -> ENTITY`.
4. **MERGE the interaction graph into Neo4j**: `SESSION -> DIALOG -> TURN`, plus `NEXT` and turn/entity links.
5. **Extract and write typed entity relations** from chunk evidence, creating durable `ENTITY-[RELTYPE]->ENTITY` edges.
6. **Promote vectors** from `stm_chunks` / `stm_turns` into `ltm_chunks` / `ltm_turns`.
7. **Optionally summarize topics**: build communities over flushed entities, gather evidence chunks, LLM-summarize them, then write Neo4j topics plus `ltm_topics` vectors.
8. **Optionally run feature engineering / GDS enrichment**: Node2Vec, cosine edge weights, PageRank, degree, betweenness, and community properties over the promoted LTM entity graph.
9. **Optionally clear STM**: delete the Memgraph session graph and `stm_*` pgvector rows after successful promotion.

Current implementation detail: during `flush`, topic summarization runs before optional GDS enrichment.

---

## STM Schema

### Memgraph graph layer

- `SESSION {id, ...metadata}`
- `FILE {id, session_id, title, parser_name, source_hint}`
- `CHUNK {id, session_id, file_id, text, heading_path, chunk_index, metadata}`
- `ENTITY {name, session_id, type, confidence}`
- `DIALOG {id, session_id, started_at}`
- `TURN {id, session_id, dialog_id, turn_index, role, text, timestamp}`
- `TOOL_CALL {turn_id, args_hash, name, latency_ms}`

Core STM edges:

- `(SESSION)-[:IN_SESSION]->(FILE)`
- `(SESSION)-[:IN_SESSION]->(DIALOG)`
- `(FILE)-[:HAS_CHUNK]->(CHUNK)`
- `(CHUNK)-[:MENTIONS]->(ENTITY)`
- `(DIALOG)-[:HAS_TURN]->(TURN)`
- `(TURN)-[:NEXT]->(TURN)`
- `(TURN)-[:MENTIONS {confidence}]->(ENTITY)`
- `(TURN)-[:CALLED]->(TOOL_CALL)`
- Optional session entity relations: `[:REL {name, session_id, fact, confidence}]`
- Legacy/compat edges still accepted on read: `[:HAS_ENTITY]` and `[:RELATES {type, session_id}]`

### pgvector STM tables

- `stm_chunks(id, session_id, file_id, text, heading_path, embedding, metadata, created_at)`
- `stm_turns(id, session_id, dialog_id, turn_index, role, text, embedding, created_at)`

---

## LTM Schema

### Durable corpus graph from `graphgen`

This is the document/corpus side of LTM, defined by `graphgen.config.schema`:

- Nodes: `SEGMENT`, `DOC`, `PASSAGE`, `CHUNK`, `ENTITY`, `TOPIC`, `SUBTOPIC`
- Core hierarchy: `SEGMENT-[:CONTAINS_DOC]->DOC-[:HAS_PASSAGE]->PASSAGE-[:HAS_CHUNK]->CHUNK-[:HAS_ENTITY]->ENTITY`
- Higher-level structure: `ENTITY-[:**]->ENTITY`, `ENTITY-[:IN_TOPIC]->SUBTOPIC-[:PARENT_TOPIC]->TOPIC`
- `SEGMENT` is the provenance anchor for each durable pipeline run and carries fields such as `segment_id`, `ingested_at`, `run_id`, `git_revision`, `config_hash`, `pipeline_version`, `source_count`, `input_dir`, and `taxonomy_name`

### Durable session graph written during `flush`

This is the session-memory side of LTM, written into Neo4j during promotion:

- Nodes: `SESSION`, `FILE`, `CHUNK`, `ENTITY`, `DIALOG`, `TURN`, topic nodes
- Core edges: `(SESSION)-[:IN_SESSION]->(FILE|DIALOG)`, `(FILE)-[:HAS_CHUNK]->(CHUNK)`, `(CHUNK)-[:HAS_ENTITY]->(ENTITY)`, `(DIALOG)-[:HAS_TURN]->(TURN)`, `(TURN)-[:NEXT]->(TURN)`, `(TURN)-[:HAS_ENTITY]->(ENTITY)`, dynamic typed `ENTITY->ENTITY` relation edges, and `(ENTITY)-[:IN_TOPIC]->(Topic)`
- `ENTITY` is globally merged in LTM by `name`; session provenance is retained by the surrounding `SESSION`, `FILE`, `CHUNK`, `DIALOG`, and `TURN` structure rather than by keeping separate per-session entity nodes in Neo4j
- Core uniqueness constraints exist for `SESSION.id`, `FILE.id`, `CHUNK.id`, `ENTITY.name`, `DIALOG.id`, `TURN.id`, and topic id

### pgvector durable tables

- `ltm_chunks(id, session_id, file_id, text, heading_path, embedding, metadata, flushed_at)`
- `ltm_turns(id, session_id, dialog_id, turn_index, role, text, embedding, flushed_at)`
- `ltm_topics(topic_id, summary, embedding, entity_names[], session_ids[], created_at)`

---

## Retrieval: HybridRetriever

Located at `graphknows/retrieval/retriever.py`. Supports six modes:

| Mode | Strategy |
|------|----------|
| `HYBRID` (default) | RRF(K=60) fusion of GRAPH_TRAVERSAL + SEMANTIC |
| `SEMANTIC` | Pure pgvector ANN on chunks |
| `GRAPH_TRAVERSAL` | Entity match → multi-hop  neighbors → chunks |
| `SUMMARIES` | Entity match → Leiden community topic summaries |
| `FEELING_LUCKY` | Auto-route based on query heuristics |
| `CYPHER` | Parameterized read-only Cypher (injection-guarded) |

Output: `RetrievalContext` (entity-first + semantic + RRF-fused, XML-serialisable).

---

## MCP Tools (13 tools via FastMCP)

Available via `python -m graphknows.mcp.stdio_server` (stdio) or mounted at `/mcp/` in the REST API:

`retrieve`, `entity_search`, `entity_neighbors`, `get_chunks`, `session_memory`, `ingest`, `poll_job`, `get_session_graph`, `get_corpus_stats`, `community_summary`, `list_segments`, `segment_info`, `cypher_query`

---

## Key Design Conventions

- **Framework-agnostic**: No agent framework imported inside `graphknows/`. Adapters are caller-side.
- **Async-first**: All I/O is async. Neo4j sync driver is wrapped in `asyncio.to_thread()`.
- **Dependency injection**: Stores are injected via constructor or `set_dependencies()`. No globals in library code.
- **Two coupled pipelines**: `graphgen` handles standalone corpus ingestion; `memory.flush` handles STM → LTM promotion for live agent memory.
- **Session isolation**: Every retrieval/ingest call accepts `session_id` to scope data. Sessions can be flushed explicitly and may auto-merge to corpus on TTL expiry.
- **Flush-only topic materialization**: raw STM ingestion captures evidence; topic summaries are produced when a session is flushed into LTM.
- **Feature engineering at the promotion boundary**: the important enrichment step is not raw ingest alone, but the STM → LTM promotion path where entities, relations, communities, and graph metrics become durable.
- **Idempotent writes**: Neo4j uses MERGE; pgvector uses upsert. Safe to replay.
- **Config from environment**: All settings via `GraphKnowsSettings` (pydantic-settings, `.env`, no hardcoded URLs).
- **SEGMENT provenance**: Every ingestion batch writes a SEGMENT node in Neo4j for audit trail.
- **Cypher safety**: All Cypher uses parameterized queries. The `cypher_query` MCP tool rejects write keywords and string interpolation.

---

## Entry Points

| Use Case | Command / Import |
|----------|-----------------|
| Library (framework-agnostic) | `from graphknows import GraphKnows; async with GraphKnows.from_env() as gk` |
| MCP stdio (Claude/Cursor) | `python -m graphknows.mcp.stdio_server` |
| Knowledge pipeline | `from graphknows.graphgen.orchestrator import run_pipeline` |
| Direct retrieval | `from graphknows.retrieval.retriever import HybridRetriever` |
| Direct stores | `from graphknows.retrieval.stores import Neo4jStore, PgVectorStore, MemgraphStore` |

---

## Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `NEO4J_URI` | `bolt://localhost:7687` | Long-term memory store |
| `NEO4J_USER` / `NEO4J_PASSWORD` | — | Neo4j credentials |
| `PGVECTOR_DSN` | — | Semantic memory (asyncpg DSN) |
| `MEMGRAPH_URI` | `bolt://localhost:7688` | Short-term session graph |
| `EMBED_MODEL` | `BAAI/bge-small-en-v1.5` | Sentence embedding model (384 dims) |
| `SESSION_TTL_MINUTES` | `60` | Idle session auto-merge threshold |

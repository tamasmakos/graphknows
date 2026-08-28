# Architecture

graphknows is an agentic memory system. It ingests conversations and documents,
builds a knowledge graph over [ArcadeDB](https://arcadedb.com), and serves hybrid
retrieval to an agent. This document traces the moving parts and where each lives.

## Public entry point

Everything is reached through `Memory` (`graphknows/memory.py`), a
transport-neutral, async facade. It is wrapped by the MCP stdio server
(`graphknows/server/mcp/`), but the class is usable directly:

```python
from graphknows import Memory

async with Memory() as mem:
    await mem.ingest_memory(messages, session_id="s1")
    await mem.flush_memory("s1")
    hits = await mem.recall_memory("who did Caroline meet?", session_id="s1")
```

Async methods on the facade:

| Method | Purpose |
| --- | --- |
| `ingest_memory` | Ingest text or a message list into the graph (lifecycle state `raw`). |
| `recall_memory` | Ranked hybrid recall; returns fused hits plus a fact sheet. |
| `flush_memory` | Consolidate one session in place: entity resolution, analytics, `raw` → `consolidated`. |
| `purge_memory` | Delete the namespace's data (all, or one session). |
| `drop_namespace` | Drop the whole namespace database (refused for the default namespace). |
| `stats` | Node counts for the namespace, chunks split by lifecycle state. |
| `doctor` | Backend connectivity check for the namespace database. |
| `corpus_ingest` | Batch-ingest documents into one session, then consolidate it. |
| `ltm_topics` / `ltm_entity` / `ltm_entities` | Read-only graph browsing helpers. |

A namespace-first SDK surface (`add`, `search`, `flush`) wraps the same verbs with
mem0-style `user_id` / `agent_id` / `run_id` scoping (`Memory._scoped_session_id`).

## Configuration and modes

Config is `GraphKnowsSettings` (`graphknows/settings.py`), a `pydantic-settings`
model. Every field maps to a `GRAPHKNOWS_`-prefixed environment variable (also
read from a `.env` file). Key ones: `GRAPHKNOWS_MODE`, `GRAPHKNOWS_NAMESPACE`,
`GRAPHKNOWS_ARCADEDB_URL`, `GRAPHKNOWS_LLM_MODEL`, `GRAPHKNOWS_LLM_API_KEY`,
`GRAPHKNOWS_EMBED_MODEL`, `GRAPHKNOWS_EMBED_API_BASE`, `GRAPHKNOWS_ONTOLOGY_FILE`,
`GRAPHKNOWS_ENABLE_FRAMES`, `GRAPHKNOWS_FACT_CONTEXT`.

`GRAPHKNOWS_MODE` (enum `MemoryMode`) governs extraction. There is no separate
profile object: the mode fork lives in exactly one factory,
`build_relation_extractor`
(`graphknows/ingestion/extraction/relations/__init__.py`), on the write path.
Retrieval does not fork — `build_retriever`
(`graphknows/retrieval/__init__.py`) always returns `DETRetriever`.

| Mode | Extra extraction | Retrieval | API keys |
| --- | --- | --- | --- |
| `llm_free` (default) | GLiNER2 entities + its precision-gated relations; no LLM extractor | `DETRetriever` (deterministic, channel fan-out + RRF) | none |
| `llm_assisted` | additionally DSPy ontology-guided relation extraction | same `DETRetriever` | LLM + optional embed |

Embeddings default to a local `sentence-transformers` model, so `llm_free` runs
with zero API keys.

## Memory model: one golden-layer database

There is **one physical ArcadeDB database per namespace** — `mem` for the default
namespace, `mem_<ns>` otherwise (`graphknows/storage/namespace.py`, `db_name`).
The old three-store split (separate `stm` / `ltm` databases plus a vector store)
is gone: `GraphStore` (`graphknows/storage/arcadedb/graph_store.py`) absorbs all
of it, including embeddings, which live directly on `CHUNK` / `ENTITY` / `TOPIC` /
`FRAME` vertices behind `LSM_VECTOR` indexes.

Short-term versus long-term is a **lifecycle property, not a database**. A `state`
in `{raw, consolidated}` sits on `SESSION` / `CHUNK` / `ENTITY` and on relation
edges; every write lands `raw`, and flush flips a session to `consolidated` in
place. Retrieval scope maps onto that filter (`_scope_to_state` in
`graphknows/retrieval/retriever.py`): `stm` → `raw`, `ltm` → `consolidated`,
`both` → no filter.

### Schema

The DDL is a pure ordered statement list, `_CORE_DDL`
(`graphknows/storage/arcadedb/_schema.py`), applied idempotently by
`GraphStore.ensure_schema(dims)` — `dims` is substituted into the vector indexes so
they match the configured embedder.

- Vertices: `SESSION`, `TURN`, `FILE`, `CHUNK`, `ENTITY` (keyed on `name_norm`),
  `TEMPORAL` (one node per ISO date), `TOPIC`, `FRAME` (canonical — one vertex
  per frame type for the whole namespace, keyed by the frame name),
  `FRAME_INSTANCE`, `ONTOLOGY`, `ONTOLOGY_CLASS`.
- Edges: `IN_SESSION` (FILE→SESSION), `PART_OF` (CHUNK→FILE), `MENTIONS`
  (CHUNK→ENTITY), `MENTIONS_DATE` (CHUNK→TEMPORAL),
  `IN_TOPIC` (ENTITY→TOPIC, with a
  `pagerank` property — the cluster membership itself, and the only persisted
  topic edge), `EVOKED_BY`
  (FRAME→CHUNK, with the per-chunk `confidence` — a canonical FRAME carries no
  chunk or session of its own, so everything per-evocation lives here),
  `BROADER` (the subsumption taxonomy, FRAME→FRAME and
  ONTOLOGY_CLASS→ONTOLOGY_CLASS), `FRAME_EVOKED_IN` (FRAME_INSTANCE→CHUNK),
  `PLAYS_ROLE` (FRAME_INSTANCE→ENTITY, with a `role` property), `EVOKES`
  (FRAME_INSTANCE→FRAME, the fact plane's link into the frame taxonomy, since
  `BROADER` hangs off the canonical FRAME vertex), and `REL`.
- `REL` carries **every** ENTITY→ENTITY fact; the predicate is the `predicate`
  property, not the edge type name. The vocabulary is open and mostly single-use,
  so a type-per-predicate minted hundreds of near-empty types plus a DDL
  round-trip on the ingest hot path.

Schema conventions are `type` not `label`, `confidence` not `score`, and
`valid_at` / `invalid_at` / `is_current` for bi-temporal relation edges.

**Bi-temporal supersedence.** For a known single-valued predicate
(`_SINGLE_VALUED_RELATIONS` — `WORKS_AT`, `LIVES_IN`, `MARRIED_TO`, …) a new fact
tombstones the existing current edge via `GraphStore._supersede_edge`: it sets
`invalid_at` and `is_current = false` rather than deleting, preserving history.

### Ingest

`ingest_memory` builds an `STMService` through `build_stm_service`
(`graphknows/ingestion/__init__.py`) and routes on input shape
(`graphknows/ingestion/stm/service.py`):

- A structured message list goes through `ingest_messages`, writing one `CHUNK`
  per message with `kind='turn'`.
- Plain text goes through `ingest` → `_STMIngestHandler`
  (`graphknows/ingestion/stm/ingest.py`): the text is parsed
  (`graphknows/ingestion/parsers/`) and chunked into `FILE` + `CHUNK` nodes.

Both paths converge on the same write helper, which per chunk:

1. embeds the text and writes it onto the `CHUNK` vertex;
2. runs `GLiNER2EntityExtractor`
   (`graphknows/ingestion/extraction/entities/extractor.py`) and writes `ENTITY`
   nodes plus `MENTIONS` edges;
3. writes the extractor's open-vocabulary relations as `REL` edges with
   chunk-level evidence provenance (no ontology filter — it would reject every
   open-vocab label);
4. in `llm_assisted`, additionally runs the injected DSPy relation extractor and
   writes its triplets as `REL` edges with `source='dspy'`;
5. when `enable_frames` is on, matches FrameNet frames by embedding
   (`graphknows/frames/index.py`) into `FRAME` / `EVOKED_BY`, and
   role-fills `FRAME_INSTANCE` / `PLAYS_ROLE` via
   `graphknows/ingestion/extraction/relations/frame_srl.py` — role filling is
   scoped to the trigger's sentence, not its whole chat turn, so two events on
   one turn don't share fillers; an instance is written only once at least two
   role fillers resolve to an `ENTITY`, and a same-chunk sibling with a strict
   subset of another's fillers collapses into it;
6. resolves dates (`resolve_temporal` in `graphknows/temporal.py`) into
   `TEMPORAL` nodes and `MENTIONS_DATE` edges.

`OntologyPredicateMapper` (`graphknows/ingestion/extraction/relations/filter.py`)
remains available for ontology-constrained predicate mapping, but the hot ingest
path does not filter GLiNER2 labels through it.

### Flush / consolidation

`Memory.flush_memory(session_id)` consolidates a session **in place** — there is
no promotion pipeline and nothing is copied between databases. Each step is
awaited through a local `step()` helper that records a named error instead of
aborting, so an analytics failure never skips the state flip:

1. **Entity resolution** — `resolve_entities`
   (`graphknows/ingestion/consolidation/entity_resolution.py`), a layered
   most-confident-first resolver: junk/hub filter, name-standardization merge for
   non-person entities, then person-nickname merge. Embedding similarity is
   deliberately not the primary mechanism (a true nickname pair scores below two
   distinct co-occurring people).
2. **Analytical view** — `GraphStore._ensure_analytical_view` is rebuilt once and
   shared by the three algorithms below (`skip_view_ensure=True`), since rebuilding
   is not free.
3. **PageRank** — `compute_and_persist_pagerank`, native ArcadeDB `algo.*` with a
   numpy power-iteration fallback; persisted onto `ENTITY.pagerank`.
4. **Graph embeddings** — `compute_graph_embeddings` → `ENTITY.graph_embedding`.
5. **Communities** — `compute_communities` → `ENTITY.community_id`.
6. **Community quality** — `compute_community_quality`, writing modularity /
   conductance metrics onto `COMMUNITY` vertices.
7. **State flip** — `consolidate_session(session_id)` moves the session's chunks
   from `raw` to `consolidated`. Nothing is deleted; relation canonicalization and
   topic summarization are not part of this path.

`corpus_ingest` is a thin loop: `ingest_memory` per document into one session,
then a single `flush_memory`. Documents without text are recorded in `errors` and
skipped.

## Storage: ArcadeDB

ArcadeDB is reached through a thin async httpx client, `ArcadeDBClient`
(`graphknows/storage/arcadedb/client.py`), wrapping the REST API (query, command,
transactions) with HTTP Basic Auth. `graphknows/storage/__init__.py` holds the
namespace-aware factories — `build_arcadedb_client`, `build_graph_store` — which
return **unconnected** stores so the client binds to the caller's event loop when
`connect()` is awaited.

ArcadeDB constraints the store works around, documented in its module docstring:
Cypher has no DDL surface (schema is SQL); embedding values must be inline SQL
literals because LIST params conflict with the `LSM_VECTOR` index; openCypher has
no `IN $list`, so every interpolated string goes through
`graphknows/storage/arcadedb/_sql.py`.

`ArcadeDBSTMStore` and `ArcadeDBLTMStore` still exist in
`graphknows/storage/arcadedb/` and back the read-only browsing helpers and the
web server's graph views; the ingest and retrieval paths use `GraphStore` only.

## Embeddings

`graphknows/storage/embedder.py` provides `embed` / `embed_one` (LRU-cached),
and `embed_dim`. By default a local
`sentence-transformers` model (`BAAI/bge-small-en-v1.5`, 384-dim) runs in-process
on CPU — no API key. Setting `GRAPHKNOWS_EMBED_API_BASE` routes embeddings through
a remote OpenAI-compatible `/embeddings` endpoint instead (model and key from
`GRAPHKNOWS_EMBED_MODEL` / `GRAPHKNOWS_EMBED_API_KEY`). There are no silent
embedding fallbacks — errors propagate.

## Retrieval

`recall_memory` builds the retriever lazily (`Memory._get_retriever`, guarded by
an `asyncio.Lock`).

### Channels

A **channel** (`graphknows/channels/base.py`) is one memory signal exposing
`collect(ctx, rt, top_k)` — ranked query candidates as
`{chunk_id: (score, info)}`. `default_channels`
(`graphknows/channels/registry.py`) returns the live set; insertion order defines
fusion order.

The baseline signals (entity, vector, bm25, temporal) are collected directly by
the retriever spine, with frames and ontology added as optional channels. The old
STM/LTM channel split collapsed with the databases, so each queries the one
`GraphStore` under a lifecycle `state` filter instead of routing to a physical
store:

- `entity` — entity-graph traversal from query-token entity nodes
- `vector` — dense ANN over chunk embeddings (plus a declarative query variant)
- `bm25` — lexical search (`rank_bm25` when importable, tokenised fallback otherwise)
- `temporal` — date-anchored traversal (`graphknows/temporal.py`)

Only the optional signals are `Channel`s: `FrameChannel`
(`graphknows/channels/frames.py`, name `frame`) when `settings.enable_frames`,
and `OntologyChannel` (`graphknows/channels/ontology.py`, name `ontology`) when
`settings.enable_ontology_channel`. A graph-traversal channel (`ppr`) and a
`topic` channel both existed and were removed — `ppr` returned the worst
gold-hit ratio of six channels, and nothing on the read path consumed what the
topic stage wrote. See the `registry.py` docstring for the measurements,
and [topics.md](topics.md) for what replaced the topic channel.

### DETRetriever (llm_free)

`DETRetriever` (`graphknows/retrieval/retriever.py`) is the channel-blind spine:
it builds a `ChannelContext` (query embedding + declarative variant, two token
sets), fans the channels out concurrently with `asyncio.gather`, then fuses and
ranks. A failure in any channel raises loudly rather than silently degrading
recall.

Candidates are fused by **RRF** (`graphknows/ranking/rrf.py`, `rrf_score`,
`RRF_K = 60`), then shaped by community boost (`community_boost` over
`GraphStore.chunk_community_weights`, derived from Louvain `community_id` on
`MENTIONS`-linked entities) and memory-class weighting (`shape_score`) before
sorting. Cosine ordering of the fused pool is the final rank: the cross-encoder
reranker was removed after measurement (evidence_recall 0.636 → 0.692 without
it).

PageRank shaping and IDF discriminativeness were removed: the former read
`pagerank` from CHUNK metadata while the value lives on `ENTITY`, so the
multiplier was always 1.0, and nothing ever called the latter.

Separately from the ranked passages, `_fact_sheet` resolves the query's entities
(`GraphStore.resolve_query_entities`) and returns their typed relations and
frame-instance role fills (`GraphStore.entity_fact_context`) as a compact fact
sheet on `RetrievalContext.facts`. It soft-fails and is disabled by
`GRAPHKNOWS_FACT_CONTEXT=false`.

`DETRetriever` is the only retriever. The agentic (DSPy ReAct) retriever, its
query decomposition, and the `GRAPHKNOWS_RETRIEVAL` strategy knob that selected
it have all been removed.

## Transports

- **MCP stdio** — `graphknows/server/mcp/stdio_server.py` (console script
  `graphknows-mcp`) over the shared FastMCP app in `graphknows/server/mcp/_app.py`.
  `graphknows/server/mcp/tools/__init__.py` is the authoritative inventory:
  `memory_ingest`, `memory_flush`, `memory_query`, `memory_topics`,
  `memory_stats`, `memory_doctor`, `memory_purge`, `memory_drop_namespace`,
  `corpus_ingest`, `ltm_topics`, `ltm_entity`, `ltm_entities`. A thin
  wrapper over one `Memory` instance.
- **CLI** — `graphknows-topics` (`graphknows/cli/topics.py`) rebuilds or sweeps
  the topic layer against a namespace. See [topics.md](topics.md).

## Ingest path

```mermaid
flowchart TD
    A[ingest_memory: text or messages] --> B{structured messages?}
    B -- yes --> C[ingest_messages: one CHUNK per turn]
    B -- no --> D[_STMIngestHandler: parse + chunk -> FILE + CHUNK]
    C --> E[GLiNER2 entities + relations, embed chunk]
    D --> E
    E --> F[ENTITY + MENTIONS, REL with chunk evidence]
    E --> G[frames: FRAME / FRAME_INSTANCE / PLAYS_ROLE]
    E --> H[TEMPORAL + MENTIONS_DATE]
    E --> I[llm_assisted: DSPy triplets -> REL]
    F & G & H & I --> DB[(GraphStore: one ArcadeDB db 'mem_ns', state=raw)]

    DB --> J[flush_memory]
    J --> K[resolve_entities: junk filter, standardize, nickname merge]
    K --> L[analytical view -> pagerank / graph embeddings / communities / quality]
    L --> M[consolidate_session: state raw -> consolidated]
    M --> DB
```

## Retrieval path

```mermaid
flowchart TD
    Q[recall_memory: query] --> DET[DETRetriever]

    subgraph DETRetriever
        CTX[ChannelContext: embed + tokenise, state]
        CTX --> C1[entity]
        CTX --> C2[vector]
        CTX --> C3[bm25]
        CTX --> C4[temporal]
        CTX --> C5[frame - optional]
        CTX --> C6[ontology - optional]
        C1 & C2 & C3 & C4 & C5 & C6 --> RRF[RRF fusion]
        RRF --> SHAPE[community boost + memory-class shaping]
        SHAPE --> FS[cosine order + fact sheet]
    end

    DET --> OUT[fused hits + facts]
```

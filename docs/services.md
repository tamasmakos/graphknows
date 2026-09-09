# The main services

GraphKnows is a **modular monolith**, not a set of microservices: one installable
package, one process, one deployment unit. "Service" here means an *enforced
layer* — a package with a single responsibility and a declared position in the
import order.

That order is not aspirational prose. It lives in
[`.importlinter`](../.importlinter) and is checked on every CI run, so this page
cannot quietly drift out of sync with the code the way a hand-drawn diagram
would. If the table below is wrong, the build is red.

## The layer stack

Top depends on bottom; nothing depends upward. Packages sharing a row are peers
and must not import each other.

```
  cli  |  server  |  integrations      ← entry points (transports)
                 memory                ← the facade every transport calls
      ingestion  |  retrieval          ← the write path and the read path
                channels               ← optional, symmetric memory signals
                ontology               ← domain vocabulary
        ranking  |  topics             ← fusion maths, clustering stage
                storage                ← ArcadeDB + embeddings
  models  |  settings  |  llm          ← layer-neutral vocabulary & config
               exceptions              ← the error taxonomy
```

Two extra contracts sharpen it:

- **`ingestion` and `retrieval` may not import each other.** The write path and
  the read path meet only at `storage`. This is what keeps a retrieval change
  from silently altering what gets written.
- **`integrations.client` stays stdlib-only.** The out-of-process SDK may not
  import `ingestion`, `retrieval`, `storage`, `channels` or `memory` — so a
  consumer can drive the server over MCP without pulling ~2.5 GB of ML
  dependencies into its own process.

## What each package owns

| Package | Owns | Start reading at |
| --- | --- | --- |
| `memory` | The runtime facade. Transport-neutral; owns store lifecycle, turn buffering and flush ordering. Every transport goes through it. | `memory.py` |
| `ingestion` | The **write path**: parse → chunk → extract entities and relations → write the graph. | `ingestion/stm/service.py`, `ingestion/stm/ingest.py` |
| `retrieval` | The **read path**: a signal-blind spine that fans collectors out concurrently, then fuses. | `retrieval/retriever.py` |
| `channels` | Optional memory signals with symmetric `ingest()` / `collect()` hooks, selected by `default_channels`. | `channels/base.py`, `channels/registry.py` |
| `ontology` | Loading RDF/OWL terms and matching them by **definition embedding**. | `ontology/loader.py`, `ontology/catalog.py` |
| `ranking` | RRF fusion constants and scoring. | `ranking/rrf.py` |
| `storage` | The single golden-layer graph store per namespace, plus embeddings. | `storage/arcadedb/graph_store.py`, `storage/embedder.py` |
| `models` | Layer-neutral vocabulary — `Hit`, `Message`, `MemoryScope`, `IngestResult`. | `models/` |
| `settings` | `GraphKnowsSettings`; mode presets expanded into capability knobs. | `settings.py` |
| `llm` | Provider-neutral LM construction. | `llm.py` |
| `exceptions` | `GraphKnowsError` and friends, including `MissingExtraError`. | `exceptions.py` |
| `server` | The MCP stdio server and its tool inventory. | `server/mcp/` |
| `integrations` | Framework adapters (LangGraph) and the stdlib-only MCP client SDK. | `integrations/` |

## The three paths

### Ingest — `Memory.add` / `ingest_memory`

Text or a message list is parsed and chunked, then per chunk: embed → match the
embedding against ontology term definitions for label hints → extract entities
and relations → normalise dates → let each active channel tag the chunk →
write the `CHUNK` spine, `ENTITY` + `MENTIONS`, relation edges with chunk-level
evidence, and the temporal/frame/ontology/WordNet layers.

Everything lands at `state='raw'`.

### Flush — `Memory.flush` / `flush_memory`

Consolidates one session **in place**; nothing is copied between databases.
Entity resolution merges duplicates, the analytical view is built once and
shared by PageRank / graph embeddings / communities, then the session flips
`raw → consolidated`. An analytics failure is recorded, never fatal — the state
flip still happens.

### Retrieval — `Memory.search` / `recall_memory`

The retriever builds a query context (embedding, tokens, the lifecycle `state`
implied by the scope), fans out four always-on collectors — entity traversal,
dense ANN, BM25, temporal — plus whichever channels are active, fuses the
candidates with RRF, shapes and cuts to `top_k`.

GraphKnows returns **ranked evidence, not an answer**. Generating the final
response is the caller's job.

Each path is traced statement by statement, with diagrams, in
[architecture.md](architecture.md).

## Entry points

**Library** — `from graphknows import Memory`. An async context manager; the
single in-process entry point.

**Console scripts**

| Command | Does |
| --- | --- |
| `graphknows-mcp` | Runs the MCP stdio server. |

**MCP tools** — ten, each taking an optional `namespace`:

| Group | Tools |
| --- | --- |
| Write | `memory_ingest`, `memory_flush`, `corpus_ingest` |
| Read | `memory_query`, `ltm_entity`, `ltm_entities` |
| Admin | `memory_stats`, `memory_doctor`, `memory_purge`, `memory_drop_namespace` |

There is **no HTTP API**. `graphknows/server/` contains only `mcp/`; MCP stdio
and the in-process facade are the two ways in.

## Storage shape in one paragraph

One physical ArcadeDB database per namespace — `mem` for the default namespace,
`mem_<ns>` otherwise. Short-term versus long-term is a `state` property in
`{raw, consolidated}`, not a separate database, and embeddings live directly on
the vertices behind `LSM_VECTOR` indexes rather than in a side-car vector store.
See [multi-tenancy.md](multi-tenancy.md) for the isolation guarantees.

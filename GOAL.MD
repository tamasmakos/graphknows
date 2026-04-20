## What We Are Building

**civicsgraph** is a political intelligence platform built on a knowledge graph stored in Neo4j. It helps politicians, researchers, and their staff navigate official political texts — debates, votes, press releases, legislation — with clarity, neutrality, and source citations.

The product has three layers:

1. **The Knowledge Graph** — a structured memory of modern politics: who said what, who voted how, what happened, and when.
2. **The GraphRAG Service** — a retrieval API that queries the graph with hybrid retrieval (Neo4j graph topology + pgvector embeddings + keyword search), fused with RRF, with full provenance back to source documents.
3. **The Chatbot Frontend** — a politically neutral, source-citing conversational interface. Not a search engine — a trusted analyst.

The long-term vision: become the institutional memory of modern democracy.

***

## Primary Goal (Right Now)

**Audit and solidify [`graphknows`](https://github.com/tamasmakos/graphknows), then build the React chatbot frontend against it.**

The immediate sequence:
1. Audit `graphknows`: identify gaps in GraphRAG quality, API design, and testability.
2. Deprecate `dashboard/`. Build the React chatbot from scratch in `apps/web` using Kibo UI components.
3. Bring in the first real dataset: **US Congress** (congress.gov API / GovTrack). EU Parliament is deferred.
4. Expand schema and ingestion iteratively as data quality improves.

### MVP scope

**In scope**
- One production-quality GraphRAG backend (Neo4j + pgvector).
- First jurisdiction: **US Congress only**.
- One locally runnable React chatbot that answers political questions with source citations.

**Not in scope for MVP**
- Auth, user accounts, subscriptions, billing.
- Multi-country ingestion.
- Real-time monitoring, predictive analytics, or persuasive outputs.

***

## Technology Stack

| Layer | Technology |
|---|---|
| Graph database | **Neo4j** |
| Vector database | **pgvector** |
| Graph query | **Cypher** |
| Backend services | **Python** (FastAPI) |
| Frontend | **React** (TypeScript) in `apps/web` |
| LLM | **Kimi K2** via OpenRouter (`openroutercustom/moonshotai/kimi-k2.5`) |
| UI components | **Kibo UI** — use CLI (`npx kibo-ui add <component>`) pulling from https://www.kibo-ui.com/components/ |
| Graph retrieval | **GraphRAG** (graph topology + pgvector + keyword, fused with RRF) |

***

## Architecture

### Service boundaries

- **`services/graphgen`** — ingestion only: parsing, entity extraction, normalization, embeddings, idempotent graph writes.
- **`services/graphrag`** — query only: retrieval orchestration, RRF fusion, context assembly, answer generation, citation packaging.
- **`apps/web`** — the sole frontend. `dashboard/` is deprecated and will be removed.
- **`services/product-api`** — auth, subscriptions, billing. Add only after graph + retrieval is solid. Do not add to MVP.

### Shared packages

- **`packages/schemas`** — shared request/response contracts, provenance, citation, stable IDs.
- **`packages/retrieval`** — retrieval pipelines, RRF, deduplication, context assembly.
- **`packages/db-adapters`** — Neo4j and pgvector adapters (no vendor assumptions in domain layer).
- **`packages/evals`** — benchmark questions, golden answers, retrieval regression checks.

### Repo structure

```text
graphknows/
  services/
    graphgen/        # ingestion, parsing, graph build, embeddings
    graphrag/        # query API, retrieval, RRF, answer assembly
  apps/
    web/             # React chatbot frontend (replaces dashboard/)
  packages/
    schemas/         # shared contracts and provenance models
    retrieval/       # RRF, hybrid retrieval, context assembly
    db-adapters/     # Neo4j and pgvector adapters
    evals/           # benchmark datasets and regression harness
  infra/
    docker/          # local stack definitions
```

### Technical boundaries (hard rules)

- Frontend never calls databases directly.
- `graphgen` writes to the graph. `graphrag` reads from the graph. Never mix.
- No service encodes Neo4j- or pgvector-specific assumptions in its domain layer — all storage goes through `packages/db-adapters`.
- Prompt templates, retrieval settings, and RRF parameters live in config or package modules — never in route files.
- Every user-visible answer must carry: source URL, document ID, retrieval path, and supporting chunk.
- Upserts must be idempotent. Use `MERGE` in Cypher, not `CREATE`.
- API payloads and citation structures are treated as stable contracts. Version them if they change.
- RRF is mandatory, not optional. Evaluate GraphRAG quality against a vector-only baseline before marking any retrieval feature done.

***

## Data Strategy

### Ingestion priority

1. **Verbatim debate transcripts** — attributed, timestamped, linked to session/date/institution.
2. **Roll-call votes** — every vote, every legislator, every bill/amendment, the outcome.
3. **Press releases and official statements**.
4. **Legislative texts** — bills, amendments (high volume, secondary priority).

### Data source: US Congress

| Source | What it provides |
|---|---|
| congress.gov API | Bills, votes, members |
| GovTrack | Roll-call votes, transcripts |

EU Parliament, Asian legislatures: deferred until US Congress graph is stable.

### Graph schema (start minimal)

Node types: `Person`, `Party`, `Institution`, `Speech`, `Vote`, `Bill`, `Event`, `Document`.  
Add new types only when a concrete use case demands them.

Every node must carry: source URL, document ID, scrape timestamp, confidence level.

### Retrieval design

- Chunk transcripts and documents into citation-sized passages.
- Embed both chunks and higher-level records for semantic retrieval.
- Use Cypher filters whenever a query implies time, institution, person, or bill.
- Retrieve candidates via three paths in parallel: graph traversal, keyword/metadata, pgvector semantic search.
- Fuse with Reciprocal Rank Fusion. Deduplicate and preserve provenance before passing to generation.

***
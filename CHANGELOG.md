# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and the project aims to
follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **Ontology injection in both memory modes.** `GRAPHKNOWS_ONTOLOGY` takes an
  RDF/OWL file *or a folder* and steers entity and relation extraction toward
  its vocabulary in `llm_free` as well as `llm_assisted`. The chunk's ingest
  embedding selects candidate classes/properties from the terms' definitions,
  which are unioned into the mined label space. Extraction is enriched, never
  constrained: entities and relations that map to nothing are still written
  under their extracted labels. See `docs/ontology.md`.
- Ontologies are persisted per namespace as `ONTOLOGY` / `ONTOLOGY_CLASS` /
  `ONTOLOGY_PROPERTY` vertices with their definitions and definition
  embeddings, linked to the data by `INSTANCE_OF` (entity → class) and
  `MATCHES_CLASS` (chunk → class), plus `REL.ontology_property`.
- `OntologyChannel` — retrieval signal that vector-searches those class
  definitions and traverses to the chunks tagged with them, contributing to the
  RRF fusion. Registered only when an ontology is configured.
- **Topic layer, deterministically.** `GRAPHKNOWS_TOPICS=extractive` builds
  TOPIC nodes over the Louvain communities with no LLM — titles from PageRank,
  summaries led by the member chunk nearest the community centroid — so the
  layer is available under `llm_free`. `topics=llm` keeps prose summaries.
  Previously topic summarisation was dead in *both* modes.
- **Topics as a re-runnable stage** (`graphknows/topics/`). `Memory.build_topics()`,
  the `memory_topics` MCP tool and `python -m graphknows.topics` rebuild the
  topic layer at new settings over an unchanged graph — no re-ingest, and none
  of the flush's entity resolution / PageRank / consolidation. `--sweep` compares
  settings and prints the metric table. `flush_memory` calls the same stage, so a
  rebuild is never a different code path. See `docs/topics.md`.
- `TopicChannel` now routes a matched topic to its member chunks via
  `HAS_TOPIC` instead of only injecting the topic summary, so the topic layer
  contributes real evidence to retrieval. Scores accumulate across matched
  topics, are discounted by topic coherence, and carry the same saturation
  guard as the frame/entity/ontology channels.
- Topic outputs now include TF-IDF `keywords` (the terms distinguishing a topic
  from its neighbours), member entities ranked by PageRank, per-topic
  `coherence`, `lead_chunk_id`, and a real Python-computed `newman_modularity`.
  node2vec's `walk_length` / `walks` / `dimensions` are exposed, having been
  hardcoded.
- **Per-call retrieval strategy.** `memory_query` / `recall_memory` / `search`
  take `strategy` (`det` | `agentic`), so one server can answer either way
  without a restart, and agentic retrieval can run over an `llm_free`-ingested
  graph. `memory_query` also returns `search_type`.

### Changed
- **`GRAPHKNOWS_MODE` is now a preset, not a switch.** It expands into
  independent knobs — `GRAPHKNOWS_TOPICS`, `GRAPHKNOWS_DSPY_RELATIONS`,
  `GRAPHKNOWS_RETRIEVAL` — each overridable on its own, so combinations like
  `llm_free` + agentic retrieval are expressible. Read the resolved values via
  `settings.topic_mode` / `dspy_relations` / `retrieval_strategy`. Existing
  `GRAPHKNOWS_MODE` values keep their previous behaviour.
- The production API-key guard follows the resolved capability rather than the
  mode label: agentic retrieval requires a key even under `llm_free`, and
  `llm_assisted` with every LLM capability off does not.
- Ontology code moved from `graphknows/ingestion/extraction/relations/llm_assisted/ontology/`
  to **`graphknows/ontology/`**, a shared leaf now used by both ingestion and
  retrieval.
- `GRAPHKNOWS_ONTOLOGY_FILE` → `GRAPHKNOWS_ONTOLOGY` (old name still honoured as
  a deprecated alias, single file only).

### Fixed
- `search_topics` had no session filter, so in a multi-session namespace the
  topic channel could match topics belonging to other sessions.
- A topic written without an embedding is stored but unreachable, because
  `TopicChannel` vector-searches `TOPIC.embedding`. The stage now always derives
  a centroid from the member chunks, so the community-clustered path (which has
  no entity vectors) no longer produces an inert layer.
- Ontology term embeddings could be paired with the wrong term. Term order came
  from an rdflib subject *set* (process-randomised) while the disk cache
  validated only the row *count*, so a cached matrix loaded against a
  differently-ordered term list returned cosine 1.0 for the wrong label. Terms
  are now sorted and the cache carries its URI list.
- `corpus_ingest` was broken for every caller: it constructed `STMService` with
  keyword arguments removed in 2.0.0, so the MCP tool returned an error and
  processed zero documents. It now runs the golden-layer path (per-document
  ingest, then one consolidation).
- Entity names containing an apostrophe were unfindable. `sanitize()` stripped
  quotes while `ENTITY.name_norm` only casefolds, so `O'Brien` was stored as
  `o'brien` and looked up as `'obrien'`. Quotes and backslashes are now escaped.
- `Memory.flush_memory` ran its pipeline under one `try`, so an analytics failure
  skipped `consolidate_session` — the raw → consolidated flip that makes a flush
  meaningful. Each step now fails independently and names itself in `errors`.
- `AgenticRetriever` kept its ReAct module and hit buffer on the instance, so
  concurrent recalls with different sessions overwrote each other's evidence.
- A partially failed `delete_session` reported success, leaving orphaned nodes;
  it now raises `StoreError` (a missing vertex type is still a no-op).
- Documented import paths that did not exist: `graphknows.adapters.langgraph`
  (→ `graphknows.integrations.langgraph`), `graphknows.client`
  (→ `graphknows.integrations.client`), `graphknows.ports`
  (→ `graphknows.storage.base`), and install extras that were folded into the
  base package in 2.0.0.
- `GRAPHKNOWS_CHUNK_SIZE` / `GRAPHKNOWS_CHUNK_OVERLAP` were documented and
  settable but never read; they now reach the chunker (defaults realigned to the
  values it was already using, so behaviour is unchanged).

### Removed
- The STM→LTM promotion pipeline (`graphknows.ingestion.promotion`,
  `PromotionStep`, `default_promotion_steps`) and the corpus-ingest workflow —
  both were superseded by in-place consolidation in 2.0.0 and had no remaining
  caller.
- `ArcadeDBVectorStore` — embeddings live on the graph vertices; the separate
  vector store was unreachable.
- Unwired ranking levers `pagerank_boost` / `idf_weight`, and the unused
  `graphknows.llm` helpers `get_model_name` / `get_temperature`.

## [2.0.0] — 2026-07-13

Capability-first architectural refactor. The package is recut along the memory
pipeline (parse → extract → STM → promote → LTM; collect → rank → context), the
two modes (`llm_free` / `llm_assisted`) are separated inside each extraction
concern, and the install collapses to a batteries-included base plus a single
extra. **Breaking**: internal import paths changed and deprecated 1.x aliases
were removed.

### Added
- **Namespace-first SDK** on `Memory`: `add` / `search` / `flush` take
  `user_id` / `agent_id` / `run_id` scope kwargs (mem0-style). The `*_memory`
  verbs remain as the MCP/web transport handlers.
- **`graphknows.channels`** — the plugin axis. One `Channel` class implements the
  symmetric contract (`ingest`/`promote` write hooks + `collect` read hook);
  `register(name, cls)` adds a signal to both the ingestion and retrieval paths
  at once.
- **Promotion step pipeline** — STM→LTM flush is now an ordered list of
  `PromotionStep` objects (`graphknows.ingestion.promotion`); insert a stage with
  `default_promotion_steps()` + a list edit. `NullRelationExtractor` removes the
  llm_free bool-branching.
- **`graphknows.integrations.langchain`** — a LangChain retriever adapter
  alongside the existing LangGraph one; both live under `graphknows.integrations`.

### Changed
- **Modules** recut into `settings`, `models`, `storage`, `channels`,
  `ingestion` (write path, with `ingestion/extraction/relations/{llm_free,llm_assisted}`),
  `retrieval` (read path), `memory` (facade/composition root), `integrations`,
  and `server` (`mcp` + `web`). Mode selection lives in three factories
  (`build_relation_extractor`, `build_retriever`, `default_channels`); the
  `MemoryProfile`/`get_profile` seam is gone.
- **Install**: `pip install graphknows` is now batteries-included (extraction,
  embeddings, analytics, parsers, MCP + web server) and works out of the box in
  `llm_free` mode. The only extra is `graphknows[assisted]` (dspy + rdflib).

### Removed
- Deprecated `GraphKnowsRuntime` alias, the `graphknows.ports` package
  (protocols now in `graphknows.storage.base` + `…relations.base`), and the
  `IVectorDb` alias.
- The `local-embeddings`, `extraction`, `analytics`, `parsers`, `server`, `web`,
  `langgraph`, and `all` extras (folded into the base install).

### Migration (old → new import paths)
- `graphknows.runtime` / `graphknows.GraphKnowsRuntime` → `graphknows.Memory`
- `graphknows.config` → `graphknows.settings`
- `graphknows.ports` → `graphknows.storage.base` (+ `…extraction.relations.base`)
- `graphknows.memory.query.*` → `graphknows.retrieval.*`
- `graphknows.memory.stores.*` → `graphknows.storage.*`
- `graphknows.memory.stm.*` / `flush` / `workflows` → `graphknows.ingestion.*`
- `graphknows.extraction.*` → `graphknows.ingestion.extraction.*` / `graphknows.channels.*`
- `graphknows.memory.channels.*` → `graphknows.channels.*`
- `graphknows.mcp` / `graphknows.web` → `graphknows.server.mcp` / `graphknows.server.web`
- `graphknows.client` → `graphknows.integrations.client`
- `graphknows.adapters.langgraph` → `graphknows.integrations.langgraph`

## [1.0.0] — 2026-07-07

First stable release: the internal service becomes an importable,
pip-installable, framework-agnostic library with a first-class server story.

### Added
- **`Memory`** — the single transport-neutral facade (async context manager).
  `GraphKnowsRuntime` is kept as a deprecated alias for the 1.x line.
- **`graphknows.client`** — a stdlib-only out-of-process SDK: `RemoteMemory`
  (mirrors the `Memory` verbs over MCP) and `GraphKnowsMCPClient`. No ML deps.
- **`graphknows.ports`** — public store protocols (`ISTMStore`, `IVectorStore`,
  `ILTMStore`) plus `IEmbedder`/`IRelationExtractor`, so a backend is swappable
  in principle (ArcadeDB is the shipped implementation).
- **`graphknows.adapters.langgraph`** — a reference framework adapter
  (`GraphKnowsStore`, `recall`/`remember` node helpers) + a generic integration
  recipe (`docs/integrations.md`).
- **Extras-based install**: a small core (no ML), with `local-embeddings`,
  `extraction`, `analytics`, `parsers`, `server`, `web`, `assisted`,
  `langgraph`, and `all`. Missing extras raise `MissingExtraError` with an
  install hint. Ships a `py.typed` marker.
- Public API surface from the package root: `Memory`, `GraphKnowsSettings`,
  `MemoryMode`, `IngestResult`, `__version__`, and the exception hierarchy
  (`GraphKnowsError`, `ConfigurationError`, `StoreError`, `IngestError`,
  `RetrievalError`, `ExtractionError`, `MissingExtraError`). Pinned by a
  public-surface snapshot test.
- Local-by-default embeddings (sentence-transformers), so `llm_free` mode runs
  with **zero API keys**. A remote OpenAI-compatible endpoint is opt-in via
  `GRAPHKNOWS_EMBED_API_BASE`.
- Provider-neutral LLM configuration: `GRAPHKNOWS_LLM_MODEL` is a litellm model
  string (any provider), with optional `GRAPHKNOWS_LLM_API_BASE`.
- `assisted` extra for `llm_assisted` mode: `pip install "graphknows[assisted]"`.
- Documentation: `README.md`, `docs/` (architecture, modes, ontology,
  configuration), `CHANGELOG.md`.

### Changed
- The core install is now small: heavy ML/NLP and transport dependencies moved
  behind extras (`import graphknows` and `graphknows.client` pull no ML deps).
- Ingestion is async-first: `Memory.ingest_memory` and the STM service run store
  I/O on the caller's event loop; the background-thread bridge is gone from the
  ingest path.
- Stores expose public `connect()`/`close()` (no more private client access);
  internal store handles renamed to semantic names (`stm`/`vec`/`ltm`).
- `GraphKnowsSettings` validates production secrets at construction (fail fast).
- **BREAKING:** all environment variables are now prefixed `GRAPHKNOWS_`
  (e.g. `ARCADEDB_URL` → `GRAPHKNOWS_ARCADEDB_URL`, `LLM_MODEL` →
  `GRAPHKNOWS_LLM_MODEL`, `MODE` → `GRAPHKNOWS_MODE`). Old names are ignored.
- **BREAKING:** default embedder changed to a local model (384-dim). Vector
  indexes are now created at the configured embedder's dimension. Existing
  ArcadeDB databases keep their old 1536-dim index, so **drop and recreate the
  `stm`/`ltm` databases** (or point at a fresh ArcadeDB) and re-ingest after
  upgrading — `CREATE INDEX IF NOT EXISTS` will not rebuild an existing index.
- **BREAKING:** removed `openrouter_api_key` / `llm_provider` settings; use
  `GRAPHKNOWS_LLM_API_KEY` and a litellm model prefix instead.
- Settings now use standard precedence (env beats `.env`); removed the
  repo-`.env` parent-directory walk.
- Relicensed from BSL-1.1 to **Apache-2.0**.
- Default `GRAPHKNOWS_ARCADEDB_URL` is now `http://localhost:2480`.

### Removed
- No hardcoded OpenRouter endpoint anywhere in the library.
- 16 unused dependencies (fastapi, langchain-core/openai/experimental, gliner v1,
  click, tiktoken, pymupdf, pdfminer.six, langsmith, python-multipart,
  scikit-learn, scipy, pyjwt, idna, urllib3).
- Repo cruft: Caddyfile, orphaned config modules; demo notebooks moved to
  `examples/`.

### Known limitations
- ArcadeDB database names (`stm` / `ltm`) are fixed; running multiple instances
  against one ArcadeDB server is not yet supported.
- Runtime results are plain dicts (typed result models are planned).

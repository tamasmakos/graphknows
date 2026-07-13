# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and the project aims to
follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and the project aims to
follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0rc1] — unreleased

First public release candidate: the internal service becomes an importable,
pip-installable library with a first-class server story.

### Added
- Public API surface from the package root: `GraphKnowsRuntime` (now an async
  context manager), `GraphKnowsSettings`, `MemoryMode`, `__version__`, and the
  exception hierarchy (`GraphKnowsError`, `ConfigurationError`,
  `StoreConnectionError`, `ExtractionError`, `MissingExtraError`).
- Local-by-default embeddings (sentence-transformers), so `llm_free` mode runs
  with **zero API keys**. A remote OpenAI-compatible endpoint is opt-in via
  `GRAPHKNOWS_EMBED_API_BASE`.
- Provider-neutral LLM configuration: `GRAPHKNOWS_LLM_MODEL` is a litellm model
  string (any provider), with optional `GRAPHKNOWS_LLM_API_BASE`.
- `assisted` extra for `llm_assisted` mode: `pip install "graphknows[assisted]"`.
- Documentation: `README.md`, `docs/` (architecture, modes, ontology,
  configuration), `CHANGELOG.md`.

### Changed
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

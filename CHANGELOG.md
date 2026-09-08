# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and the project aims to
follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed
- **`llm_assisted` is now a replacement decoder, not a sidecar.** In that mode the
  LLM is the sole producer of a chunk's entities, relations and frame roles; no
  local extraction model is loaded. DSPy moved to a core dependency and the
  `assisted` extra is **removed** — installing `graphknows[assisted]` now warns
  about an unknown extra. It also pulled in `rdflib` and `networkx` as a side
  effect, so anyone who relied on that for RDF should install
  `graphknows[ontology]`, which is unchanged.

### Added
- **Tag-triggered release workflow** (`.github/workflows/release.yml`). Pushing
  a `vX.Y.Z` tag reuses `ci.yml` by reference (Sonar skipped, since a tag has
  no PR context), then refuses to publish unless the tag matches the
  `__version__` literal in `graphknows/_version.py`, and attaches the wheel CI
  itself built, plus a deployment bundle, as assets on this private
  repository's GitHub Release for the tag — nowhere public, no PyPI publish,
  no Trusted Publisher OIDC, no stored API token; see `docs/versioning.md`.

### Removed

Dead-code sweep (Tiers 1–2 of a whole-repo over-engineering audit): ~6,200 lines
deleted, no behaviour change intended. Everything below had **zero production
callers** at the time of removal, verified by grep; where a claim turned out to be
wrong the code was kept (`IN_TOPIC` is written by `topics/persist.py` and read by
`Memory.ltm_topics`, so it stays).

**BREAKING — documented public surface** (see `docs/versioning.md`). These went
without a deprecation window; the package version is unchanged at `2.0.0`, so
that decision is still open:

- `graphknows.integrations.client.RemoteMemory` — a wrapper mirroring `Memory`'s
  verbs over MCP. Zero importers outside its own test; `GraphKnowsMCPClient` is
  the transport and now the whole surface. Call tools by name
  (`memory_ingest`, `memory_query`, …) via `client.call_tool`.
- `graphknows.retrieval.BaseGraphRetriever` — an ABC with one implementation,
  whose docstring still promised the long-deleted `AgenticRetriever`. Use
  `DETRetriever` (or `build_retriever`) directly.
- `graphknows.ingestion.BaseParser` — an ABC with one implementation, and the
  inheritance ran backwards: all the chunking logic lived in the "abstract" base
  while `PlainTextParser` contributed only the heading split. The two modules are
  merged into `graphknows/ingestion/parsers/text.py`; `ParsedFile`, `TextSegment`,
  `search_surface` and `strip_content_heading` moved there with it.
- `graphknows.storage.base` — the `IGraphStore` / `IEmbedder` Protocols. One
  implementation existed (`GraphStore`) and `IEmbedder` had none at all. Custom
  backends should target `GraphStore` directly.
- `graphknows.integrations.langchain` (`GraphKnowsRetriever`) and the
  `langchain` extra. Zero importers; it raised a bare `ImportError` rather than
  `MissingExtraError`, so the extra it named guarded nothing.
- `IngestError`, `RetrievalError`, `ExtractionError` from `graphknows.__all__` —
  never raised, never caught. `StoreError` / `ConfigurationError` /
  `BrokenInstallError` / `MissingExtraError` are unaffected.
- MCP tool contract: the `memory_recall` alias (a pure re-forward to
  `memory_query`) and the `ltm_query` tool. The `strategy` argument is gone from
  `memory_query` — it accepted `"agentic"` and then raised `ValueError`
  downstream, since that retriever no longer exists.
- `Memory`: `connect()`, the `memory_class` parameter (accepted and silently
  dropped — it never reached `STMService`), and `purge_memory(scope=...)`, which
  the docstring already admitted "no longer selects a physical store". It now
  returns a single `deleted` count.

**Configuration.** `GRAPHKNOWS_RETRIEVAL` (a one-member enum since the agentic
retriever was deleted), `GRAPHKNOWS_CHUNK_SIZE` and `GRAPHKNOWS_CHUNK_OVERLAP`
(inert — chunking reads module constants, not settings), `GRAPHKNOWS_PORT` (no
readers since the web server was removed), and `GRAPHKNOWS_SESSION_TTL_MINUTES`
(no readers, though it was documented as live).

**Internal.** The conversational taxonomy subsystem and the GLiNER2 label-space
helpers; the `code`/`docx` parsers and their registry, with the `python-docx` and
`pyyaml` dependencies (`PlainTextParser` is the only parser any caller reached);
the `RetrievalContext` XML/dict wire format, whose `graph_matches` and
`semantic_matches` sections were permanently `count="0"`; the per-query retrieval
`trace`, which had no readers; `filter_by_validity_window`, which nothing could
wire because no ingest path writes `REL.valid_at`; the `IRelationExtractor`
Protocol, `NullRelationExtractor` and the DSPy adapter passthrough
(`build_relation_extractor` now returns `DSPyExtractor | None`); a second
NumPy PageRank kept as a fallback (it now soft-fails to 0 with a log line, like
its two sibling algorithms); the channel registry's `register()`, profile filter
and four delegating collector subclasses; and write-only graph properties
(`REL.map_score`, `CHUNK.atom_type`, `CHUNK.entity_names`, `FILE.name`,
`FILE.media_type`, `SESSION.label`, the ONTOLOGY_PROPERTY / DECLARES /
SUBCLASS_OF layer, and an `ENTITY.embedding` vector index built over a property
nothing ever wrote).

Also removed, from a second audit pass over the same tree:

- **`TOPIC.embedding`, its `LSM_VECTOR` index, and the `CHUNK-[:HAS_TOPIC]->TOPIC`
  edge.** All three existed only for `TopicChannel`, which went with the rest of
  the topic read path — leaving a vector index and an edge that were written on
  every flush and read by nothing. `ENTITY-[:IN_TOPIC]->TOPIC` stays: it is the
  clustering's actual output and `Memory.ltm_topics` reads it. The centroid is
  still computed (it is what `coherence` is measured against), just not stored.
- **The `graph_embedding` topic source**, with `TopicConfig.source`,
  `distance`, the four `n2v_*` / `recompute_embeddings` fields, the matching
  `--source`/`--distance`/`--n2v-*` CLI flags, `--sweep-param`, and the same
  arguments on the `memory_topics` MCP tool. It clustered node2vec vectors, but
  `compute_graph_embeddings` skips entities with fewer than 3 relation edges, so
  on a sparse conversational graph it reached 33 of 1049 entities (LoCoMo
  conv-26) — topics describing 3% of the corpus. The MCP tool also defaulted it
  to `graph_embedding` while `TopicConfig` defaulted to `community`, so the two
  entry points disagreed. `min_size` is now the only knob, which is honest:
  ArcadeDB's Louvain ignores its parameter map. node2vec itself still runs at
  flush — nothing reads it, which is a wiring gap, not dead code.
- `graphknows.retrieval.query` — a package whose entire content was a docstring
  for query decomposition that does not exist.
- `graphknows.ingestion.models` and `graphknows.ingestion.normalization` — a
  package re-exporting one 6-line function; `standardize_label` now lives in its
  only caller.
- The `python-json-logger` dependency, which nothing imported.

### Fixed
- **A model's declared `suffix` was never applied, inverting ontology class
  matching.** sentence-transformers applies `prompts`, which are *prefixes
  only* — 5.6.1 has no suffix support whatsoever. `zembed-1` declares
  `suffix: "<|im_end|>\n"` in `config_sentence_transformers.json` **and** pools
  the last token (`pooling_mode_lasttoken`), so every embedding was pooled at an
  arbitrary content token instead of the sentinel the model was trained on. The
  geometry was wrong, not merely noisy: for "He finally bought that vintage
  motorcycle he had been saving up for", the ontology class `Possession` scored
  0.105 and ranked **11th of 11**, behind `Emotion` at 0.425; `Place` ranked 7/11
  for a residence sentence. Applying the declared suffix moves them to 2/11 and
  1/11. `embedder._model_suffix` now reads it from the model config and
  `_embed_local` appends it. A no-op for models that declare none — `bge-small`
  output is bit-identical — so only opt-in models like zembed-1 are affected.
  - **Requires re-embedding if you use a suffix-declaring model.** Vectors
    already persisted in ArcadeDB, and any `~/.cache/graphknows` index, were
    computed the old way. Same constraint as changing `GRAPHKNOWS_EMBED_MODEL`.
- **Embedding cache keys ignored *how* text is encoded.** The ontology-catalog
  and frame-index disk caches keyed on model + dimension + content, so the fix
  above did not invalidate them: a matrix built under the old encoding was
  silently reused against vectors from the new one, producing plausible-looking
  scores that were meaningless (`Possession` 0.0999 against a stale matrix vs
  0.6829 against a fresh one). Both keys now include an encoding component.
- **`gliner` was imported but never declared.** The default `llm_free`
  extraction path loads `knowledgator/gliner-relex-large-v1.0` through
  `from gliner import GLiNER` (`entities/gliner_model.py`), but `pyproject.toml`
  declared only `gliner2`. A clean install — or a freshly built image — raised
  `ModuleNotFoundError` on first ingest. It went unnoticed because
  `docker-compose.yaml` bind-mounts the working tree over the image and puts it
  first on `PYTHONPATH`, so containers ran live source against months-old
  site-packages that happened to still carry `gliner` from an earlier layout.
- **The BM25 channel was never running BM25, and was broken when switched on.**
  `rank-bm25` was imported by `retrieval/retriever.py` and declared nowhere, so
  it was never installed, `_HAS_BM25` was always False, and the channel silently
  served a fallback tf-scorer. Declaring it exposed a latent bug in the branch
  that had therefore never executed: `_bm25_rank` filtered results on
  `score > 0.0`, which is equivalent to "contains a query term" only for the
  fallback's always-positive IDF. BM25Okapi's IDF goes negative once a term
  appears in more than ~half the corpus — routine at candidate-pool sizes — so
  genuine matches were discarded and the channel could return nothing at all.
  Now filtered on term presence. **Any past measurement attributed to the BM25
  arm of the fused pool was measuring the fallback.**
- **`nltk` was imported by default-mode code but declared only in the `eval`
  dependency-group** (`relations/svo.py`, `channels/_framenet.py`). Those call
  sites catch `LookupError` (a missing *corpus*), which does not catch the
  `ModuleNotFoundError` from a missing *package*, so `pip install graphknows`
  produced a hard crash. Only the Docker image worked, because it installed the
  eval group. `nltk` is now a core dependency.
- **`torch`, `transformers` and `huggingface-hub` are now declared.** All three
  are imported directly by `embedder.py`, `rerank.py`, `gliner_model.py` and
  `_gliner2_verifier.py`; relying on the transitive edge through
  `sentence-transformers` meant any of its releases could break us silently.
- **`.[all]` is not a declared extra.** Both Dockerfiles installed it; uv only
  *warns* on an unknown extra, so the shipped image contained no extras at all
  and `GRAPHKNOWS_MODE=llm_assisted` raised `ModuleNotFoundError` for `dspy`.
  Now installs `.[assisted]`.
- **The embed-model default disagreed with itself.** `settings.py` defaulted to
  `zeroentropy/zembed-1-embedding` (2560-dim) while `docker-compose.yaml` and
  `.env.example` defaulted to `BAAI/bge-small-en-v1.5` (384-dim). ArcadeDB
  vector indexes are dimension-typed, so a library user and a container user
  silently built incompatible indexes. `BAAI/bge-small-en-v1.5` is now the
  single default everywhere, and `EMBED_DIM` was corrected from 2560 to 384.
- **Six of seven `MissingExtraError` call sites named extras that do not
  exist** (`parsers`, `server`, `web`, `local-embeddings`, and previously
  `langgraph`/`langchain`), telling users to run a `pip install` that installs
  nothing. Those guarding *core* dependencies now raise the new
  `BrokenInstallError`; `langgraph` and `langchain` became real extras.

### Added
- **All models are baked into the Docker image at build time.**
  `scripts/bake_models.py` downloads the relex extractor, embedder, reranker,
  relation verifier and its DeBERTa encoder, the spaCy pipeline, and the NLTK
  `wordnet`/`omw-1.4`/`framenet_v17` corpora into `/opt/models`. Model ids are
  resolved from the code that owns them, so the list cannot drift from the app.
  The image also sets `HF_HUB_OFFLINE=1`: without it the hub client makes a
  revision check on every model load, which is per-call latency on a cold
  worker, a hard runtime dependency on huggingface.co, and an outright failure
  when air-gapped (it raises rather than falling back to cache). Set
  `HF_HUB_OFFLINE=0` to run a model that is not baked.
  - Previously nothing was prefetched, and `channels/_framenet.py` called
    `nltk.download("framenet_v17")` from inside the request path. When a corpus
    was absent the call sites caught `LookupError` and returned a degraded
    result with no warning.
  - `nltk.download()` needs an explicit `download_dir`: it does not honour
    `NLTK_DATA`, and as root (as in a build) it writes to a system directory,
    reports success, and leaves nothing where the runtime looks.
- **`scripts/bake_models.py --check`** runs as the container healthcheck. It
  asserts every model is present *offline*, so it cannot pass by downloading,
  and verifies corpora through the same corpus-reader calls the runtime makes
  rather than `nltk.data.find` (which fails on the still-zipped `wordnet` and
  `omw-1.4` even though the readers load them fine).
- **`docker-compose.prod.yaml`** — a deployment stack where the image is the
  artifact: no repo bind-mount, no `PYTHONPATH=/app`, a required (not
  defaulted-to-`changeme`) ArcadeDB password, and restart policies on every
  service.
- **`tests/test_packaging.py`** — asserts that every third-party module the
  package imports resolves to a declared distribution, that default-path
  dependencies are core rather than extras, and that every `MissingExtraError`
  names a real extra. The cheap seam for the whole class of bug above; it is
  what found `rank-bm25`, `torch`, `transformers` and `huggingface-hub`.
- **`scripts/docker-smoke.sh`** — runs the shipped image with `--network none`
  and asserts every dependency imports, every model loads from cache, and warm
  embedding stays under 1s/call. It deliberately does not inject offline env
  vars: the image must be self-sufficient by its own configuration.

### Changed
- **The Docker image ships CPU torch by default** (`--build-arg TORCH_INDEX`
  selects a CUDA index for a GPU image). The PyPI wheel bundles ~2.5GB of CUDA
  kernels a CPU deployment never executes.
- **The runtime image no longer contains the developer toolchain.** It was
  installing `--group dev --group eval` plus `jupyterlab`, shipping `pytest`,
  `ruff`, `mypy`, `bandit`, `tach`, `pandas`, `datasets` and JupyterLab to
  users, along with `uv` and `git`. Those now live only in the `dev` stage.
- **`Dockerfile.web` was folded into a `web` stage in `Dockerfile`.** The two
  files duplicated the install and had already drifted: the web image omitted
  the dependency groups that were smuggling `nltk` in, so it lacked packages the
  mcp image had from the same commit.

### Removed
- **The manifest/registry/router ontology stack.** `OntologyRouter`,
  `OntologyRegistry`, `OntologyProvider`, `ManifestOntologyProvider`,
  `OntologyLabelExtractor` / `extract_ontology_labels`, `OntologyManifest`,
  `OntologyLoader`, and `graphknows/ontology/assets/registry.json`, plus
  `RelationOntologyFilter`, `build_relation_filter` and
  `build_predicate_mapper`. None had a production caller: an ontology is
  addressed by path (`GRAPHKNOWS_ONTOLOGY`) and loaded via
  `load_ontology_terms` / `load_rdf_ontology_file`, which the registry lookup
  never served. The bundled ontologies under `assets/` are unchanged and are
  still referenced by path. `OntologyPredicateMapper` and `RdfOntologyManifest`
  are unaffected.
- As a side effect, `graphknows.ontology` no longer requires `rdflib` to
  *import*: the removed `rdf_parser` raised `ImportError` at module scope, which
  the package's `__init__` triggered eagerly, defeating the lazy imports the
  term loader uses everywhere else.

### Added
- **The container refuses to start when its dependencies are stale.**
  `scripts/preflight.py` runs from the image `ENTRYPOINT`, so it wraps every
  command — `graphknows-mcp`, `graphknows-web` and the dev shell alike. It fails
  the boot when a dependency declared in the mounted `pyproject.toml` is not
  installed, or when the mounted `uv.lock` differs from the one the venv was
  resolved from (its hash is baked to `/opt/venv/.lock-hash` at build time).
  `GRAPHKNOWS_SKIP_PREFLIGHT=1` bypasses it, which is needed to get a shell in an
  already-wedged container. In a deployment there is no bind-mount, so both
  checks no-op: the image is the truth by definition.
  - This is the guard for the root cause of everything below: `docker-compose.yaml`
    mounts the working tree over `/app` and puts it first on `PYTHONPATH`, so a
    container runs **live source against image dependencies**. Nothing detected
    the gap, and the shipped `mcp` image was seven weeks stale.
- **CI validates the artifact, not just the source.** A new `Image` stage builds
  the image and runs `scripts/docker-smoke.sh` against it. Every other gate does
  `uv sync` into a fresh environment, so all of them passed against a correct
  dependency set while the image itself was months out of date — no stage had
  ever built it. Gated on `main` and on changes to `Dockerfile`, `pyproject.toml`,
  `uv.lock` or `scripts/*`.
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
- **The CI ban on sentence-transformers.** The Lint stage rejected the import in
  production code, the dependency in `pyproject.toml` and the entry in `uv.lock`
  — a rule from when embeddings were remote-only. Local sentence-transformers
  embeddings are the documented design (the `llm_free` path runs with zero API
  keys) and `pyproject.toml` declares it as a core dependency, so the gate
  contradicted the package it was gating and had been failing for a long time.
  A permanently-red pipeline is part of why nothing caught the dependency drift.
- **The dangling `scripts/check_imports.py` CI step.** The script was added in
  `66619ab` and later deleted without updating the `Jenkinsfile`, so the Lint
  stage errored before reaching anything useful. `tests/test_packaging.py` now
  covers what it was for.
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

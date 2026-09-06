# Configuration

All configuration is read from `GRAPHKNOWS_*` environment variables, or from a
`.env` file in the current working directory. Precedence is standard: constructor
arguments > environment variables > `.env` file. Settings are defined in
[`graphknows/settings.py`](../graphknows/settings.py) (`GraphKnowsSettings`).

```python
from graphknows import GraphKnowsSettings, Memory

# From environment / .env:
async with Memory() as mem:
    ...

# Or explicitly:
settings = GraphKnowsSettings(mode="llm_assisted", llm_api_key="sk-...")
async with Memory(settings) as mem:
    ...
```

## Reference

| Variable | Default | Purpose |
|---|---|---|
| `GRAPHKNOWS_MODE` | `llm_free` | `llm_free` or `llm_assisted`. Drives both extraction and retrieval. |
| `GRAPHKNOWS_ENV` | `development` | Set to `production` to enforce secret checks at server startup. |
| **ArcadeDB** | | |
| `GRAPHKNOWS_ARCADEDB_URL` | `http://localhost:2480` | ArcadeDB HTTP base URL. |
| `GRAPHKNOWS_ARCADEDB_USER` | `root` | ArcadeDB user. |
| `GRAPHKNOWS_ARCADEDB_PASSWORD` | `changeme` | ArcadeDB password. Required in production. |
| `GRAPHKNOWS_NAMESPACE` | `""` | Tenant isolation boundary → one physical database `mem_<ns>` (`""` = the shared `mem`). The default when an MCP call omits `namespace`. See [multi-tenancy.md](multi-tenancy.md). |
| `GRAPHKNOWS_ENABLE_ADMIN_TOOLS` | `false` | Register the destructive MCP tools `memory_purge` and `memory_drop_namespace`. Off: they are not advertised at all. Read once at startup; the server names them on stderr when on. |
| **Embeddings** | | |
| `GRAPHKNOWS_EMBED_MODEL` | `BAAI/bge-small-en-v1.5` | Local sentence-transformers model, or a remote model id when `EMBED_API_BASE` is set. |
| `GRAPHKNOWS_EMBED_API_BASE` | _(empty)_ | Set to use a remote OpenAI-compatible embeddings endpoint instead of local. |
| `GRAPHKNOWS_EMBED_API_KEY` | _(empty)_ | API key for the remote embeddings endpoint. |
| **LLM** (only read by the LLM-backed capabilities) | | |
| `GRAPHKNOWS_LLM_MODEL` | `openrouter/deepseek/deepseek-v4-flash` | litellm model string; provider inferred from the prefix. |
| `GRAPHKNOWS_LLM_API_KEY` | _(empty)_ | LLM provider API key. |
| `GRAPHKNOWS_LLM_API_BASE` | _(empty)_ | Override the provider base URL (self-hosted/custom endpoints only). |
| `GRAPHKNOWS_LLM_TEMPERATURE` | `0.0` | LLM temperature. |
| `GRAPHKNOWS_LLM_MAX_TOKENS` | `2048` | LLM max tokens. |
| `GRAPHKNOWS_EXTRACTION_MODEL` | _(empty)_ | Override model for relation extraction (defaults to `LLM_MODEL`). |
| `GRAPHKNOWS_SUMMARIZATION_MODEL` | _(empty)_ | Override model for topic summarization (defaults to `LLM_MODEL`). |
| **Capability knobs** (each overrides the `GRAPHKNOWS_MODE` preset) | | |
| `GRAPHKNOWS_TOPICS` | _(preset)_ | `none` \| `extractive` \| `llm`. `extractive` is deterministic, so it is valid in `llm_free`. See [modes.md](modes.md). |
| `GRAPHKNOWS_DSPY_RELATIONS` | _(retired)_ | **Deprecated and ignored.** Setting it warns; `llm_assisted` replaces the whole extractor, so there is no "LLM relations on top" to switch. Use `GRAPHKNOWS_MODE`. |
| `GRAPHKNOWS_ONTOLOGY` | _(empty)_ | RDF/OWL ontology **file or folder** to inject. Mode-independent. See [ontology.md](ontology.md). |
| `GRAPHKNOWS_ONTOLOGY_FILE` | _(empty)_ | Deprecated alias for `GRAPHKNOWS_ONTOLOGY` (single file only). |
| `GRAPHKNOWS_RELATION_ONTOLOGY` | `personal` | Vocabulary relation properties are drawn from. Bare tokens `personal` \| `cco`, or a path to a custom vocabulary. Entity typing stays on `GRAPHKNOWS_ONTOLOGY`. See [ontology.md](ontology.md). |
| **Extraction / chunking** | | |
| `GRAPHKNOWS_RELEX_MODEL` | `knowledgator/gliner-relex-large-v1.0` | Joint entity+relation extraction model (HuggingFace id). |
| `GRAPHKNOWS_SPACY_MODEL` | `en_core_web_lg` | spaCy model for relation mining. |
| **LLM decoder** (only read in `llm_assisted`) | | |
| `GRAPHKNOWS_DECODE_ENTITIES` | `true` | Ask the one structured call for entities. |
| `GRAPHKNOWS_DECODE_RELATIONS` | `true` | Ask the one structured call for relations. |
| `GRAPHKNOWS_DECODE_FRAMES` | `true` | Ask the one structured call for frames. |
| `GRAPHKNOWS_DECODE_CONFIDENCE_MIN` | `0.5` | Drop relations below this confidence. `0` disables the gate. |
| `GRAPHKNOWS_DECODE_ATTEMPTS` | `3` | Per-chunk retry budget. A chunk that exhausts it abstains. |
| `GRAPHKNOWS_DECODE_BACKOFF_BASE_S` | `1.0` | Base of the exponential-with-full-jitter wait between attempts. |
| `GRAPHKNOWS_DECODE_BACKOFF_CAP_S` | `30.0` | Ceiling on that wait. |
| `GRAPHKNOWS_DECODE_HONOUR_RETRY_AFTER` | `true` | Let a provider `Retry-After` header win over the computed wait. |
| `GRAPHKNOWS_DECODE_CONCURRENCY` | `8` | Ceiling on per-chunk calls in flight during a batch. |
| **Frame layer** | | |
| `GRAPHKNOWS_ENABLE_FRAMES` | `true` | Deterministic FrameNet frame layer (ingest + retrieval channel). On by default; set to `false` to disable. See [modes.md](modes.md). |
| `GRAPHKNOWS_ENABLE_FE_TYPE_GATE` | `false` | Frame-element semantic-type veto: drops a role filler whose graph entity type conflicts with the frame element's FrameNet semantic type. Off by default; the FE plane's coreness / Requires / Excludes / CoreSet checks run regardless. See [architecture.md](architecture.md). |

## Model downloads

The GLiNER2 and sentence-transformers models download from the HuggingFace Hub on
first use; set `HF_HOME` to control the cache location. The spaCy model must be
downloaded once:

```bash
python -m spacy download en_core_web_lg
```

Unlike the models above, this one isn't distributed on PyPI, so neither `pip install
graphknows` nor any extra can pull it — it has to be downloaded manually, once. Use
the model named by `GRAPHKNOWS_SPACY_MODEL` if you've changed it from the default. A
missing model raises `graphknows.exceptions.MissingModelError`, naming the model and
this command.

## Choosing a provider (llm_assisted)

`GRAPHKNOWS_LLM_MODEL` is a [litellm](https://docs.litellm.ai/docs/providers) model
string, so any supported provider works by changing the prefix:

```bash
GRAPHKNOWS_LLM_MODEL=openai/gpt-4o-mini          GRAPHKNOWS_LLM_API_KEY=sk-...
GRAPHKNOWS_LLM_MODEL=anthropic/claude-sonnet-5   GRAPHKNOWS_LLM_API_KEY=sk-ant-...
GRAPHKNOWS_LLM_MODEL=openrouter/deepseek/deepseek-v4-flash  GRAPHKNOWS_LLM_API_KEY=sk-or-...
```

For a self-hosted OpenAI-compatible endpoint (e.g. vLLM), set `GRAPHKNOWS_LLM_API_BASE`.

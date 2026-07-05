# Configuration

All configuration is read from `GRAPHKNOWS_*` environment variables, or from a
`.env` file in the current working directory. Precedence is standard: constructor
arguments > environment variables > `.env` file. Settings are defined in
[`graphknows/config/__init__.py`](../graphknows/config/__init__.py) (`GraphKnowsSettings`).

```python
from graphknows import GraphKnowsSettings, GraphKnowsRuntime

# From environment / .env:
rt = GraphKnowsRuntime()

# Or explicitly:
settings = GraphKnowsSettings(mode="llm_assisted", llm_api_key="sk-...")
rt = GraphKnowsRuntime(settings)
```

## Reference

| Variable | Default | Purpose |
|---|---|---|
| `GRAPHKNOWS_MODE` | `llm_free` | `llm_free` or `llm_assisted`. Drives both extraction and retrieval. |
| `GRAPHKNOWS_ENV` | `development` | Set to `production` to enforce secret checks at server startup. |
| `GRAPHKNOWS_PORT` | `8025` | Web Memory Mirror port. |
| **ArcadeDB** | | |
| `GRAPHKNOWS_ARCADEDB_URL` | `http://localhost:2480` | ArcadeDB HTTP base URL. |
| `GRAPHKNOWS_ARCADEDB_USER` | `root` | ArcadeDB user. |
| `GRAPHKNOWS_ARCADEDB_PASSWORD` | `changeme` | ArcadeDB password. Required in production. |
| **Embeddings** | | |
| `GRAPHKNOWS_EMBED_MODEL` | `BAAI/bge-small-en-v1.5` | Local sentence-transformers model, or a remote model id when `EMBED_API_BASE` is set. |
| `GRAPHKNOWS_EMBED_API_BASE` | _(empty)_ | Set to use a remote OpenAI-compatible embeddings endpoint instead of local. |
| `GRAPHKNOWS_EMBED_API_KEY` | _(empty)_ | API key for the remote embeddings endpoint. |
| **LLM** (llm_assisted only) | | |
| `GRAPHKNOWS_LLM_MODEL` | `openrouter/deepseek/deepseek-v4-flash` | litellm model string; provider inferred from the prefix. |
| `GRAPHKNOWS_LLM_API_KEY` | _(empty)_ | LLM provider API key. |
| `GRAPHKNOWS_LLM_API_BASE` | _(empty)_ | Override the provider base URL (self-hosted/custom endpoints only). |
| `GRAPHKNOWS_LLM_TEMPERATURE` | `0.0` | LLM temperature. |
| `GRAPHKNOWS_LLM_MAX_TOKENS` | `2048` | LLM max tokens. |
| `GRAPHKNOWS_EXTRACTION_MODEL` | _(empty)_ | Override model for relation extraction (defaults to `LLM_MODEL`). |
| `GRAPHKNOWS_SUMMARIZATION_MODEL` | _(empty)_ | Override model for topic summarization (defaults to `LLM_MODEL`). |
| `GRAPHKNOWS_ONTOLOGY_FILE` | _(empty)_ | Path to an RDF/OWL ontology to inject. See [ontology.md](ontology.md). |
| **Extraction / chunking** | | |
| `GRAPHKNOWS_GLINER_MODEL` | `fastino/gliner2-multi-v1` | GLiNER2 entity model (HuggingFace id). |
| `GRAPHKNOWS_GLINER_THRESHOLD` | `0.5` | Entity confidence threshold. |
| `GRAPHKNOWS_SPACY_MODEL` | `en_core_web_lg` | spaCy model for relation mining. |
| `GRAPHKNOWS_CHUNK_SIZE` | `1200` | Chunk size (characters). |
| `GRAPHKNOWS_CHUNK_OVERLAP` | `100` | Chunk overlap (characters). |
| `GRAPHKNOWS_SESSION_TTL_MINUTES` | `60` | Session inactivity before auto-merge. |

## Model downloads

The GLiNER2 and sentence-transformers models download from the HuggingFace Hub on
first use; set `HF_HOME` to control the cache location. The spaCy model must be
downloaded once:

```bash
python -m spacy download en_core_web_lg
```

## Choosing a provider (llm_assisted)

`GRAPHKNOWS_LLM_MODEL` is a [litellm](https://docs.litellm.ai/docs/providers) model
string, so any supported provider works by changing the prefix:

```bash
GRAPHKNOWS_LLM_MODEL=openai/gpt-4o-mini          GRAPHKNOWS_LLM_API_KEY=sk-...
GRAPHKNOWS_LLM_MODEL=anthropic/claude-sonnet-5   GRAPHKNOWS_LLM_API_KEY=sk-ant-...
GRAPHKNOWS_LLM_MODEL=openrouter/deepseek/deepseek-v4-flash  GRAPHKNOWS_LLM_API_KEY=sk-or-...
```

For a self-hosted OpenAI-compatible endpoint (e.g. vLLM), set `GRAPHKNOWS_LLM_API_BASE`.

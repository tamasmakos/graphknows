# Modes: llm_free vs llm_assisted

GraphKnows has one switch — `GRAPHKNOWS_MODE` — that governs **both** extraction and
retrieval. Mode resolution lives in
[`graphknows/config/profiles.py`](../graphknows/config/profiles.py).

| | `llm_free` (default) | `llm_assisted` |
|---|---|---|
| Entity extraction | GLiNER2 (deterministic) | GLiNER2 |
| Relation extraction | spaCy dependency mining (deterministic) | DSPy + LLM, ontology-constrained |
| Retrieval | `DETRetriever` — vector ANN + graph traversal + BM25 + temporal, fused with RRF, cross-encoder rerank | `AgenticRetriever` — DSPy ReAct decomposition over the same collectors |
| Embeddings | local sentence-transformers (default) | local sentence-transformers (default) |
| API keys required | **none** | LLM key (`GRAPHKNOWS_LLM_API_KEY`) |
| Install | `pip install graphknows` | `pip install "graphknows[assisted]"` |
| Determinism | fully deterministic ingest + retrieval | LLM-driven, non-deterministic |
| Best for | privacy, cost, reproducibility, offline | domain-specific graphs, higher recall on hard queries |

Note: in both modes, answer **generation** (if you use the LLM to write a final
answer over retrieved context) is a separate concern from memory — GraphKnows returns
ranked evidence; wiring an LLM to synthesize a final answer is up to your agent.

## llm_free

```python
from graphknows import GraphKnowsRuntime  # GRAPHKNOWS_MODE defaults to llm_free

async with GraphKnowsRuntime() as rt:
    await rt.ingest_memory("Maria moved to Berlin in 2021.", session_id="s1")
    hits = await rt.recall_memory("Where does Maria live?", session_id="s1")
```

No API keys, no network calls on the ingest or retrieval path (embeddings run
locally). This is the recommended starting point.

## llm_assisted

```bash
pip install "graphknows[assisted]"
export GRAPHKNOWS_MODE=llm_assisted
export GRAPHKNOWS_LLM_MODEL=openrouter/deepseek/deepseek-v4-flash
export GRAPHKNOWS_LLM_API_KEY=sk-...
# optional — keep the graph domain-specific:
export GRAPHKNOWS_ONTOLOGY_FILE=/path/to/ontology.ttl
```

```python
from graphknows import GraphKnowsRuntime, GraphKnowsSettings

async with GraphKnowsRuntime(GraphKnowsSettings(mode="llm_assisted")) as rt:
    await rt.ingest_memory(conversation, session_id="s1")
    hits = await rt.recall_memory("...", session_id="s1")
```

The LLM extracts richer relations; an injected ontology constrains them to your
domain's vocabulary. See [ontology.md](ontology.md).

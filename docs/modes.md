# Modes: llm_free vs llm_assisted

`GRAPHKNOWS_MODE` is a **preset**, not a switch. It expands into independent
capability knobs, every one of which can be overridden on its own. The preset
table lives in [`graphknows/settings.py`](../graphknows/settings.py) (`_PRESETS`).

| Knob | Env var | `llm_free` (default) | `llm_assisted` |
|---|---|---|---|
| Topic layer | `GRAPHKNOWS_TOPICS` | `none` | `llm` |
| LLM relation extraction | `GRAPHKNOWS_DSPY_RELATIONS` | `false` | `true` |
| Retrieval strategy | `GRAPHKNOWS_RETRIEVAL` | `det` | `agentic` |

Read the resolved values through `settings.topic_mode`, `settings.dspy_relations`
and `settings.retrieval_strategy` — never off the raw fields, which are `None`
when unset.

## What the mode does *not* control

These are mode-independent and behave identically in both:

| | |
|---|---|
| Entity extraction | GLiNER relex + spaCy-mined labels — always model-driven, never LLM-driven |
| Deterministic relations | relex + SVO mining — runs in both modes; DSPy is *additive on top* |
| Frame layer | `GRAPHKNOWS_ENABLE_FRAMES`, on by default |
| Ontology injection | `GRAPHKNOWS_ONTOLOGY` — steers labels and links entities in both modes ([ontology.md](ontology.md)) |
| Embeddings | local sentence-transformers by default |

So the residual difference the preset names is: **DSPy relation extraction, LLM
topic summaries, and agentic retrieval.**

## Mixing knobs

The combinations the preset table alone cannot express are the useful ones.

```bash
# Deterministic ingest, but let a ReAct loop answer hard queries.
GRAPHKNOWS_MODE=llm_free GRAPHKNOWS_RETRIEVAL=agentic

# Topic layer without any network call: titles from PageRank, summaries from
# the member chunk nearest the community centroid.
GRAPHKNOWS_MODE=llm_free GRAPHKNOWS_TOPICS=extractive

# LLM relations, but deterministic retrieval — isolates which half of
# llm_assisted a result came from.
GRAPHKNOWS_MODE=llm_assisted GRAPHKNOWS_RETRIEVAL=det
```

An API key is required by *capability*, not by mode name: `llm_free` with
`GRAPHKNOWS_RETRIEVAL=agentic` needs `GRAPHKNOWS_LLM_API_KEY`, and
`llm_assisted` with every LLM capability switched off does not. See
`GraphKnowsSettings.uses_llm`.

## Retrieval strategy is also per call

Retrieval strategy is not welded to how the graph was ingested — nothing writes
a mode marker to the store, and the agentic retriever wraps the deterministic
one over the same store. So one server can answer both ways:

```python
await memory.recall_memory("...", session_id="s1", strategy="det")
await memory.recall_memory("...", session_id="s1", strategy="agentic")
```

The MCP `memory_query` tool takes the same `strategy` argument and returns
`search_type` so callers can confirm which retriever answered. Omit it to use
`GRAPHKNOWS_RETRIEVAL`.

## llm_free

```python
from graphknows import Memory  # GRAPHKNOWS_MODE defaults to llm_free

async with Memory() as mem:
    await mem.ingest_memory("Maria moved to Berlin in 2021.", session_id="s1")
    hits = await mem.recall_memory("Where does Maria live?", session_id="s1")
```

No API keys and no network calls on the ingest or retrieval path (embeddings run
locally). This is the recommended starting point.

## llm_assisted

```bash
pip install "graphknows[assisted]"
export GRAPHKNOWS_MODE=llm_assisted
export GRAPHKNOWS_LLM_MODEL=openrouter/deepseek/deepseek-v4-flash
export GRAPHKNOWS_LLM_API_KEY=sk-...
# optional, and equally available in llm_free:
export GRAPHKNOWS_ONTOLOGY=/path/to/ontology.ttl
```

```python
from graphknows import Memory, GraphKnowsSettings

async with Memory(GraphKnowsSettings(mode="llm_assisted")) as mem:
    await mem.ingest_memory(conversation, session_id="s1")
    hits = await mem.recall_memory("...", session_id="s1")
```

The LLM adds relations the deterministic pass cannot state, on top of the ones
it already found.

Note: in both modes, answer **generation** is a separate concern from memory —
GraphKnows returns ranked evidence; wiring an LLM to synthesize a final answer
is up to your agent.

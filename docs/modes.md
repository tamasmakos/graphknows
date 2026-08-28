# Modes: llm_free vs llm_assisted

`GRAPHKNOWS_MODE` is a **preset**, not a switch. It expands into independent
capability knobs, every one of which can be overridden on its own. The preset
table lives in [`graphknows/settings.py`](../graphknows/settings.py) (`_PRESETS`).

| Knob | Env var | `llm_free` (default) | `llm_assisted` |
|---|---|---|---|
| Topic layer | `GRAPHKNOWS_TOPICS` | `none` | `llm` |
| LLM relation extraction | `GRAPHKNOWS_DSPY_RELATIONS` | `false` | `true` |

Read the resolved values through `settings.topic_mode` and
`settings.dspy_relations` — never off the raw fields, which are `None` when
unset.

There is one retrieval engine and it is deterministic. The agentic (ReAct)
retriever and the `GRAPHKNOWS_RETRIEVAL` knob that selected it have been
removed.

## What the mode does *not* control

These are mode-independent and behave identically in both:

| | |
|---|---|
| Entity extraction | GLiNER relex + spaCy-mined labels — always model-driven, never LLM-driven |
| Deterministic relations | relex + SVO mining — runs in both modes; DSPy is *additive on top* |
| Frame layer | `GRAPHKNOWS_ENABLE_FRAMES`, on by default |
| Frame-element semantic-type veto | `GRAPHKNOWS_ENABLE_FE_TYPE_GATE`, off by default — drops a role filler whose entity type conflicts with the frame element's FrameNet semantic type; the coreness/Requires/Excludes/CoreSet validity checks it layers on top of always run |
| Ontology injection | `GRAPHKNOWS_ONTOLOGY` — steers labels and links entities in both modes ([ontology.md](ontology.md)) |
| Embeddings | local sentence-transformers by default |

So the residual difference the preset names is: **DSPy relation extraction and
LLM topic summaries.**

## Mixing knobs

The combinations the preset table alone cannot express are the useful ones.

```bash
# Topic layer without any network call: titles from PageRank, summaries from
# the member chunk nearest the community centroid.
GRAPHKNOWS_MODE=llm_free GRAPHKNOWS_TOPICS=extractive

# LLM relations without the LLM topic layer — isolates which half of
# llm_assisted a result came from.
GRAPHKNOWS_MODE=llm_assisted GRAPHKNOWS_TOPICS=none
```

An API key is required by *capability*, not by mode name: `llm_assisted` with
every LLM capability switched off does not need one. See
`GraphKnowsSettings.uses_llm`.

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

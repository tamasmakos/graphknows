# Modes: llm_free vs llm_assisted

`GRAPHKNOWS_MODE` is a **preset**, not a switch. It expands into independent
capability knobs; the topic layer can be overridden on its own. The preset
table lives in [`graphknows/settings.py`](../graphknows/settings.py) (`_PRESETS`).

| Knob | Env var | `llm_free` (default) | `llm_assisted` |
|---|---|---|---|
| Topic layer | `GRAPHKNOWS_TOPICS` | `none` | `llm` |
| Decoder | _(none — follows the mode)_ | `local` | `llm` |

Read the resolved values through `settings.topic_mode` and `settings.decoder` —
never off the raw fields, which are `None` when unset. The decoder has no
per-capability override: it *is* the mode, so overriding it separately would be
overriding the mode. The retired `GRAPHKNOWS_DSPY_RELATIONS` knob warns and is
ignored.

There is one retrieval engine and it is deterministic. The agentic (ReAct)
retriever and the `GRAPHKNOWS_RETRIEVAL` knob that selected it have been
removed.

## What the mode does *not* control

These are mode-independent and behave identically in both:

| | |
|---|---|
| spaCy passes | chunk entity-schema mining, speaker and deixis resolution, the verb-surface filter, lexical label selection, modality — spaCy is the parse, not the decoder |
| Frame candidates | the LU-trigger lookup proposes the frames; only role filling moves with the decoder |
| Frame layer | `GRAPHKNOWS_ENABLE_FRAMES`, on by default |
| Frame-element semantic-type veto | `GRAPHKNOWS_ENABLE_FE_TYPE_GATE`, off by default — drops a role filler whose entity type conflicts with the frame element's FrameNet semantic type; the coreness/Requires/Excludes/CoreSet validity checks it layers on top of always run |
| Ontology injection | `GRAPHKNOWS_ONTOLOGY` — steers labels and links entities in both modes ([ontology.md](ontology.md)) |
| Embeddings | local sentence-transformers by default |

So the difference the preset names is: **which decoder extracts, and whether
topic summaries are written by an LLM.**

## Mixing knobs

The combinations the preset table alone cannot express are the useful ones.

```bash
# Topic layer without any network call: titles from PageRank, summaries from
# the member chunk nearest the community centroid.
GRAPHKNOWS_MODE=llm_free GRAPHKNOWS_TOPICS=extractive

# The LLM decoder without the LLM topic layer — isolates which half of
# llm_assisted a result came from.
GRAPHKNOWS_MODE=llm_assisted GRAPHKNOWS_TOPICS=none

# The LLM decoder without its relation section — attributes a measured
# regression to one section of the response rather than to "the LLM".
GRAPHKNOWS_MODE=llm_assisted GRAPHKNOWS_DECODE_RELATIONS=false
```

The `GRAPHKNOWS_DECODE_*` knobs are read only under the `llm` decoder and are
listed in [configuration.md](configuration.md).

An API key is required by *capability*: the LLM decoder or the LLM topic layer.
`llm_assisted` runs the LLM decoder, so it always needs one — and says so at
construction. See `GraphKnowsSettings.uses_llm`.

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
export GRAPHKNOWS_LLM_MODEL=openrouter/deepseek/deepseek-v4-flash
export GRAPHKNOWS_LLM_API_KEY=sk-...
# optional, and equally available in llm_free:
export GRAPHKNOWS_ONTOLOGY=/path/to/ontology.ttl
```

```python
from graphknows import Memory

async with Memory(mode="llm_assisted") as mem:
    await mem.ingest_memory(conversation, session_id="s1")
    hits = await mem.recall_memory("...", session_id="s1")
```

The LLM **replaces** the extraction stack rather than layering on it. One
structured call per chunk returns the entities, the relations (each with a
confidence) and the frame roles, over the same closed per-chunk vocabularies the
local decoder is handed — so everything downstream, from entity resolution to
retrieval, sees the same shape and never branches on who produced it. GLiNER2
and the DeBERTa verifier are not loaded at all in this mode.

A missing key fails at construction, not at the first chunk of a long ingest.
A call that fails its retry budget abstains: the chunk is still stored, embedded
and retrievable, the abstention is counted under its own gate, and the chunk is
marked in the graph. There is no per-chunk fallback to the local decoder, ever —
`await mem.redecode(session_id)` re-drives just the marked chunks later.

Note: in both modes, answer **generation** is a separate concern from memory —
GraphKnows returns ranked evidence; wiring an LLM to synthesize a final answer
is up to your agent.

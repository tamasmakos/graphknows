---
title: "Deep Research Agent Design"
created: 2026-05-12
updated: 2026-05-12
type: concept
tags: [agent, langgraph, design, research-loop, novelty]
confidence: high
---

# Deep Research Agent Design

The agent implements a **generate → search → reflect → re-query** loop pattern, drawing from STORM (Shao et al. 2024 — outline-driven multi-perspective synthesis) and GPT-Researcher (Assaf Elovic 2023 — parallel sub-query search with deduplication). Unlike both, Graphknows writes every iteration's evidence into a per-session KG+vector store before synthesis, enabling graph-structured retrieval rather than flat context concatenation.

See [[product-architecture-v2]] for the macro architecture. See [[dual-write-hybrid-retrieval]] for the write/retrieval layer.

---

## Loop Pattern

```
plan_node
  │
  └─► for each sub-question in plan:
        search_loop iteration n:
          1. generate query strings (from plan + reflection on prior results)
          2. search (services/search HTTP)
          3. embed + novelty check
          4. dual_write (pgvector chunks + Memgraph entities)
          5. reflect: what gaps remain?
          6. re-query with gap-filling queries
          → exit if dup_ratio > 0.6 OR iteration_count > 12
  │
  └─► hybrid_retrieval
  └─► brief_node
```

**Reflect step:** After each iteration, the agent scores coverage against the original plan sub-questions. Uncovered sub-questions generate new query strings for the next iteration. This is the "reflect" in generate→search→reflect→re-query.

---

## Novelty Stopping Criterion

After each iteration, the agent computes:

```
dup_ratio = count(chunks with cosine_sim > 0.92 to any prior chunk) / total_new_chunks
```

Exit condition (either):
- `dup_ratio > 0.6` — majority of new chunks are near-duplicates of already-stored material
- `iteration_count > 12` — hard cap to bound latency and cost

Rationale: a cosine threshold of 0.92 (BAAI/bge-small-en-v1.5 space) reliably separates paraphrase from novel evidence at EU policy text density. The 0.6 ratio allows partial novelty in a batch before stopping.

Implementation: `graph/nodes/investigate.py` → `_novelty_check(state: InvestigativeState) -> bool`

---

## LangGraph Subgraph Structure

`graph/workflow.py` defines a 3-phase macro `StateGraph`:

```python
workflow = StateGraph(InvestigativeState)
workflow.add_node("scope",       scope_node)        # plan_node
workflow.add_node("investigate", investigate_node)  # search_loop + dual_write
workflow.add_node("communicate", communicate_node)  # hybrid_retrieval + brief_node

workflow.set_entry_point("scope")
workflow.add_edge("scope", "investigate")
workflow.add_conditional_edges(
    "investigate",
    should_continue,   # novelty check → loop or exit
    {"continue": "investigate", "done": "communicate"}
)
workflow.add_edge("communicate", END)
```

`investigate_node` is itself a subgraph (inline loop, not a nested StateGraph) that increments `state.iteration_count` and appends to `state.chunks` and `state.entities` each pass.

---

## InvestigativeState Schema

Defined in `graph/state.py`:

```python
class InvestigativeState(TypedDict):
    session_id:      str                   # UUID; scope of all KG + pgvector writes
    topic:           str                   # original user research topic
    plan:            list[SubQuestion]     # output of plan_node
    queries:         list[str]             # current iteration's query strings
    raw_docs:        list[RawDoc]          # fetched docs this iteration
    chunks:          list[Chunk]           # embedded chunks (all iterations)
    entities:        list[Entity]          # extracted entities (all iterations)
    relations:       list[Relation]        # extracted relations (all iterations)
    iteration_count: int                   # current search loop iteration
    dup_ratio:       float                 # last computed novelty signal
    retrieval_hits:  list[Chunk]           # top-8 from hybrid_retrieval
    briefing:        str | None            # final EPRS-style briefing text
    ws_events:       list[WSEvent]         # event log for streaming to frontend
```

`SubQuestion`, `RawDoc`, `Chunk`, `Entity`, `Relation` are typed dataclasses in `graph/state.py`.

---

## Tool List

The agent (`services/agent/`) exposes or internally uses the following tools:

| Tool | Location | Description |
|---|---|---|
| `search_web` | `tools/search_client.py` | Tavily web search via `services/search/` HTTP |
| `search_eu` | `tools/search_client.py` | EU adapter queries (EP/EUR-Lex/Council/EC/ECB/ECA/EPRS) |
| `graphgen_extract` | HTTP → `services/graphgen/` | GLiNER2 NER + DSPy RE on raw text |
| `dual_write` | `graph/nodes/investigate.py` | Write chunks to pgvector + entities to Memgraph |
| `hybrid_retrieve` | `retrieval/hybrid_retriever.py` | pgvector dense + 2-hop Memgraph + RRF |
| `render_brief` | `graph/nodes/communicate.py` | EPRS-style briefing renderer |

Tools are **not** LangChain tool-call tools — they are plain async Python functions called directly by the StateGraph nodes. No LLM function-calling schema wrapping needed at this layer.

---

## WebSocket Event Schema

All state transitions stream WSEvents to `apps/web/` via the agent's WS endpoint. Schema:

```python
class WSEvent(TypedDict):
    session_id:  str
    event_type:  Literal[
        "plan_ready",          # plan_node done; plan attached
        "search_start",        # iteration n starting; queries attached
        "docs_fetched",        # raw docs count attached
        "chunks_written",      # n chunks written to pgvector
        "entities_written",    # n entities/relations written to Memgraph
        "novelty_check",       # dup_ratio + iteration_count attached
        "search_done",         # loop exited; reason attached
        "retrieval_done",      # top-8 chunks attached
        "brief_chunk",         # streaming briefing text token
        "brief_done",          # briefing complete
        "error",               # error message attached
    ]
    payload:     dict          # event-specific data
    ts:          float         # unix timestamp
```

Frontend uses `event_type` to update: plan panel, live doc counter, KG node count (via `entities_written`), and briefing text stream.

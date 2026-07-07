# Integrating graphknows into an agent framework

graphknows exposes one small verb set. Integrating it into any framework is a
matter of wiring that framework's memory hooks to these verbs — in-process via
`graphknows.Memory`, or out-of-process via `graphknows.client.RemoteMemory`
(same method names, MCP transport).

## The generic recipe

| Framework concept | graphknows verb |
| --- | --- |
| write hook / after a turn | `ingest_memory(text, session_id=...)` (or `record_turn`) |
| read hook / before generation | `recall_memory(query, session_id=..., top_k=...)` |
| conversation / thread id | `session_id` |
| end of episode / consolidation | `flush_memory(session_id)` (promote STM → LTM) |
| reset | `purge_memory(scope=..., session_id=...)` |

Map the framework's "remember" point to `ingest_memory`, its "recall" point to
`recall_memory`, and its thread identifier to `session_id`. That is the whole
integration.

## LangGraph (shipped reference adapter)

Install: `pip install 'graphknows[langgraph]'`.

Two ready-made, framework-agnostic nodes live in
`graphknows.adapters.langgraph` — they take a `state` dict and a memory, and
return a partial state update, so they drop into any `state -> partial_state`
graph:

```python
from functools import partial
from graphknows import Memory
from graphknows.adapters.langgraph import recall, remember

memory = Memory()
graph.add_node("recall", partial(recall, memory=memory, session_id=thread_id))
graph.add_node("remember", partial(remember, memory=memory, session_id=thread_id))
```

`recall` reads the last message as the query and writes hit texts to
`state["memories"]`; `remember` ingests the last message. See
`examples/langgraph_agent.py` for a runnable graph.

For LangGraph's `BaseStore` extension point, use `GraphKnowsStore`:

```python
from graphknows.adapters.langgraph import GraphKnowsStore

store = GraphKnowsStore(memory)          # namespace tuple → session_id
app = graph.compile(store=store)         # store.search → recall_memory
```

## Claude Agent SDK (sketch)

Claude Code / the Agent SDK already speak MCP, so the cleanest integration is
out-of-process: point them at the `graphknows-mcp` server, or drive it yourself
with `RemoteMemory`, which mirrors the `Memory` verbs over MCP:

```python
from graphknows.client import RemoteMemory

async with RemoteMemory(command="graphknows-mcp") as mem:
    await mem.ingest_memory(user_turn, session_id=thread_id)
    hits = await mem.recall_memory(next_question, session_id=thread_id, top_k=5)
```

Wrap `ingest_memory` and `recall_memory` as two tools (or a pre/post hook) and
the agent gains persistent, graph-backed memory with no ML dependencies in its
own process.

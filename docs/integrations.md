# Integrating graphknows into an agent framework

graphknows exposes one small verb set. Integrating it into any framework is a
matter of wiring that framework's memory hooks to these verbs — in-process via
`graphknows.Memory`, or out-of-process via
`graphknows.integrations.client.GraphKnowsMCPClient` (the same verbs as MCP tool
names, over the MCP transport).

## The generic recipe

| Framework concept | graphknows verb |
| --- | --- |
| write hook / after a turn | `ingest_memory(text, session_id=...)` |
| read hook / before generation | `recall_memory(query, session_id=..., top_k=...)` |
| conversation / thread id | `session_id` |
| end of episode / consolidation | `flush_memory(session_id)` (consolidates the session in place) |
| reset | `purge_memory(scope=..., session_id=...)` |

Map the framework's "remember" point to `ingest_memory`, its "recall" point to
`recall_memory`, and its thread identifier to `session_id`. That is the whole
integration.

## LangGraph (shipped reference adapter)

Install LangGraph alongside graphknows: `pip install graphknows langgraph`. The adapter ships in the base package; it imports LangGraph lazily.

Two ready-made, framework-agnostic nodes live in
`graphknows.integrations.langgraph` — they take a `state` dict and a memory, and
return a partial state update, so they drop into any `state -> partial_state`
graph:

```python
from functools import partial
from graphknows import Memory
from graphknows.integrations.langgraph import recall, remember

memory = Memory()
graph.add_node("recall", partial(recall, memory=memory, session_id=thread_id))
graph.add_node("remember", partial(remember, memory=memory, session_id=thread_id))
```

`recall` reads the last message as the query and writes hit texts to
`state["memories"]`; `remember` ingests the last message. See
`examples/langgraph_agent.py` for a runnable graph.

For LangGraph's `BaseStore` extension point, use `GraphKnowsStore`:

```python
from graphknows.integrations.langgraph import GraphKnowsStore

store = GraphKnowsStore(memory)          # namespace tuple → session_id
app = graph.compile(store=store)         # store.search → recall_memory
```

## Claude Agent SDK (sketch)

Claude Code / the Agent SDK already speak MCP, so the cleanest integration is
out-of-process: point them at the `graphknows-mcp` server, or drive it yourself
with `GraphKnowsMCPClient`:

```python
from graphknows.integrations.client import GraphKnowsMCPClient

async with GraphKnowsMCPClient(command="graphknows-mcp") as client:
    await client.call_tool(
        "memory_ingest", {"text": user_turn, "session_id": thread_id}
    )
    hits = await client.call_tool(
        "memory_query",
        {"query": next_question, "session_id": thread_id, "top_k": 5},
    )
```

Wrap `memory_ingest` and `memory_query` as two tools (or a pre/post hook) and
the agent gains persistent, graph-backed memory with no ML dependencies in its
own process.

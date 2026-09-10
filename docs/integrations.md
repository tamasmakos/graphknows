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

## Claude Code hooks (shipped)

The hook integration wires each Claude Code event to one verb of
`python -m graphknows.integrations.claude_code`. Each verb reads the hook event
as JSON on stdin and writes a `hookSpecificOutput` block on stdout, or nothing
at all:

| Event | Verb | What it does |
| --- | --- | --- |
| `SessionStart` | `context` | Opens the session with what memory already holds. |
| `UserPromptSubmit` | `recall` | Injects the facts the prompt's symbols activate. |
| `PreToolUse` (Edit, Write) | `preview` | Surfaces what memory knows about what is about to change. |
| `PostToolUse` (Read, Grep) | `observe` | Records what the session looked at. |
| `Stop` | `remember` | Ingests the turns added since the last checkpoint. |
| `PreCompact` | `catchup` | Ingests the transcript before the harness compacts it. |

Injection is symbol-keyed: a query that resolves to no symbol injects nothing,
so an unrelated prompt costs no context. The hook process reaches memory only
over `graphknows.integrations.client`, so it loads no machine-learning
dependency and stays fast enough to run on every prompt.

Both data files ship inside the wheel, readable without a checkout:

```python
from graphknows.integrations.claude_code import guidance, settings_block

settings_block()   # the hook block to merge into a Claude Code settings.json
guidance()         # the skill telling an agent when to record and when to retrieve
```

A transcript is still being appended to while the session that writes it fires
its `Stop` and `PreCompact` hooks, so reading stops at the first line that is
not whole JSON and the checkpoint only ever names a whole record — the
half-written line is picked up on the next event, and no turn is ingested twice.

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

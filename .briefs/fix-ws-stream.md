# Brief: Fix ws_stream to use ADR-002 workflow with event streaming

## Context
File: `/root/graphknows/services/agent/router.py`
Branch: `feat/deep-research-agent-v2`
Model: claude-sonnet-4.6 via GitHub Copilot

The `ws_stream` function (line 831) and its alias `ws_chat` (line 962) currently use
the OLD `langgraph_agent.build_graph`. We need it to use the ADR-002 `build_workflow`
already wired for the `POST /sessions/{id}/query` endpoint.

## Bug 1: Wrong message key
Client sends `{"query": "..."}` but server reads `data.get("message", "")`.
Fix: read both `data.get("query") or data.get("message", "")`.

## Bug 2: Old workflow used
`ws_stream` calls `agent.langgraph_agent.build_graph` — replace with the new workflow.

## What the ADR-002 workflow produces
`build_workflow(pg_client, mg_driver, search_client)` returns a LangGraph `CompiledGraph`.
It takes this initial state dict:

```python
{
    "session_id": str,
    "query": str,
    "plan": [],
    "sub_questions": [],
    "iterations": [],
    "chunks": [],
    "entities": [],
    "relations": [],
    "ws_events": [],  # list of WSEvent dicts the nodes append to
    "done": False,
    "iteration_count": 0,
    "dup_ratio": 0.0,
}
```

The `ws_events` list in the final state contains dicts like:
```python
{"type": "thought_token", "token": "..."}
{"type": "tool_start", "tool": "...", "input": "..."}
{"type": "tool_result", "tool": "...", "output": "..."}
{"type": "answer_token", "token": "..."}
{"type": "graph_update", "nodes": [...], "edges": [...]}
```

The final state also has `briefing: str`, `iteration_count: int`, `dup_ratio: float`.

## Required change to ws_stream

Replace the body of `ws_stream` with an implementation that:

1. Reads `query = data.get("query") or data.get("message", "")` — accept both keys
2. Checks `if not query` → send `{"type": "error", "message": "empty query"}` and return
3. Builds the ADR-002 workflow: `workflow = build_workflow(pg_client=_get_pg_client(), mg_driver=_get_sgm().mg_driver, search_client=_get_search_client())`
4. Constructs initial state as above using `session_id` and `query`
5. Runs `result = await workflow.ainvoke(initial_state)` (simple invoke — ws_events are collected in state)
6. After invoke, replays all `result["ws_events"]` to the websocket in order
7. Then sends the final `done` event: `{"type": "done", "briefing": result.get("briefing",""), "iteration_count": result.get("iteration_count",0), "dup_ratio": result.get("dup_ratio",0.0)}`
8. Keep the existing error handling (WebSocketDisconnect, generic except)

## Import needed
```python
from agent.graph.workflow import build_workflow  # already used in POST endpoints
```
This import is already done inside the POST handlers; move it to top of file or keep it inline.

## IMPORTANT constraints
- Do NOT change the `POST /chat` or `POST /sessions/{id}/query` endpoints — those are working
- Do NOT change the function signature of `ws_stream` or `ws_chat`
- Keep `_ws_connections` bookkeeping
- Keep `_active_sessions` bookkeeping and `sgm.create_session` call
- The old imports from `agent.langgraph_agent` inside `ws_stream` can be removed

## Acceptance criteria
After your change:
```bash
cd /root/graphknows
set -a && source .env && set +a
PYTHONPATH=/root/graphknows/services:/root/graphknows/services/agent:/root/graphknows/services/scrapers \
python3 -c "
import sys; sys.path.insert(0,'services'); sys.path.insert(0,'services/agent')
from agent.router import ws_stream
print('ws_stream import OK')
"
```
Must print `ws_stream import OK` with no errors.

Run the existing test suite after your change:
```bash
cd /root/graphknows
PYTHONPATH=/root/graphknows/services:/root/graphknows/services/agent \
python3 -B -m pytest tests/test_workflow_wired.py tests/test_hybrid_retriever.py tests/test_communicate_node.py -x -q 2>&1 | tail -10
```
All tests must pass.

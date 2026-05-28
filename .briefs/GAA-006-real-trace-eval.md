# GAA-006: Real-trace evaluation harness — Brief

## Context
Working directory: `/root/graphknows`
GAA-001 through GAA-005 are complete. The adapter, types, and summarizer all exist.

Real session data lives in `/opt/data/state.db` (SQLite, Hermes session messages).

Read before starting:
```
cat /root/graphknows/.hermes/instructions/python-style.instructions.md
```

Also read:
- `graphknows/memory/adapter_types.py`
- `graphknows/memory/adapter.py`
- `graphknows/memory/tool_summarizer.py`
- `research/proto_generic_agent_adapter.py` (if it exists — reference prototype)
- `research/proto_generic_agent_adapter_results.json` (if it exists — baseline metrics)

## Objective
Write a standalone eval harness in `research/` (gitignored) that:
1. Reads the 3 most recent sessions from `/opt/data/state.db`
2. Parses each session's messages into `TurnRecord` objects using the `GraphknowsAdapter` types
3. Summarizes tool calls using `ToolSummarizer`
4. Writes the parsed data to graphknows STM (Memgraph + pgvector) via the real `GraphknowsAdapter`
   — using live DBs: Memgraph bolt://172.18.0.1:17687, Postgres 172.18.0.1:5432
5. Runs recall queries against the written data
6. Prints a JSON evaluation report with metrics

## File to create
- `research/eval_generic_adapter_real_traces.py`

## SQLite schema
Run `sqlite3 /opt/data/state.db .schema` first to inspect the actual schema.
The messages table likely has: id, session_id, role, content, created_at, and possibly tool_name/tool_output fields.

## Parsing rules

### Message → TurnRecord mapping
- `role == "user"` → `TurnRecord(role="user", ...)`
- `role == "assistant"` → `TurnRecord(role="assistant", ...)`
- `role == "tool"` → attach as `ToolCallRecord` to the PREVIOUS assistant turn
- Skip messages with empty content

### ToolCallRecord extraction
When `role == "tool"`:
- `tool_name`: from `tool_name` column or "unknown" if missing
- `input_summary`: first 200 chars of the previous assistant message that triggered this tool, or ""
- `output_summary`: first 300 chars of this message's content
- `exit_code`: None (not tracked in SQLite)
- `error`: None if output_summary doesn't contain "Error" or "Traceback", else "error"

### ArtifactRecord extraction
Look for messages where content matches: starts with "```" or contains "def ", "class ", "import "
and content length > 200. Extract as `ArtifactRecord` with:
- `artifact_id`: f"artifact-{message_id}"
- `path`: f"session:{session_id}:msg:{message_id}"
- `mime_type`: "text/x-python" if it looks like Python, else "text/plain"
- `memory_class`: `MemoryClass.raw_evidence`

## Eval script design

```python
#!/usr/bin/env python3
"""
Real-trace evaluation harness for GraphknowsAdapter.

Reads recent sessions from /opt/data/state.db, parses into adapter types,
writes to graphknows STM, runs recall queries, prints JSON metrics.

Usage:
    python research/eval_generic_adapter_real_traces.py
    python research/eval_generic_adapter_real_traces.py --namespace my-eval-ns
    python research/eval_generic_adapter_real_traces.py --sessions 5
"""
import argparse
import json
import sqlite3
import sys
import time
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from graphknows.memory.adapter import GraphknowsAdapter
from graphknows.memory.adapter_types import (
    ArtifactRecord, MemoryClass, ParticipantRecord, RecallRequest, ToolCallRecord, TurnRecord,
)
from graphknows.memory.tool_summarizer import ToolSummarizer
from graphknows.retrieval.stores.memgraph import MemgraphStore
from graphknows.retrieval.stores.pgvector import PgVectorStore
```

## Store initialization
```python
import asyncio
import os

MEMGRAPH_BOLT = os.getenv("MEMGRAPH_BOLT", "bolt://172.18.0.1:17687")
POSTGRES_DSN = os.getenv("POSTGRES_DSN", "postgresql://...")  # read from .env
NAMESPACE = args.namespace  # e.g. "eval-real-traces-2026-05-23"

# Load .env for Postgres DSN
from dotenv import load_dotenv
load_dotenv("/root/graphknows/.env")
```

## Metrics to collect and report
```json
{
  "namespace": "eval-real-traces-2026-05-23",
  "sessions_processed": 3,
  "total_messages": 127,
  "turns_parsed": 85,
  "tool_calls_summarized": 42,
  "artifacts_detected": 12,
  "nodes_written": 213,
  "errors": [],
  "recall_eval": [
    {
      "query": "...",
      "top_5_hits": ["...", "..."],
      "top_hit_memory_class": "fact"
    }
  ],
  "duration_seconds": 12.4
}
```

## Recall queries to run (hardcoded)
After writing all sessions, run these 3 recall queries:
1. `"graphknows memory adapter implementation"`
2. `"EU AI Act compliance requirements"`
3. `"pytest test failures circular import"`

For each query, report top 5 hits (text[:80]) and top hit's memory_class.

## Error handling
- Skip sessions or messages that fail to parse; record error count in metrics
- If Memgraph or Postgres is unreachable, print clear error and exit 1
- If no sessions found in state.db, print message and exit 0

## Acceptance criteria
- Script runs to completion with `uv run python research/eval_generic_adapter_real_traces.py`
- Prints valid JSON to stdout
- `sessions_processed >= 1`
- `turns_parsed >= 10`
- `errors` list is empty (or contains only non-fatal skips)
- `recall_eval` has 3 entries
- `top_hit_memory_class` is a valid `MemoryClass` value

## End-of-task report required
```
File: research/eval_generic_adapter_real_traces.py
Run output: <paste actual JSON output>
Metrics: sessions_processed=N, turns_parsed=N, nodes_written=N, errors=[]
```

## Do NOT
- Modify any production code
- Write to gitignored paths outside research/
- Commit or push anything
- Use LLMs for summarization (use ToolSummarizer only)

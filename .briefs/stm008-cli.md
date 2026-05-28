# STM-008: CLI — `memory` + `retrieve` + `doctor` + `forget` command groups

## Context
Wiki: /root/graphknows/llm-wiki/architecture/memory-layer-v2.md
Project root: /root/graphknows
Read .hermes/instructions/python-style.instructions.md first.
Read .hermes/instructions/tdd-test-driven-development.instructions.md first.
Model: github-copilot/claude-sonnet-4.6

Depends on: STM-003 (MemoryIngestor), STM-005 (DialogWriter), STM-006 (SessionFlusher),
STM-007 (recall_short_term, recall_long_term)

## Goal
Wiring the CLI is the most important deliverable. Build a deep, composable, scriptable CLI.
All output: human-readable by default, JSON via --json flag. Progress to stderr, results to stdout.

## Read existing cli/main.py and cli/memory.py first
Understand what already exists. Replace fully — no backward compatibility.

## Rewrite services/graphknows/graphknows/cli/main.py

```python
"""graphknows-cli — root CLI entry point."""
import click
from graphknows.cli.memory import memory
from graphknows.cli.retrieve import retrieve
from graphknows.cli.doctor import doctor
from graphknows.cli.forget import forget

@click.group()
@click.option("--json", "output_json", is_flag=True, default=False, help="Output machine-readable JSON")
@click.option("-v", "--verbose", is_flag=True, default=False, help="Verbose logging")
@click.pass_context
def cli(ctx: click.Context, output_json: bool, verbose: bool) -> None:
    """GraphKnows CLI — memory ingestion, retrieval, and session management."""
    ctx.ensure_object(dict)
    ctx.obj["json"] = output_json
    ctx.obj["verbose"] = verbose
    if verbose:
        import logging
        logging.basicConfig(level=logging.DEBUG)

cli.add_command(memory)
cli.add_command(retrieve)
cli.add_command(doctor)
cli.add_command(forget)
```

## Rewrite services/graphknows/graphknows/cli/memory.py

Commands:
```
graphknows-cli memory ingest --text "..." --session-id <id> [--title "..."] [--source-hint "..."]
graphknows-cli memory ingest --file path/to/file --session-id <id>
graphknows-cli memory dialog start --session-id <id> --dialog-id <id> [--participants "user:u1,agent:a1"]
graphknows-cli memory dialog turn --session-id <id> --dialog-id <id> --role user --text "..."
graphknows-cli memory dialog turn ... --tool-calls '[{"name":"search","latency_ms":120}]'
graphknows-cli memory flush --session-id <id> [--no-topics] [--no-clear] [--run-gds] [--confirm]
graphknows-cli memory status --session-id <id>
```

For `memory ingest`:
- Load MemoryIngestor (lazy: import only when command runs, not at module load)
- GLiNER2 must be lazy-loaded (already is via InProcessExtractor._ensure_model_loaded)
- Print stage progress to stderr: [1/3] Parsing... [2/3] Extracting entities... [3/3] Writing STM...
- On success: print IngestResult summary (human or JSON)

For `memory flush`:
- Require --confirm OR --session-id=<id> to confirm they know what they're doing
- If summarize_topics=True (default): warn "This may take 30-60s" to stderr
- Print FlushResult summary

For `memory status`:
```
STM status for session s1:
  Files:    3
  Chunks:   28
  Entities: 147
  Dialogs:  1
  Turns:    24
```

For `memory dialog turn --tool-calls`:
- Accept JSON string of list[{name, latency_ms, args?, result_summary?}]
- Parse and pass as list[ToolCall] to DialogWriter.write_turn()

## Create services/graphknows/graphknows/cli/retrieve.py

Commands:
```
graphknows-cli retrieve query "..." --short-term --session-id <id> [--top-k 5]
graphknows-cli retrieve query "..." --long-term [--session-id <id>] [--top-k 5]
graphknows-cli retrieve explain "..." --short-term --session-id <id>
```

Human output:
```
Query: "Who signed the European Green Deal?"
Scope: short-term | session: s1

  [0.91] ## European Policy / ### Green Deal    (chunk)
         European Green Deal signed by von der Leyen...
         Entities: Ursula von der Leyen, European Commission

  [0.78] Turn #4 [agent]    (turn)
         I found that the EGD was signed in 2019...
```

For `--explain`: show per-source scores before RRF fusion.
Load recall_short_term or recall_long_term based on flag.
--short-term and --long-term are mutually exclusive (use Click's mutually_exclusive if available,
or manual check).

## Create services/graphknows/graphknows/cli/doctor.py

Checks:
1. Memgraph bolt reachability (bolt://172.18.0.1:17687 — from env MEMGRAPH_URI or default)
2. pgvector TCP reachability + schema check (stm_chunks, ltm_chunks tables exist)
3. Neo4j bolt reachability (from env NEO4J_URI or default bolt://172.18.0.1:7687)
4. OpenRouter API key present (OPENROUTER_API_KEY env var — just check presence)
5. graphknows package version

Exit 0 on all pass, 1 on any failure.
Use socket.create_connection for TCP checks (not nc/curl).

## Create services/graphknows/graphknows/cli/forget.py

Commands:
```
graphknows-cli forget session --session-id <id> [--confirm]
graphknows-cli forget session --all --confirm
graphknows-cli forget isolated [--confirm]
graphknows-cli forget ttl [--days 90] [--confirm]
```

For `forget session`:
- Delete from Memgraph: MATCH (n {session_id: $sid}) DETACH DELETE n
- Delete from pgvector: DELETE FROM stm_chunks WHERE session_id=$sid; same for stm_turns
- Print before/after counts

For `forget isolated`: delete ENTITY nodes with no edges (Neo4j LTM)
For `forget ttl`: prune by last_seen timestamp (Neo4j LTM)

## Create services/graphknows/graphknows/cli/output.py

```python
"""Shared output helpers: emit to stdout, progress to stderr, JSON mode."""
import json
import sys
from typing import Any

def emit(data: Any, human_fn: callable, ctx: dict | None = None) -> None:
    """Print result. JSON if ctx['json'], else call human_fn(data)."""
    if ctx and ctx.get("json"):
        print(json.dumps(data, indent=2, default=str))
    else:
        human_fn(data)

def progress(msg: str) -> None:
    """Print progress message to stderr."""
    print(msg, file=sys.stderr)
```

## Update pyproject.toml
In [project.scripts]:
```
graphknows-cli = "graphknows.cli.main:cli"
```
Remove any old entry like `graphknows-memory = ...` if present.

## Write tests/test_cli_memory.py, tests/test_cli_retrieve.py, tests/test_cli_doctor.py, tests/test_cli_forget.py

Use Click's `CliRunner` for all CLI tests. Mock all store constructors.

test_cli_memory.py:
1. `memory ingest --text "hello" --session-id s1` exits 0
2. `memory ingest --file /nonexistent` exits non-zero with error message
3. `memory flush --session-id s1` (without --confirm) exits non-zero (requires confirm)
4. `memory flush --session-id s1 --confirm` exits 0 (mock SessionFlusher)
5. `memory status --session-id s1` prints Files/Chunks/Entities counts
6. `memory dialog turn --role user --text "hi" --dialog-id d1 --session-id s1` exits 0
7. `memory --json ingest --text "hello" --session-id s1` outputs valid JSON

test_cli_retrieve.py:
1. `retrieve query "test" --short-term --session-id s1` exits 0
2. `retrieve query "test" --long-term` exits 0
3. `retrieve query "test"` (no scope) exits non-zero with clear error message
4. `retrieve query "test" --short-term --long-term` exits non-zero (mutually exclusive)
5. `retrieve --json query "test" --short-term --session-id s1` outputs valid JSON list

test_cli_doctor.py:
1. All checks pass → exits 0
2. One check fails → exits 1
3. Uses mocked socket.create_connection (no live network)

test_cli_forget.py:
1. `forget session --session-id s1` without --confirm exits non-zero
2. `forget session --session-id s1 --confirm` exits 0
3. `forget session --all` without --confirm exits non-zero
4. `forget isolated` with mocked Neo4j exits 0

## Acceptance criteria
- ALL CLI tests pass with Click CliRunner (no live stack needed)
- ruff + mypy clean
- check_imports.py clean
- `graphknows-cli --help` shows: memory, retrieve, doctor, forget
- All GLiNER2/store imports lazy (not at module load time)

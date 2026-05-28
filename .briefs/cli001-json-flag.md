# Brief: CLI-001 — --json flag on graphknows-memory commands

## Context
Project root: `/root/graphknows`
Package: `graphknows/` (single package, no `services/` anymore)
CLI file: `graphknows/graphknows/cli/memory.py`
Venv: `/root/graphknows/.venv`
Run tests: `cd /root/graphknows && uv run pytest tests/ -x -q -m "not integration"`

Read BEFORE writing any code:
- `/root/graphknows/.hermes/instructions/python-style.instructions.md`
- `/root/graphknows/.hermes/instructions/tdd-test-driven-development.instructions.md`

## Task
Add a `--json` flag to each command in `graphknows/graphknows/cli/memory.py` so the CLI is machine-readable.

## Commands to update + output format

### `ingest --json`
```json
{"file_id": "8a203254ff021fa1", "chunks": 1, "entities": 1, "elapsed_s": 15.6, "errors": []}
```

### `recall-short --json` and `recall-long --json`
Emit one JSON object per line (JSONL — streamable):
```
{"score": 0.016, "type": "chunk", "text": "We are building...", "session_id": "hermes-test-001"}
{"score": 0.012, "type": "turn", "text": "...", "session_id": "hermes-test-001"}
```
If no results: emit nothing to stdout (zero lines). Do NOT emit `{"results": []}`.

### `stats --json`
```json
{"stm": {"sessions": 1, "nodes": 4, "edges": 2}, "ltm": {"entities": 0, "topics": 0, "chunks": 0}}
```

### `flush --json`
```json
{"nodes": 3, "edges": 2, "chunks": 1, "turns": 0, "topics": 0, "merged": 0, "elapsed_s": 2.1, "errors": []}
```

### `doctor --json`
```json
{"memgraph": true, "neo4j": true, "pgvector": true}
```

### `forget --json`
```json
{"deleted": 5}
```

## Implementation rules

1. Add `--json` as a `@click.option("--json", "as_json", is_flag=True, help="Emit JSON to stdout")` on each command. Use `as_json` as the Python variable name to avoid shadowing the `json` module.

2. When `as_json=True`:
   - Output the JSON/JSONL to stdout via `click.echo(json.dumps(data))`
   - ALL informational text (GLiNER2 model banner, LiteLLM warnings, progress messages) must go to stderr. Use `click.echo("...", err=True)` for any human-readable progress lines.
   - The GLiNER2 model config banner currently prints to stdout. You can suppress it when `as_json` is True by redirecting stdout to stderr around the heavy imports. Simplest approach: catch it via a context manager or just ensure it goes to stderr naturally (check if it already does — the banner may already go to stderr via logging).

3. When `as_json=False`: existing human-readable output unchanged. No regressions.

4. Import `json` at the top of the file (check if already imported).

5. For `recall-short`/`recall-long`: inspect the `RecallResult` object. Extract: `result.score`, `result.text`, `result.session_id`, and determine type from `result.provenance.memory_class` or `result.chunk_type` (check the actual field names in `graphknows/retrieval/recall_stm.py`).

## Tests
Add tests to `tests/test_cli_json_output.py` using Click's `CliRunner`:

```python
from click.testing import CliRunner
from graphknows.cli.memory import cli
import json

def test_doctor_json():
    runner = CliRunner()
    result = runner.invoke(cli, ["doctor", "--json"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert set(data.keys()) == {"memgraph", "neo4j", "pgvector"}
    assert isinstance(data["memgraph"], bool)
```

Mock the store connectivity checks (MemgraphStore, PgVectorStore, Neo4jStore) so tests don't need live infra. Mark any test requiring live infra with `@pytest.mark.integration`.

## Acceptance criteria
1. `graphknows-memory doctor --json` → valid JSON, exit 0
2. `graphknows-memory stats --session foo --json` → valid JSON
3. `graphknows-memory recall-short --session foo --top-k 3 --json "query"` → 0-N JSONL lines
4. `graphknows-memory ingest --session foo --json "some text"` → valid JSON
5. Human-readable output unchanged when `--json` absent
6. `uv run pytest tests/ -x -q -m "not integration"` passes

## Do NOT touch
- `graphknows/mcp/server.py` — out of scope
- Any file outside `graphknows/cli/memory.py` and `tests/test_cli_json_output.py`
- Existing option names or command names

## End-of-task summary required
```
Files changed: <list>
What was modified: <one line per file>
Files intentionally not touched: <list>
Follow-up needed: <list or none>
```

# GAA-001: Core record types — Brief

## Context
Working directory: `/root/graphknows`
Repo layout:
- Production code: `graphknows/` package
- Tests: `tests/`
- Prototypes: `research/` (gitignored)

Read before starting:
```
cat /root/graphknows/.hermes/instructions/python-style.instructions.md
cat /root/graphknows/.hermes/instructions/tdd-test-driven-development.instructions.md
```

## Objective
Create canonical, host-agnostic record types for a generic agent memory adapter.
These types are the contract layer — no Hermes terms, no host-specific fields.

## Files to create
- `graphknows/memory/adapter_types.py` — new module with all types
- `tests/test_adapter_types.py` — tests for construction, invariants, and edge cases

## Files to modify
- `graphknows/memory/__init__.py` — export the new public types

## Types to implement

### `MemoryClass` enum
```python
class MemoryClass(str, Enum):
    raw_evidence = "raw_evidence"
    summary = "summary"
    state = "state"
    episode = "episode"
    fact = "fact"
    other = "other"
```

### `ArtifactRecord`
Represents a file or code artifact ingested into memory.
Fields:
- `artifact_id: str` — caller-assigned, stable identifier
- `path: str` — file path or URL
- `content: str` — full text content
- `mime_type: str` — e.g. "text/x-python", "text/markdown"
- `memory_class: MemoryClass` — defaults to `raw_evidence`
- `metadata: dict[str, Any]` — optional caller-provided metadata (default empty dict)

### `ToolCallRecord`
Represents one tool invocation during an agent turn.
Fields:
- `tool_id: str` — unique call identifier
- `tool_name: str` — e.g. "terminal", "read_file", "web_search"
- `input_summary: str` — human-readable summary of the input (NOT raw stdin/args dump)
- `output_summary: str` — human-readable summary of output (NOT raw stdout)
- `exit_code: int | None` — for shell/process tools; None for non-shell tools
- `error: str | None` — first meaningful error line; None on success
- `memory_class: MemoryClass` — defaults to `raw_evidence`
- `metadata: dict[str, Any]` — optional (default empty dict)

### `TurnRecord`
Represents one complete agent conversation turn.
Fields:
- `turn_id: str` — unique turn identifier
- `session_id: str` — parent session
- `role: str` — "user" / "assistant" / "system" / "tool"
- `content: str` — cleaned text content (no raw tool outputs inline)
- `tool_calls: list[ToolCallRecord]` — tool calls made in this turn (default empty list)
- `memory_class: MemoryClass` — defaults to `raw_evidence`
- `metadata: dict[str, Any]` — optional (default empty dict)

### `ParticipantRecord`
Represents a participant in a dialog.
Fields:
- `participant_id: str`
- `role: str` — e.g. "user", "assistant", "system"
- `display_name: str | None` — optional
- `metadata: dict[str, Any]` — optional (default empty dict)

### `RecallRequest`
Fields:
- `query: str` — free-text query
- `session_id: str | None` — scope to session if provided
- `top_k: int` — default 10
- `prefer_classes: list[MemoryClass]` — retrieval should boost these classes (default: `[state, episode, fact, summary]`)
- `include_raw: bool` — whether to include `raw_evidence` in results (default True)

### `PromoteOptions`
Controls STM→LTM promotion behavior.
Fields:
- `session_id: str`
- `run_gds: bool` — run GDS enrichment after promotion (default True)
- `run_topics: bool` — run Louvain topic summarization (default True)

### `AdapterCapabilities`
Describes what a concrete adapter instance supports (for introspection/testing).
Fields:
- `supports_stm: bool`
- `supports_ltm: bool`
- `supports_dialog_graph: bool`
- `supports_artifact_ingest: bool`
- `supports_recall: bool`
- `supports_promote: bool`

## Implementation rules
- Use `dataclasses.dataclass(frozen=True)` for all record types (immutable value objects)
- Use `MemoryClass` (not raw strings) as the type for `memory_class` fields
- All fields with defaults must come after required fields
- All public classes and fields must have Google-style docstrings
- Type annotations on every field — use `X | None`, not `Optional[X]`
- No Hermes, no LangGraph, no OpenAI SDK imports — pure stdlib + `enum`

## Tests to write (tests/test_adapter_types.py)
- Construct each type with all required fields
- Verify defaults are applied correctly
- Verify `memory_class` is a `MemoryClass` enum value
- Verify frozen dataclass prevents mutation
- Verify `MemoryClass` string values match expected literals
- Edge case: `TurnRecord` with empty `tool_calls` list
- Edge case: `RecallRequest` with and without `session_id`

## Acceptance criteria
- All types are importable from `graphknows.memory.adapter_types`
- All types are re-exported from `graphknows.memory`
- `uv run pytest tests/test_adapter_types.py -v` passes (100% green)
- `ruff format graphknows/ tests/ && ruff check --no-fix graphknows/ tests/` — no errors
- `mypy graphknows/memory/adapter_types.py --strict` — no errors (or minimal justified ignores documented inline)

## End-of-task report required
```
Files changed: <list>
What was modified: <one line per file>
Files not touched: <list>
Test result: <pytest summary line>
Follow-up: <any gaps or issues>
```

## Do NOT
- Touch any existing memory layer files except `__init__.py` for the export
- Add any behavior (no ingestion, no retrieval, no DB calls)
- Add LLM calls, httpx, or any non-stdlib/non-enum dependency
- Commit or push anything

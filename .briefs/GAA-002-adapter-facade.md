# GAA-002: Generic adapter facade — Brief

## Context
Working directory: `/root/graphknows`
GAA-001 is complete. `graphknows/memory/adapter_types.py` and exports in `graphknows/memory/__init__.py` are in place.

Read before starting:
```
cat /root/graphknows/.hermes/instructions/python-style.instructions.md
cat /root/graphknows/.hermes/instructions/tdd-test-driven-development.instructions.md
```

Also read these existing files to understand the interfaces you will wrap:
- `graphknows/memory/ingestor.py` — `MemoryIngestor`
- `graphknows/memory/dialog.py` — `DialogWriter`
- `graphknows/memory/flush.py` — `SessionFlusher`
- `graphknows/retrieval/recall_stm.py` — `recall_short_term`
- `graphknows/retrieval/recall_ltm.py` — `recall_long_term`
- `graphknows/memory/adapter_types.py` — all record types

## Objective
Create a thin, host-agnostic adapter facade that wraps the existing graphknows memory primitives behind a stable, generic API. This is the single entry point a future host adapter (Hermes, LangGraph, etc.) will use.

## Files to create
- `graphknows/memory/adapter.py` — the `GraphknowsAdapter` class
- `tests/test_memory_adapter.py` — unit tests (mock stores/components)

## Files to modify
- `graphknows/memory/__init__.py` — export `GraphknowsAdapter`

## `GraphknowsAdapter` class design

```python
@dataclass
class GraphknowsAdapter:
    """Host-agnostic facade over the graphknows memory backend.

    Composes MemoryIngestor, DialogWriter, SessionFlusher, and recall
    functions behind a stable generic contract. Host-specific parsing
    and normalization stay outside this class.

    Args:
        mg_store: Memgraph store instance.
        pg_store: PgVector store instance.
        neo_store: Neo4j store instance.
        extractor: Entity extractor (GLiNER2EntityExtractor or compatible).
        embed_fn: Optional embedding function (text -> list[float]).
    """
```

### Methods (all public)

#### `open_session(session_id: str, metadata: dict[str, Any] | None = None) -> None`
Record that a session has started. Calls any setup needed on stores. No-op if stores don't require it. Store `session_id` internally as `self._active_session_id`.

#### `start_dialog(session_id: str, dialog_id: str, participants: list[ParticipantRecord]) -> None`
Delegate to `DialogWriter.start_dialog(session_id, dialog_id, participants=[Participant(id=p.participant_id, role=p.role, name=p.display_name) for p in participants])`.

#### `write_turn(turn: TurnRecord) -> None`
Delegate to `DialogWriter.write_turn(turn.session_id, dialog_id, Turn(role=turn.role, content=turn.content, ...))`.
Store the current `dialog_id` as `self._active_dialog_id` (set by `start_dialog`).
Map `turn.tool_calls` to `ToolCall` objects for `DialogWriter`.

#### `ingest_artifact(artifact: ArtifactRecord, session_id: str) -> None`
Delegate to `MemoryIngestor.ingest_text(artifact.content, session_id=session_id, title=artifact.path, source_hint=artifact.mime_type, metadata={**artifact.metadata, "memory_class": artifact.memory_class.value, "artifact_id": artifact.artifact_id})`.

#### `recall(request: RecallRequest) -> list[dict[str, Any]]`
Call `recall_short_term(mg_store, pg_store, query=request.query, session_id=request.session_id, top_k=request.top_k, embed_fn=embed_fn)`.
Return the hits as a list of dicts (convert `RecallResult` fields to dict).
If `not request.include_raw`, filter out hits where `metadata.get("memory_class") == "raw_evidence"`.

#### `promote(options: PromoteOptions) -> dict[str, Any]`
Instantiate `SessionFlusher(session_id=options.session_id, ...)` and call `await flusher.flush(options.session_id)`.
Return a summary dict with `session_id`, `run_gds`, `run_topics`, and any flush result fields available.
Note: This method must be `async` because `SessionFlusher.flush()` is async.

#### `capabilities(self) -> AdapterCapabilities`
Return `AdapterCapabilities(supports_stm=True, supports_ltm=True, supports_dialog_graph=True, supports_artifact_ingest=True, supports_recall=True, supports_promote=True)`.

### Internal state (set by lifecycle methods)
- `_active_session_id: str | None = None`
- `_active_dialog_id: str | None = None`
- `_ingestor: MemoryIngestor` — created in `__post_init__`
- `_dialog_writer: DialogWriter` — created in `__post_init__`

## Implementation rules
- `GraphknowsAdapter` is a `@dataclass` (not frozen — it tracks session state)
- Orchestration only — no new extraction, no new Cypher, no new SQL
- No host-specific parsing inside the adapter
- Type annotations on all public methods
- Google-style docstrings on the class and all public methods
- Use `X | None` not `Optional[X]`
- All imports must be at top of file

## Tests to write (tests/test_memory_adapter.py)
Mock `MemoryIngestor`, `DialogWriter`, `SessionFlusher`, `recall_short_term`, and `recall_long_term` at the point of use (patch at `graphknows.memory.adapter.*`).

Required test cases:
- `test_open_session_sets_active_session_id` — verify internal state
- `test_start_dialog_delegates_to_dialog_writer` — assert `DialogWriter.start_dialog` called with correct args
- `test_write_turn_delegates_to_dialog_writer` — assert `DialogWriter.write_turn` called; verify `TurnRecord` mapped correctly
- `test_ingest_artifact_delegates_to_ingestor` — assert `MemoryIngestor.ingest_text` called; verify metadata includes `memory_class`
- `test_recall_returns_hits_as_dicts` — mock `recall_short_term` return, assert output is list of dicts
- `test_recall_filters_raw_evidence_when_include_raw_false` — assert raw_evidence hit filtered out
- `test_capabilities_returns_all_true` — spot check
- `test_promote_calls_session_flusher` (async test) — mock `SessionFlusher.flush`, assert it was called

## Acceptance criteria
- `GraphknowsAdapter` is importable from `graphknows.memory`
- `uv run pytest tests/test_memory_adapter.py -v` — all tests pass
- `ruff format graphknows/ tests/ && ruff check --no-fix graphknows/ tests/` — no new errors in new files (pre-existing errors in other files are acceptable)
- Does not modify any existing file other than `graphknows/memory/__init__.py`

## End-of-task report required
```
Files changed: <list>
What was modified: <one line per file>
Files not touched: <list>
Test result: <pytest summary line>
Follow-up: <any gaps>
```

## Do NOT
- Add DB calls, Cypher, SQL, or extraction logic
- Modify any existing memory layer files except `__init__.py`
- Commit or push anything

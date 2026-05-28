# GAA-005: Conformance tests + synthetic host fixtures — Brief

## Context
Working directory: `/root/graphknows`
GAA-001 through GAA-004 are complete. The following new modules exist:
- `graphknows/memory/adapter_types.py` — `TurnRecord`, `ArtifactRecord`, `ToolCallRecord`, etc.
- `graphknows/memory/adapter.py` — `GraphknowsAdapter`
- `graphknows/memory/tool_summarizer.py` — `ToolSummarizer`, `summarize_tool_call`

Read before starting:
```
cat /root/graphknows/.hermes/instructions/python-style.instructions.md
cat /root/graphknows/.hermes/instructions/tdd-test-driven-development.instructions.md
cat /root/graphknows/graphknows/memory/adapter_types.py
cat /root/graphknows/graphknows/memory/adapter.py
```

## Objective
Write a conformance test suite that validates the full adapter contract using a
synthetic host fixture. These tests prove that any host adapter (Hermes, LangGraph, etc.)
that feeds data through `GraphknowsAdapter` in the correct order produces the correct
side effects on the mocked memory backend.

This is an integration-style test — it exercises multiple adapter methods in sequence,
but all backend stores are mocked (no live DB required).

## Files to create
- `tests/fixtures/synthetic_host_session.py` — shared fixture factory
- `tests/test_adapter_conformance.py` — conformance test suite

## `synthetic_host_session.py` fixture design

```python
"""
Shared fixture factory for synthetic host adapter conformance tests.

A synthetic host session is a minimal, deterministic sequence of events that
any real host adapter (Hermes, LangGraph, OpenAI) could produce.
"""

from graphknows.memory.adapter_types import (
    ArtifactRecord,
    MemoryClass,
    ParticipantRecord,
    PromoteOptions,
    RecallRequest,
    ToolCallRecord,
    TurnRecord,
)

SESSION_ID = "test-session-001"
DIALOG_ID = "test-dialog-001"

PARTICIPANTS = [
    ParticipantRecord(
        participant_id="user-1",
        role="user",
        display_name="Alice",
    ),
    ParticipantRecord(
        participant_id="agent-1",
        role="assistant",
        display_name="Agent",
    ),
]

TURNS = [
    TurnRecord(
        session_id=SESSION_ID,
        dialog_id=DIALOG_ID,
        turn_index=0,
        role="user",
        content="What are the EU AI Act compliance requirements?",
        tool_calls=[],
        memory_class=MemoryClass.raw_evidence,
    ),
    TurnRecord(
        session_id=SESSION_ID,
        dialog_id=DIALOG_ID,
        turn_index=1,
        role="assistant",
        content="The EU AI Act requires risk classification...",
        tool_calls=[
            ToolCallRecord(
                tool_name="web_search",
                input_summary="EU AI Act compliance 2025",
                output_summary="Found 10 results about risk categories",
                exit_code=None,
                error=None,
                memory_class=MemoryClass.raw_evidence,
            )
        ],
        memory_class=MemoryClass.raw_evidence,
    ),
]

ARTIFACTS = [
    ArtifactRecord(
        artifact_id="file-001",
        path="/root/graphknows/graphknows/memory/adapter.py",
        content="# adapter module source...\n",
        mime_type="text/x-python",
        memory_class=MemoryClass.raw_evidence,
        metadata={"lang": "python"},
    ),
]

RECALL_REQUEST = RecallRequest(
    query="EU AI Act compliance",
    session_id=SESSION_ID,
    top_k=5,
    include_raw=False,
)

PROMOTE_OPTIONS = PromoteOptions(
    session_id=SESSION_ID,
    run_gds=False,
    run_topics=False,
)
```

## Conformance tests to write (tests/test_adapter_conformance.py)

All stores/components mocked. No live DB. Each test simulates a complete or partial
host adapter lifecycle in order.

### `TestAdapterLifecycle`
All tests in this class call methods in the correct host lifecycle order.

#### `test_full_session_write_sequence`
Simulate a complete session:
1. `adapter.open_session(SESSION_ID)` — assert `_active_session_id == SESSION_ID`
2. `adapter.start_dialog(SESSION_ID, DIALOG_ID, PARTICIPANTS)` — assert DialogWriter.start_dialog called
3. For each turn in TURNS: `adapter.write_turn(turn)` — assert DialogWriter.write_turn called N times
4. For each artifact in ARTIFACTS: `adapter.ingest_artifact(artifact, SESSION_ID)` — assert MemoryIngestor.ingest_text called N times

#### `test_recall_after_write`
1. Open session, start dialog, write turns, ingest artifacts (same as above)
2. Mock `recall_short_term` to return one `fact` hit
3. Call `adapter.recall(RECALL_REQUEST)`
4. Assert result is list of dicts with `text` and `source_type` fields

#### `test_promote_after_write` (async)
1. Open session, write one turn
2. Mock `SessionFlusher.flush` to return a mock result
3. Call `await adapter.promote(PROMOTE_OPTIONS)`
4. Assert flush was called with correct session_id

#### `test_capabilities_contract`
Assert `adapter.capabilities()` returns an `AdapterCapabilities` where ALL boolean fields are True.

#### `test_turn_with_tool_calls_maps_correctly`
Write a turn that has tool_calls. Assert that DialogWriter.write_turn receives a Turn object
whose `tool_calls` list is non-empty and maps `tool_name` correctly.

#### `test_artifact_metadata_includes_memory_class`
Ingest an artifact. Inspect the kwargs passed to `MemoryIngestor.ingest_text`.
Assert `metadata["memory_class"] == "raw_evidence"`.

#### `test_open_session_idempotent`
Call `open_session` twice with the same session_id. Assert `_active_session_id` is still set correctly.
No exception raised.

#### `test_recall_include_raw_false_filters`
Mock `recall_short_term` to return two hits:
- hit A: `metadata={"memory_class": "raw_evidence"}`
- hit B: `metadata={"memory_class": "fact"}`
Call `adapter.recall(RecallRequest(..., include_raw=False))`.
Assert only hit B is in the returned list.

## Mocking strategy
Use the same pattern as `tests/test_memory_adapter.py`:
```python
@patch("graphknows.memory.adapter.SessionFlusher")
@patch("graphknows.memory.adapter.MemoryIngestor")
@patch("graphknows.memory.adapter.DialogWriter")
```
For recall tests, patch `graphknows.retrieval.recall_stm.recall_short_term`.

## Acceptance criteria
- `uv run pytest tests/test_adapter_conformance.py -v` — all 8 tests pass
- `ruff format tests/test_adapter_conformance.py tests/fixtures/synthetic_host_session.py && ruff check --no-fix tests/test_adapter_conformance.py tests/fixtures/synthetic_host_session.py` — no new errors
- No live DB connections anywhere in the test file

## End-of-task report required
```
Files changed: <list>
What was modified: <one line per file>
Files not touched: <list>
Test result: <pytest summary line>
Follow-up: <any gaps>
```

## Do NOT
- Touch any production code files
- Add live DB connections
- Commit or push anything

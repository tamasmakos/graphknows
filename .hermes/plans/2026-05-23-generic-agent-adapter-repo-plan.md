# Generic Agent Adapter — Repo-Level Implementation Plan

> For Hermes: execute this via small Kanban-backed opencode changesets. Do not build any host-specific adapter in this wave.

Goal
- Add a first-class, host-agnostic memory adapter surface to Graphknows so other agent runtimes can use the same backend without importing Hermes concepts.

Architecture
- Keep host-specific parsing/normalization outside Graphknows.
- Add canonical repo-level record types and one adapter facade that wraps existing Graphknows primitives:
  - `MemoryIngestor`
  - `DialogWriter`
  - `SessionFlusher`
  - `recall_short_term` / `recall_long_term`
- Keep raw evidence and distilled memory in the same storage for v1, but make retrieval metadata-aware so continuity queries prefer `state` / `episode` / `fact` over `raw_evidence`.
- Terminal/tool output must be summarized into structured evidence records before memory promotion.

External sequencing guidance
- Perplexity-backed ordering matched the repo discoveries:
  1. canonical record/event types
  2. unified adapter API
  3. basic ingestion/storage
  4. terminal/tool summarization
  5. metadata-aware retrieval preference
  6. evaluation on real traces
- For v1, use metadata-aware retrieval rather than fully separate lanes.

Non-goals for this wave
- No Hermes plugin work
- No LangGraph/OpenAI/MCP adapter implementation
- No new graph backend
- No LLM-based summarizer in the critical path for v1 terminal/tool summaries

---

## Implementation principles

1. Repo-level only
- No Hermes hook names in production code
- No host-native callback objects in Graphknows types

2. Safe v1 storage strategy
- Reuse current STM/LTM stores
- Reuse chunk metadata for `memory_class`
- Avoid schema churn unless a test shows it is required

3. Retrieval before host integrations
- No host adapter should be implemented until generic retrieval returns distilled memory above raw evidence on real trace evaluation

4. Opencode execution mode
- One Kanban card per change set
- Each card includes unit tests in the same patch
- Real-trace evaluation stays in `research/`

---

## Proposed Kanban wave

### GAA-001 — Core record types
Objective
- Create canonical repo-level types for generic memory operations.

Files
- Create: `graphknows/memory/adapter_types.py`
- Modify: `graphknows/memory/__init__.py`
- Test: `tests/test_adapter_types.py`

Acceptance criteria
- Define canonical types with no Hermes terms:
  - `ArtifactRecord`
  - `ToolCallRecord`
  - `TurnRecord`
  - `ParticipantRecord`
  - `RecallRequest`
  - `PromoteOptions`
  - `AdapterCapabilities`
- `memory_class` is first-class and supports:
  - `raw_evidence`
  - `summary`
  - `state`
  - `episode`
  - `fact`
  - `other`
- Public docstrings explain intended use and what is deliberately host-agnostic.
- Tests cover construction and invariants.

Suggested opencode brief
- Implement only the types and exports.
- No behavior, no retrieval changes, no host adapters.

---

### GAA-002 — Generic adapter facade
Objective
- Add one repo-level adapter class that wraps existing Graphknows components behind the generic contract.

Files
- Create: `graphknows/memory/adapter.py`
- Modify: `graphknows/memory/__init__.py`
- Test: `tests/test_memory_adapter.py`

Acceptance criteria
- Expose methods equivalent to:
  - `open_session`
  - `start_dialog`
  - `write_turn`
  - `ingest_artifact`
  - `recall`
  - `promote`
  - `close`
- Adapter composes existing repo primitives instead of duplicating logic.
- No host-specific parsing inside the adapter.
- Tests mock stores/components and verify correct delegation paths.

Suggested opencode brief
- Keep orchestration thin.
- Treat this as a stable facade layer, not a new backend.

Dependencies
- Depends on GAA-001.

---

### GAA-003 — Deterministic tool/terminal summarizer
Objective
- Add repo-level summarization helpers for tool and terminal execution records.

Files
- Create: `graphknows/memory/tool_summarizer.py`
- Test: `tests/test_tool_summarizer.py`

Acceptance criteria
- Deterministically summarize tool/terminal execution into structured evidence text + metadata.
- Cover fields such as:
  - tool name
  - command/tool class
  - outcome
  - error class
  - artifact paths touched
  - first meaningful signal
  - durable-fact-candidate flag
- No raw stdout passthrough as the default summary.
- Unit tests cover:
  - success
  - non-zero exit
  - traceback/error text
  - quality-tool commands like pytest/mypy/ruff

Suggested opencode brief
- Do not add LLM calls.
- This is a deterministic v1 summarizer only.

Dependencies
- Depends on GAA-001.

---

### GAA-004 — Metadata-aware recall preference
Objective
- Make STM/LTM retrieval prefer distilled memory over raw evidence by default.

Files
- Modify: `graphknows/retrieval/recall_stm.py`
- Modify: `graphknows/retrieval/recall_ltm.py`
- Possibly modify: `graphknows/cli/memory.py`
- Test: `tests/test_recall_memory_class_priority.py`

Acceptance criteria
- Retrieval can inspect `memory_class` from chunk metadata.
- Default continuity-style retrieval boosts:
  - `state`
  - `episode`
  - `fact`
  - `summary`
- Default continuity-style retrieval de-boosts:
  - `raw_evidence`
  - generic `turn` hits unless needed as fallback
- Debug/terminal-oriented retrieval can still surface `raw_evidence` when appropriate.
- Tests prove that a distilled item outranks a terminal-evidence item for a continuity-style query.

Suggested opencode brief
- Prefer small score-shaping/filtering changes over a new lane architecture.
- No separate indices in v1.

Dependencies
- Depends on GAA-001 and GAA-003.

---

### GAA-005 — Conformance tests and synthetic host fixtures
Objective
- Prove the generic adapter contract works for multiple host shapes without adding real host integrations.

Files
- Create: `tests/test_memory_adapter_conformance.py`
- Create: `tests/fixtures/generic_agent_trace_a.json`
- Create: `tests/fixtures/generic_agent_trace_b.json`

Acceptance criteria
- At least two different synthetic host-shaped inputs are normalized outside the adapter and then passed through the same adapter API.
- Tests verify:
  - session open
  - dialog start
  - turn write
  - artifact ingest
  - recall call shape
  - promotion call shape
- No Hermes-specific fields are required by the adapter.

Suggested opencode brief
- Keep fixtures small and explicit.
- This is a contract test, not a performance test.

Dependencies
- Depends on GAA-002, GAA-003, and GAA-004.

---

### GAA-006 — Real-trace evaluation harness
Objective
- Validate the generic adapter on real traces before any host adapter work begins.

Files
- Create: `research/eval_generic_adapter_real_traces.py`
- Create: `research/eval_generic_adapter_real_traces_results.json`
- Optional test wrapper: `tests/test_eval_contract_smoke.py` (small, non-heavy smoke only)

Acceptance criteria
- Replays at least one real session-derived generic event stream.
- Runs STM recall and LTM recall after promotion.
- Emits machine-readable results.
- Demonstrates that continuity queries rank distilled memory above raw terminal evidence after GAA-004.
- Documents any remaining failure cases clearly.

Suggested opencode brief
- Use `research/`, not production code, for the heavy evaluation harness.
- Real traces can be local-only and non-committed if necessary, but the script shape should be committed if it does not embed secrets or local assumptions.

Dependencies
- Depends on GAA-002, GAA-003, GAA-004, and GAA-005.

---

### GAA-007 — Docs and developer surface
Objective
- Make the repo-level adapter understandable and usable by future host adapters.

Files
- Update: `README.md` or a focused memory section if appropriate
- Update/create: `llm-wiki/architecture/generic-agent-adapter-discovery.md` (implementation follow-up section)
- Optional create: `graphknows/memory/README.md`

Acceptance criteria
- Repo docs explain:
  - what the adapter is
  - what stays outside it
  - how `memory_class` affects retrieval
  - why v1 uses metadata-aware retrieval instead of separate lanes
- Docs explicitly say host adapters are out of scope for this wave.

Dependencies
- Depends on GAA-006.

---

## Recommended dependency graph

- GAA-001 → GAA-002
- GAA-001 → GAA-003
- GAA-003 → GAA-004
- GAA-002 + GAA-003 + GAA-004 → GAA-005
- GAA-002 + GAA-003 + GAA-004 + GAA-005 → GAA-006
- GAA-006 → GAA-007

---

## Acceptance gate before any host-specific adapter work

Do not start Hermes/LangGraph/OpenAI/MCP adapter implementation until all of the following are true:
- Canonical generic types exist and are tested
- Adapter facade exists and is tested
- Deterministic tool/terminal summarization exists and is tested
- Metadata-aware retrieval surfaces distilled memory above raw evidence on continuity queries
- Real-trace evaluation harness passes and emits a report

---

## Notes for Kanban / opencode execution

- Preferred assignee lane: `backend-eng`
- If `backend-eng` is not available on disk, keep the cards as the ground truth anyway and dispatch manually via opencode using the card body.
- Keep each PR/change set single-purpose and test-backed.
- No opportunistic refactors.

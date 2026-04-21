# TASK014 — Proper AgentWorkflow + Full SSE Taxonomy

**Status:** Pending  
**Added:** 2026-04-21  
**Updated:** 2026-04-21  
**Phase:** C — Agent Intelligence (requires TASK010 + TASK011 complete)

---

## Original Request
The current `agent/workflow.py` is a naked `ReActAgent.achat(query)` call with no decomposition, no reflection loop, no sufficiency check, and no structured SSE events beyond `token|citation|graph|done`. The frontend can't show reasoning steps, tool calls, or subquestion breakdowns because the server never emits them.

---

## Thought Process
LlamaIndex's `ReActAgent` is already capable of multi-step reasoning — the problem is we're not exposing the reasoning events over SSE, and we have no explicit "decompose query → iterate → synthesize" lifecycle that we can instrument.

The target architecture is a 4-phase loop:
1. **Decompose** — LLM breaks the query into N sub-questions (structured output: `list[SubQuestion]`)
2. **Iterate** — ReActAgent answers each sub-question (emitting `reasoning`, `tool_call`, `tool_result` events per step)
3. **Rerank** — retrieved chunks from all tool calls are re-ranked by relevance (TASK015)
4. **Synthesize** — LLM synthesizes final answer from ranked chunks, emitting `token` events

The key insight is that decompose + synthesize are separate LLM calls (cheap, structured), while the iterate phase uses the full ReActAgent loop per sub-question. This gives the frontend a clear structure to render as a Timeline.

### New Pydantic Models Needed
- `SubQuestion(BaseModel)`: `id: str`, `text: str`, `answer: str | None`
- `ToolCall(BaseModel)`: `tool: str`, `input: str`, `output: str`, `duration_ms: int`
- `ReasoningStep(BaseModel)`: `step: int`, `thought: str`, `action: str | None`, `observation: str | None`
- Enrich `AgentResponse`: add `sub_questions: list[SubQuestion]`, `reasoning_steps: list[ReasoningStep]`, `tool_calls: list[ToolCall]`, `verification: VerificationResult | None`

### SSE Event Taxonomy (full)
```
{"type": "reasoning",    "content": "Breaking query into sub-questions..."}
{"type": "tool_call",    "tool": "search_chunks", "input": "...", "id": "tc_001"}
{"type": "tool_result",  "id": "tc_001", "output": "...", "duration_ms": 142}
{"type": "token",        "content": " "}           ← streaming synthesis tokens
{"type": "citation",     "chunk_id": "...", "text": "..."}
{"type": "subgraph",     "nodes": [...], "edges": [...]}
{"type": "done",         "answer": "...", "citations": [...]}
{"type": "error",        "message": "...", "recoverable": false}
```

---

## Implementation Plan

1. **New `services/graphrag/src/models/` additions** (in `models/__init__.py`):
   - Add `SubQuestion`, `ToolCall`, `ReasoningStep`, `VerificationResult`, `AgentResponse` (enriched)

2. **Refactor `agent/workflow.py`**:
   - Replace `run_agent()` with `async def run_query(query, driver, database, messages) -> AgentResponse`
   - Internal phases: `_decompose()`, `_iterate()`, `_synthesize()`

3. **Refactor `agent/workflow.py` streaming variant**:
   - `async def stream_query(query, driver, database, messages) -> AsyncGenerator[dict, None]`
   - Yields SSE dicts from all 8 event types above

4. **Instrument with Langfuse** — per-phase spans:
   ```python
   with langfuse.span(name="decompose", input=query):
       sub_questions = await _decompose(query)
   ```

5. **Update `main.py` SSE endpoint** — map all 8 event types to SSE `data:` lines; set correct `Content-Type: text/event-stream`.

6. **Update `mcp/server.py` `kg_chat` tool** — call `run_query()` (non-streaming), return full `AgentResponse` as JSON.

7. **Write unit tests**:
   - `test_decompose_returns_subquestions()`
   - `test_stream_emits_all_event_types()`
   - `test_agent_handles_empty_results_gracefully()`

---

## Progress Tracking

**Overall Status:** Not Started — 0%

### Subtasks
| ID | Description | Status | Updated | Notes |
|----|-------------|--------|---------|-------|
| 14.1 | Add SubQuestion, ToolCall, ReasoningStep models | Not Started | 2026-04-21 | models/__init__.py |
| 14.2 | Enrich AgentResponse with new fields | Not Started | 2026-04-21 | |
| 14.3 | Implement _decompose() phase | Not Started | 2026-04-21 | Structured LLM output |
| 14.4 | Implement _iterate() phase with event emission | Not Started | 2026-04-21 | Per sub-question |
| 14.5 | Implement _synthesize() phase | Not Started | 2026-04-21 | Streaming tokens |
| 14.6 | Implement full SSE event taxonomy (8 types) | Not Started | 2026-04-21 | |
| 14.7 | Add Langfuse per-phase spans | Not Started | 2026-04-21 | |
| 14.8 | Update MCP kg_chat to use run_query() | Not Started | 2026-04-21 | |
| 14.9 | Write unit tests | Not Started | 2026-04-21 | Mock LLM + Neo4j |

## Progress Log
### 2026-04-21
- Task created. Audit confirmed: no decompose, no sufficiency loop, only 4 SSE types emitted.

# TASK007 — Phase 4: AgentWorkflow Rewrite (graphrag)

**Status:** Pending  
**Added:** 2026-04-20  
**Updated:** 2026-04-20

## Original Request
Replace the linear LlamaIndex `Workflow` retrieval chain with a true iterative `AgentWorkflow` featuring 4 tools, query decomposition, reflection, sufficiency check, structured citations, SSE streaming, and per-tool Langfuse spans.

## Thought Process
The current implementation is a linear 4-step chain (keywords → seeds → expand → synthesize). It's not agentic: it can't select which retrieval to do next, can't iterate, and can't recognize when it has enough context.

The new implementation uses LlamaIndex `AgentWorkflow` with genuine tool-use iteration. The key differentiator is the reflect step after each tool call — a cheap fast-model call checks "did we answer all sub-questions?" before selecting the next tool. This prevents runaway tool loops and wasted tokens.

SSE is implemented via `sse-starlette` on the FastAPI side. Each agent event (reasoning, tool_call, token, citation, done) is emitted as a separate SSE event. The Next.js Route Handler proxies these as a `ReadableStream`.

## Interface Contracts

```python
class Citation(BaseModel):
    chunk_id: str
    doc_id: str
    doc_title: str
    heading_path: list[str]
    text_excerpt: str    # <= 300 chars
    score: float

class ToolCall(BaseModel):
    tool_name: str; input: dict; output_summary: str
    duration_ms: int; span_id: str; iteration: int

class ReasoningStep(BaseModel):
    step_type: Literal["decomposition","tool_selection","reflection","synthesis"]
    thought: str; sub_question: str | None; iteration: int

class AgentResponse(BaseModel):
    answer: str
    citations: list[Citation]
    retrieved_subgraph: GraphData
    reasoning_steps: list[ReasoningStep]
    tool_calls: list[ToolCall]
    sub_questions: list[str]
    iteration_count: int
    total_duration_ms: int
```

## The 4 Tools
1. `SearchChunksTool` — hybrid vector (pgvector cosine) + BM25 keyword on CHUNK nodes. Inputs: `query: str, top_k: int = 10`.
2. `GetEntityNeighborsTool` — Cypher subgraph expansion from entity. Inputs: `entity_id: str, depth: int = 1, max_nodes: int = 50`.
3. `GetDocumentContextTool` — fetch full DOCUMENT metadata + all its CHUNKs. Inputs: `doc_id: str, include_chunks: bool = True`.
4. `SearchEntitiesTool` — fuzzy entity name lookup (pgvector similarity + exact string). Inputs: `name: str, top_k: int = 10`.

## Langfuse Span Structure
```
Trace (trace_id = correlation header)
├── decompose_query  [model, tokens]
├── iteration_1/select_tool  [model, tool chosen]
├── iteration_1/tool.search_chunks  [query, top_k, result count, ms]
├── iteration_1/reflect  [model, open questions]
├── iteration_2/...
└── synthesize  [model, tokens, citations count]
```

## Implementation Plan
- [ ] Create `services/graphrag/src/agent/workflow.py` — AgentWorkflow class
- [ ] Create `services/graphrag/src/agent/tools.py` — 4 AgentTool subclasses
- [ ] Create `services/graphrag/src/agent/decomposer.py` — query decomposition
- [ ] Create `services/graphrag/src/agent/context.py` — ContextAccumulator with dedup
- [ ] Create `services/graphrag/src/agent/sufficiency.py` — stop condition check
- [ ] Create `services/graphrag/src/agent/prompts.py` — system/decompose/synthesize prompts
- [ ] Create `services/graphrag/src/models/response.py` — AgentResponse, Citation, ToolCall, ReasoningStep
- [ ] Create `services/graphrag/src/models/request.py` — ChatRequest
- [ ] Create `services/graphrag/src/models/graph.py` — GraphData, Node, Edge
- [ ] Create `services/graphrag/src/infrastructure/neo4j_client.py` — async singleton
- [ ] Create `services/graphrag/src/observability/langfuse.py` — @traced + span context manager
- [ ] Update `services/graphrag/src/main.py` — new endpoints with SSE + lifespan
- [ ] Update `services/graphrag/src/mcp/server.py` — wrap new workflow
- [ ] Update `services/graphrag/src/common/config/settings.py` — neo4j vars + agent settings
- [ ] Update `services/graphrag/pyproject.toml` — add neo4j, rank-bm25, sse-starlette; remove falkordb
- [ ] Delete: workflow/graph_workflow.py, services/graph_retriever.py, services/context_builder.py, infrastructure/graph_db.py

## Progress Tracking

**Overall Status:** Not Started — 0%

### Subtasks
| ID | Description | Status | Updated | Notes |
|----|-------------|--------|---------|-------|
| 7.1 | AgentResponse + Citation + ToolCall models | Not Started | 2026-04-20 | |
| 7.2 | AgentTool base + 4 concrete tools | Not Started | 2026-04-20 | |
| 7.3 | ContextAccumulator (dedup) | Not Started | 2026-04-20 | |
| 7.4 | Decomposer + Sufficiency check | Not Started | 2026-04-20 | |
| 7.5 | AgentWorkflow (loop) | Not Started | 2026-04-20 | |
| 7.6 | Neo4j client (async singleton) | Not Started | 2026-04-20 | |
| 7.7 | Langfuse @traced + span manager | Not Started | 2026-04-20 | |
| 7.8 | FastAPI SSE endpoint | Not Started | 2026-04-20 | sse-starlette |
| 7.9 | MCP server update | Not Started | 2026-04-20 | |
| 7.10 | Settings + pyproject.toml | Not Started | 2026-04-20 | |
| 7.11 | Delete old workflow/services | Not Started | 2026-04-20 | |

## Progress Log
### 2026-04-20
- Task created during planning session.

---
title: LangGraph Agent Architecture — Live Search + Session Graph
created: 2026-05-08
updated: 2026-05-08
type: concept
tags: [agent, langgraph, architecture, graphgen, streaming, research]
sources: []
confidence: high
---

# LangGraph Agent Architecture (v3)

## Decision

Replace the hand-rolled `IterativeOrchestrator` ReAct loop with a **LangGraph `StateGraph`** agent.

**Rationale (2026-05-08):**
- Custom loop couldn't stream thought tokens live — only full step strings after tool execution
- Tool list pollution: GraphRAG tools (`retrieve`, `get_chunks`, `get_leiden`) caused infinite loops
- No separation between web search, EU archive search, graphgen, and graphrag reasoning
- `astream_events(version="v2")` in LangGraph gives granular per-token streaming natively

**Framework:** LangGraph 0.2.x + `langchain-groq` (`ChatGroq`)

---

## Graph Structure

```
START
  │
  ▼
[research_node]  ← LLM with bound tools: search_web, search_eu, graphgen, graphrag
  │
  ├─► [tool_node] → back to research_node
  │
  └─► END (when LLM produces no tool call = answer message)
```

- `max_iterations = 6` enforced via recursion limit
- Conditional edge: if last message has `tool_calls` → go to tool_node; else → END

---

## Tool Architecture

| Tool | Purpose | Backend | Side-effect |
|---|---|---|---|
| `search_web` | Tavily live web + news | `tavily_tool.py` | triggers graphgen background |
| `search_eu` | EU official archives (EP, EUR-Lex, Council, EC, ECB, ECA, EPRS) | `source_router.py` adapters | triggers graphgen background |
| `graphgen` | Build session KG from collected docs | `graphgen_worker.py` subprocess | WS push `graph_update` immediately |
| `graphrag` | Query accumulated session KG + Neo4j | `session_graph_manager` + Neo4j bolt | returns entity/community JSON |

---

## WS Event Schema (v3)

All events go over the existing `/ws/{session_id}` endpoint.

```jsonc
{"type": "thought_token", "token": "I need to...", "step": 1}
{"type": "tool_start",    "tool": "search_eu", "input": {"query": "..."}, "step": 1}
{"type": "tool_result",   "tool": "search_eu", "summary": "8 docs", "docs": [...], "step": 1}
{"type": "answer_token",  "token": "## European...", "step": 5}
{"type": "graph_update",  "graph": {"nodes": [...], "edges": [...]}, "lexical": {"nodes": [...], "edges": [...]}}
{"type": "answer",        "answer": "...", "sources": [...], "session_id": "..."}
```

---

## Report Format: EPRS-style

```
# [Title]
**European Parliamentary Research Service** | Session [ID] | [Date]

## Executive Summary
## Background
## Key Findings (numbered, with inline [Doc-N] citations)
## Legislative Record (CELEX/ELI numbers)
## Context & Analysis
## Knowledge Graph Insights (from graphrag tool)
## Key Actors (table)
## Key Legislation (table)
## Sources ([🏛️ Doc-N] and [🌐 Doc-N])
```

---

## Frontend Changes (v3)

### Reasoning trace panel
- Replaces the current collapsed `🔧 step` accordion
- Each step card: tool icon + name + streaming thought tokens + result chips
- Steps stay expanded and scrollable during research

### Answer panel
- Streams `answer_token` chunks in real time (not after full completion)
- `marked.parse()` applied incrementally per paragraph

### Graph tabs (dual)
- **Knowledge Graph** — entity-relation D3 force (PERSON, ORG, LAW nodes)
- **Lexical Graph** — doc-term co-occurrence D3 force
- Both populate during research, not only at answer time
- Tab badge shows live count: `Knowledge Graph (47)`

---

## Files

```
services/agent/
  langgraph_agent.py        NEW — StateGraph, tool nodes, astream_events mapper
  orchestrator.py           DELETED
  router.py                 UPDATED — swap to langgraph_agent, new WS event mapping
  tools/
    search_web.py           NEW — Tavily-only @tool
    search_eu.py            NEW — EU archives @tool
    graphgen_tool.py        REWRITE — @tool wrapping graphgen_worker subprocess
    graphrag_tool.py        NEW — @tool querying session KG + Neo4j

apps/web/index.html         UPDATED — live streaming trace, dual graph tabs
```

---

## Dependencies

```bash
pip install langgraph langchain-groq langchain-core
```

---

## Links
- [[product-architecture-v2]]
- [[service-design-agent]]
- [[eu-source-taxonomy]]
- [[graphrag-and-downstream]]
- [[frontend-v2-design]]

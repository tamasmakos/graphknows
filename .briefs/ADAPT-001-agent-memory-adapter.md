# Brief: ADAPT-001 — AgentMemoryAdapter Protocol Class

## Working directory
`/root/graphknows`

## Goal
Implement `graphknows/memory/adapter.py` — a clean, domain-agnostic Python protocol class
that any agent runtime (Hermes, OpenCode, LangGraph, etc.) can use as a stable interface
to Graphknows memory, without knowing about Memgraph, Neo4j, pgvector, or MCP internals.

## Scope
**Only create/modify:**
- `graphknows/memory/adapter.py` (new file)
- `tests/memory/test_adapter.py` (new file)
- `graphknows/memory/__init__.py` (add export only)

Do NOT touch any other files.

## Design contract (implement exactly this)

```python
# graphknows/memory/adapter.py
"""
AgentMemoryAdapter — host-agnostic memory interface for agent runtimes.

Any agent that needs session-scoped memory can depend on this protocol
without knowing about Memgraph, Neo4j, pgvector, or MCP internals.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

# ---------------------------------------------------------------------------
# Data transfer objects
# ---------------------------------------------------------------------------

@dataclass
class IngestResult:
    file_id: str
    chunks: int
    entities: int
    elapsed_s: float
    errors: list[str] = field(default_factory=list)

@dataclass
class RecallHit:
    score: float
    text: str
    session_id: str
    entities: list[str] = field(default_factory=list)
    source_type: str = ""   # "stm_chunk", "ltm_chunk", "dialog", etc.

@dataclass
class FlushResult:
    nodes: int
    edges: int
    chunks: int
    turns: int
    elapsed_s: float
    errors: list[str] = field(default_factory=list)

@dataclass
class MemoryStats:
    stm: dict[str, Any] = field(default_factory=dict)
    ltm: dict[str, Any] = field(default_factory=dict)

# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class AgentMemoryAdapter(Protocol):
    """
    Stable interface between an agent runtime and Graphknows memory.

    Implementations must be async-safe and session-scoped.
    All methods are async so they can be used in any async agent loop.
    """

    async def ingest(
        self,
        text: str,
        session_id: str,
        title: str = "",
    ) -> IngestResult:
        """Store text as a searchable document in short-term memory."""
        ...

    async def recall(
        self,
        query: str,
        session_id: str,
        top_k: int = 5,
        scope: str = "stm",   # "stm" | "ltm" | "both"
    ) -> list[RecallHit]:
        """Retrieve the most relevant memory chunks for a query."""
        ...

    async def flush(self, session_id: str) -> FlushResult:
        """Promote short-term session memory to long-term memory."""
        ...

    async def stats(self, session_id: str = "") -> MemoryStats:
        """Return memory counts for a session (or aggregate if empty)."""
        ...
```

## Concrete implementation

Also implement `GraphknowsMemoryAdapter` — a concrete class that satisfies
`AgentMemoryAdapter` by calling the existing Graphknows services:

```python
class GraphknowsMemoryAdapter:
    """
    Concrete AgentMemoryAdapter backed by the Graphknows memory stack
    (MemoryIngestor, STM/LTM recall, SessionFlusher, memory_stats).
    """
    def __init__(self, container=None):
        # Accept optional DI container for testability.
        # If None, import from graphknows.container lazily.
        self._container = container

    # implement all 4 methods by delegating to:
    # - graphknows.memory.ingestor.MemoryIngestor  → ingest()
    # - graphknows.retrieval.recall_stm / recall_ltm → recall()
    # - graphknows.memory.flush.SessionFlusher → flush()
    # - graphknows.mcp.stdio_server memory_stats handler → stats()
    # Use asyncio.to_thread() if the underlying calls are synchronous.
```

## Tests (tests/memory/test_adapter.py)

Write unit tests that:
1. Verify `GraphknowsMemoryAdapter` satisfies `isinstance(adapter, AgentMemoryAdapter)` (Protocol check)
2. Verify each method signature matches the Protocol exactly
3. Mock the underlying services (do NOT hit live DBs in unit tests)
4. Test that `IngestResult`, `RecallHit`, `FlushResult`, `MemoryStats` are proper dataclasses
5. Test `recall()` with scope="stm", scope="ltm", scope="both"

Use `unittest.mock.AsyncMock` for async mocks. Mark tests `@pytest.mark.unit`.

## Key existing files to read first
- `graphknows/memory/ingestor.py` — understand MemoryIngestor interface
- `graphknows/memory/flush.py` — understand SessionFlusher interface
- `graphknows/retrieval/recall_stm.py` — understand STM recall
- `graphknows/retrieval/recall_ltm.py` — understand LTM recall
- `graphknows/container.py` — understand DI container pattern
- `graphknows/mcp/stdio_server.py` lines 1-100 — understand memory_stats handler

## Acceptance criteria
- [ ] `graphknows/memory/adapter.py` exists with Protocol + concrete class
- [ ] `tests/memory/test_adapter.py` exists with ≥8 passing unit tests
- [ ] `isinstance(GraphknowsMemoryAdapter(), AgentMemoryAdapter)` → True
- [ ] `uv run pytest tests/memory/test_adapter.py -q` → all green
- [ ] `uv run pytest tests/ -q -m 'not integration' --tb=short` → still all green (no regressions)
- [ ] No files outside the 3 listed above were modified

## Do not
- Add framework dependencies (no LangChain, no pydantic forced where dataclass works)
- Add HTTP or MCP transport code — that layer already exists
- Modify any existing file outside the 3 listed
- Commit changes

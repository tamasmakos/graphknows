# ADR 0001 — Runtime is the single composition root

Status: Accepted (2026-06-22)

## Context

The codebase carried two parallel composition roots: `GraphKnowsRuntime` (used by the
live MCP stdio server and web server) and `GraphKnowsContainer` (exported as the public
`GraphKnows` SDK, with its own 13 `_tool_*` functions), plus two adapter facades
(`GraphknowsAdapter`, `GraphknowsMemoryAdapter`). Codegraph analysis showed the container
and both facades had **no live callers** — they were superseded by the "Wave A" runtime
consolidation but never removed.

## Decision

`GraphKnowsRuntime` is the one transport-neutral composition root. `container.py`, both
adapter facades, `adapter_types.py` (except the `MemoryClass` enum, moved to
`memory/models/memory_class.py`), and `mcp/tool.py` were deleted. The top-level export is
`from graphknows import GraphKnowsRuntime`.

## Consequences

- One place builds the object graph; transports cannot diverge.
- The `GraphKnows` SDK alias and the `AgentMemoryAdapter` protocol are gone (breaking, by
  agreement — no external consumers).
- ~2,500 lines of dead code removed.

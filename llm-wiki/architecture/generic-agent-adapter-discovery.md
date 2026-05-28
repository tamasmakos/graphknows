---
title: Generic Agent Adapter Discovery
created: 2026-05-22
updated: 2026-05-22
type: architecture
tags: [architecture, memory, agent, ingestion, retrieval]
confidence: high
---

# Generic Agent Adapter Discovery

## Question

Can Graphknows expose a **repo-level, host-agnostic memory adapter surface** that works for many agent runtimes, instead of being designed around Hermes-specific hooks?

## Scope

This discovery pass intentionally treated Hermes as only one possible source format.
The goal was to reason from current Graphknows repo surfaces only:

- `graphknows/memory/ingestor.py`
- `graphknows/memory/dialog.py`
- `graphknows/memory/flush.py`
- `graphknows/retrieval/recall_stm.py`
- `graphknows/retrieval/recall_ltm.py`
- `graphknows/ingestion/parsers/*`
- backing stores in `graphknows/retrieval/stores/*`
- `graphknows/cli/memory.py`

## Repo-level conclusion

Yes.

Graphknows already contains enough primitives to support a generic adapter layer, but they are currently split across several components:

- `MemoryIngestor` — artifact/document-style STM writes
- `DialogWriter` — turn/dialog STM writes
- `SessionFlusher` — STM→LTM promotion
- `recall_short_term` / `recall_long_term` — retrieval

That means the missing piece is **not** the storage engine.
The missing piece is a **unified adapter contract and event envelope** at the repo level.

## Prototype artifacts

- `research/generic_agent_contract_draft.md`
- `research/proto_generic_agent_adapter.py`
- `research/proto_generic_agent_adapter_results.json`

## Minimal generic contract inferred from the repo

The smallest useful repo-level contract is:

- `open_session(session_id, metadata)`
- `start_dialog(session_id, dialog_id, participants)`
- `write_turn(session_id, dialog_id, turn)`
- `ingest_artifact(session_id, artifact)`
- `recall(query, scope, session_id, top_k)`
- `promote(session_id, options)`
- optional `forget(session_id)` / `close()`

This contract is generic because it says nothing about:
- Hermes
- LangGraph
- OpenAI Responses format
- MCP
- shell hooks
- transport

It only describes what a host runtime must give Graphknows.

## Canonical repo-level objects

### Turn
Conversation and tool-use memory unit.

Required fields:
- `turn_id`
- `index`
- `role`
- `text`

Optional:
- `timestamp`
- `tool_calls[]`

### Artifact
Any non-turn memory payload.

Required fields:
- `artifact_id`
- `title`
- `text`

Optional:
- `source_hint`
- `memory_class`
- `metadata`

### memory_class
The adapter discovery pass confirmed this should be first-class metadata:
- `raw_evidence`
- `summary`
- `state`
- `episode`
- `fact`
- `other`

This is the repo-level mechanism that lets Graphknows remain generic without confusing evidence and memory.

## Key discovery 1 — Graphknows can already act as the generic backend

The prototype wrapped the existing repo primitives into a research-only generic adapter and ran it on:

1. a **real Hermes-derived generic event stream**
2. a **synthetic non-Hermes host event stream**

Both used the same adapter surface.

This is important: the adapter itself never saw Hermes-native hook names.
Hermes-specific parsing was kept outside the adapter.

## Key discovery 2 — the host-specific work belongs outside the adapter

The prototype made this split explicit.

### Outside the adapter
Host-specific logic:
- source parsing
- tool-call reconstruction
- terminal-output summarization
- host-native role normalization

### Inside the adapter
Graphknows-native logic:
- STM graph/vector writes
- dialog persistence
- artifact persistence
- recall
- promotion to LTM

That separation is the correct generic architecture.

## Key discovery 3 — terminal encapsulation must be part of the generic event envelope

The repo should not think of terminal output as a Hermes problem.
It is a generic agent-runtime problem.

The prototype used a structured `ToolExecutionRecord` summary with fields equivalent to:
- tool name
- args
- result summary
- latency
- tool class
- outcome
- error class

For terminal-like operations, the summary included:
- command intent
- command class
- exit code / status
- artifact paths touched
- first meaningful signal
- durable-fact candidate yes/no

This is the correct abstraction level.
Not raw stdout.
Not framework-native callback payloads.

## Key discovery 4 — the current repo still mixes evidence and memory at recall time

This was the most important finding from the prototype.

### Real-host prototype result
Namespace: `generic-adapter-real-hermes`

- `turn_count = 47`
- `artifact_count = 75`
- promotion succeeded with:
  - `nodes_written = 213`
  - `edges_written = 480`
  - `chunks_promoted = 75`
  - `turns_promoted = 47`
  - `errors = []`

### Synthetic-host prototype result
Namespace: `generic-adapter-synthetic-host`

- `turn_count = 2`
- `artifact_count = 3`
- promotion succeeded with:
  - `nodes_written = 16`
  - `edges_written = 20`
  - `chunks_promoted = 3`
  - `turns_promoted = 2`
  - `errors = []`

So the generic adapter concept works operationally.

But the real-host recall result exposed the next repo-level problem:

For continuity-style STM queries, the top hit was often a `terminal_execution` chunk instead of a distilled state/fact summary.

That means:
- the adapter contract is viable
- but the current retrieval surface is still too flat

## Architectural implication

Graphknows now needs **memory-class-aware recall**, not just a single STM namespace.

At the repo level, there are two viable directions:

### Option A — single namespace, metadata-aware retrieval
Keep one STM store, but filter/boost by `memory_class` at retrieval time.

Example policy:
- continuity queries boost `state`, `episode`, `fact`
- terminal/debug queries boost `raw_evidence`, `summary`
- raw evidence is de-boosted by default

### Option B — explicit lanes
Represent evidence and memory as explicit retrieval lanes:
- evidence lane
- memory lane

This is conceptually cleaner, but requires more repo-level API and retrieval changes.

Current evidence suggests **Option A first**.

## Key discovery 5 — Graphknows is missing one repo-level interface, not a new backend

The prototype strongly suggests the next repo-level addition should be something like:

- `graphknows/memory/adapter.py`
- generic dataclasses / TypedDicts for:
  - `ArtifactRecord`
  - `TurnRecord`
  - `ToolCallRecord`
  - `RecallRequest`
  - `PromoteOptions`
- one orchestrating class wrapping:
  - `MemoryIngestor`
  - `DialogWriter`
  - `SessionFlusher`
  - recall helpers

This would let any host runtime build its own source normalizer while sharing one Graphknows memory API.

## What this discovery phase rules out

This pass rules out a repo design centered on:
- Hermes hook names
- framework-native turn models
- raw stdout ingestion as the default memory path
- one undifferentiated retrieval lane

Those are host concerns or evidence concerns, not repo-level memory concerns.

## Recommended next repo-level step

Before any host integration work, do one small repo-level design/implementation pass:

1. create a first-class generic adapter module in Graphknows
2. add `memory_class` to the canonical artifact surface
3. extend recall to filter/boost by memory class
4. keep host-specific normalizers outside the adapter

That is the cleanest path to making Graphknows reusable across:
- Hermes
- LangGraph agents
- OpenAI tool-call agents
- custom event-driven runtimes

## Related pages
- [[architecture/generic-memory-engine-discovery]]
- [[architecture/memory-layer-v2]]
- [[research/kg-construction]]

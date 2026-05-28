---
title: Generic Agent Adapter Implementation Plan
created: 2026-05-23
updated: 2026-05-23
type: architecture
tags: [architecture, planning, memory, adapter, retrieval]
confidence: high
---

# Generic Agent Adapter Implementation Plan

## Goal
Turn the generic-adapter discovery into a repo-level implementation wave that stays host-agnostic.

## Scope
This wave is intentionally limited to:
- canonical memory adapter record types
- a first-class `graphknows/memory/adapter.py` facade
- deterministic tool/terminal summarization
- metadata-aware retrieval preference for distilled memory
- conformance tests and real-trace evaluation
- documentation

## Deliberate non-scope
- Hermes plugin integration
- LangGraph/OpenAI/MCP adapters
- separate retrieval lanes or separate indices for v1
- LLM-based terminal summarization in the critical path

## Design decisions
- Host-specific parsing stays outside Graphknows.
- Raw evidence and distilled memory stay in the same storage path for v1.
- Retrieval should use `memory_class` metadata to boost `state` / `episode` / `fact` / `summary` and de-boost `raw_evidence` for continuity queries.
- Terminal output should be encapsulated as structured summaries, not stored as raw stdout for default recall.

## Kanban wave
- `t_e28600a0` — GAA-001: Core record types
- `t_e50a92d2` — GAA-002: Generic adapter facade
- `t_5b7162b8` — GAA-003: Deterministic tool and terminal summarizer
- `t_b5f900a3` — GAA-004: Metadata-aware recall preference
- `t_999d13a5` — GAA-005: Adapter conformance tests and synthetic host fixtures
- `t_5ae3f29d` — GAA-006: Real-trace evaluation harness
- `t_04b20f0b` — GAA-007: Docs and developer surface

## Dependencies
- GAA-001 -> GAA-002
- GAA-001 -> GAA-003
- GAA-003 -> GAA-004
- GAA-002 + GAA-003 + GAA-004 -> GAA-005
- GAA-002 + GAA-003 + GAA-004 + GAA-005 -> GAA-006
- GAA-006 -> GAA-007

## Exit gate before host adapters
Do not start host-specific adapters until:
- generic types exist and are tested
- adapter facade exists and is tested
- deterministic tool/terminal summarization exists and is tested
- continuity recall ranks distilled memory above raw terminal evidence on real traces
- evaluation emits a machine-readable report

## Primary artifact
- Plan file: `/root/graphknows/.hermes/plans/2026-05-23-generic-agent-adapter-repo-plan.md`

## Related pages
- [[architecture/generic-agent-adapter-discovery]]
- [[architecture/generic-memory-engine-discovery]]

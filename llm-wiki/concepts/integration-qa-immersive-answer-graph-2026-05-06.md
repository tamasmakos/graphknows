---
title: Integration QA — Immersive Answer Graph
created: 2026-05-06
updated: 2026-05-06
type: concept
tags: [frontend, architecture, graph, planning]
sources: []
confidence: high
---

# Integration QA — Immersive Answer Graph

## Scope

Principal-engineer integration of the reviewed worktree diffs into an integration branch, followed by backend verification and live browser QA against temporary local agent/web instances.

Related pages:
- [[immersive-answer-graph-redesign-2026-05-06]]
- [[frontend-v2-design]]
- [[graph-visualisation-diagnosis]]

## What integrated cleanly

- `services/agent/main.py`
  - HTTP and WebSocket payload shaping are better aligned
  - normalized reasoning step metadata is present
  - final WS answer frame includes `sources` and `reasoning_steps`
- `services/agent/session_graph.py`
  - richer session graph serialization
  - display names and node/edge property preservation improved
- `tests/test_session_graph.py`
  - integration branch passes `4 passed, 5 skipped`
- `apps/web/index.html`
  - persistent inspector exists
  - node click opens inspector
  - graph renders with integrated branch

## Verified results

### Backend quality gate
- `python3 -m py_compile services/agent/main.py services/agent/session_graph.py tests/test_session_graph.py` ✅
- `python3 -m pytest tests/test_session_graph.py -q` → `4 passed, 5 skipped` ✅
- `ruff check ...` ✅
- `ruff format --check ...` ✅

### Live QA outcome
The integrated surface is **better wired but not product-ready**.

Observed in live browser QA:
- graph renders and inspector opens
- integrated query produced `68` visible graph nodes in the browser
- backend `/chat` returned `reasoning_steps: 7`, `graph.nodes: 62`, `graph.edges: 99`

But the branch still fails the target product bar:

1. **Reasoning still leaks raw JSON into the UI**
   - `search_entities` / tool observations render raw arrays and object blobs
   - this reads like debug output, not a Perplexity-grade reasoning surface

2. **Citations / source cards are still effectively missing**
   - `/chat` returned `sources_len = 0`
   - browser UI showed `sourceCardCount = 0`
   - answer cannot provide refined direct evidence navigation yet

3. **Inspector is still too shallow**
   - opens successfully, but mainly shows relation lists
   - does not yet surface the metadata depth expected from a Neo4j-like inspector

4. **Primary graph UX still advertises layout-mode thinking**
   - visible `Force` layout control remains in the main surface
   - this contradicts the redesign direction toward one stable investigative model

5. **Reasoning presentation is not yet truly polished**
   - inline trace container exists, but the visible answer flow still presents a mixed debug/feed feel
   - the overall surface is not yet close enough to Perplexity readability

## Principal-engineer decision

Do **not** commit or push this integration branch as the redesign milestone.

Reason:
- engineering integration succeeded
- backend validation succeeded
- but live UX QA shows the branch still misses the core user requirement: Neo4j-like graph clarity + Perplexity-like answer/citation experience

This branch is a valid **integration checkpoint**, not a merge-ready product slice.

## Immediate next build priorities

1. transform reasoning observations into UI summaries instead of raw JSON dumps
2. produce answer-grounded citation objects from backend data
3. render refined source cards with direct document path / source path
4. deepen inspector metadata model
5. demote/remove visible layout-mode switching from the primary graph UX

## Branch / worktree
This QA was performed in a temporary integration branch/worktree before any canonical promotion or commit decision.

---
title: Immersive Answer + Graph Redesign
created: 2026-05-06
updated: 2026-05-06
type: concept
tags: [frontend, ux, graph, architecture, planning]
sources: []
confidence: high
---

# Immersive Answer + Graph Redesign

## Why this note exists

User feedback on 2026-05-06 was direct and correct: the current graph visualisation is not clean or analytically helpful, the lexical/provenance graph is not navigational enough, reasoning trace readability is weak, metadata is not available on click/tap, and the overall feel should move closer to **Neo4j Browser for graph interaction** and **Perplexity for answer/reasoning/citation UX**.

This note records the grounded diagnosis after inspecting the live frontend on `:8080`, the static SPA implementation in `apps/web/index.html`, and the current agent WebSocket/API behavior in `services/agent/main.py`.

## Verified current-state problems

### 1. The live app is the static SPA, not the Next.js app
The user-facing surface currently served on `:8080` is `apps/web/index.html` through `apps/web/server.py`, not the React/Next chat UI. UX work should target the static SPA first unless the frontend is consolidated.

### 2. Graph layout variety is a trap, not a feature
The graph exposes `Force / Radial / Hierarchy / Circle` layout switching in the main toolbar. This invites mode-hopping instead of giving the user one stable, legible mental model. The user feedback explicitly rejects this direction.

### 3. Click/tap does not open a real inspector
The current graph uses hover tooltip for metadata and click only toggles neighborhood highlight. This is fundamentally wrong for mobile and weak on desktop. Metadata must live in a persistent inspector panel or bottom sheet triggered by click/tap.

### 4. The reasoning stream is not truly live
`services/agent/main.py` currently waits for the full GraphRAG response, then replays `reasoning_steps` over WebSocket. This is post-hoc streaming, not true live reasoning. The graph also does not grow step-by-step from streamed deltas.

### 5. WebSocket answers do not carry refined sources
The WebSocket `answer` frame sends `answer`, `graph`, and `session_id`, but not `sources` or enriched reasoning-step metadata (`tool_name`, `observation_type`). The frontend therefore falls back to deriving source cards from graph chunk nodes, which makes citations generic and indirect.

### 6. Citations are currently weak at the data layer too
Direct call to `POST /chat` on the agent returned:
- `reasoning_steps: 6`
- `graph_nodes: 56`
- `graph_edges: 103`
- `sources: 0`

So the UI problem is not only styling; the answer pipeline is often not returning answer-grounded citations at all.

### 7. The graph is still a generic force canvas, not an investigative surface
The current graph is visually a D3 force view with filters and legend, but not yet a real knowledge-work UI. It lacks:
- stable Neo4j-like spatial behavior
- persistent details panel
- direct document navigation
- explicit provenance path view
- readable edge semantics at interaction time

## Product direction

### A. Entity graph should feel like Neo4j Browser
The main entity graph should use a **single canonical layout mode** with restrained physics, dark canvas, clear label hierarchy, persistent selection, and an inspector panel.

Target behaviors:
- click/tap selects a node and opens metadata panel
- selected node remains pinned/centered until dismissed
- immediate neighbors become visually primary
- edge semantics are shown in the inspector or focused overlays, not sprayed everywhere
- graph should look stable enough to build user memory across interactions

This is closer to `[[frontend-v2-design]]` than the current multi-layout toolbar.

### B. Provenance / lexical graph should be navigational, not decorative
The provenance layer should not reuse the same entity-graph treatment. It should behave like a **resource navigation graph**:
- `Document -> section/heading -> chunk -> claim/entity`
- every citation opens the exact supporting excerpt
- source path is explicit and traversable
- the user can go from answer sentence to exact chunk to original source document

This aligns with the missing `Document` node idea already identified in `[[graph-visualisation-diagnosis]]`.

### C. Reasoning path should become a streaming evidence timeline
The reasoning trace should look more like a **live investigative timeline / step feed** than a stack of debug cards.

Target behaviors:
- step starts stream immediately when the tool starts
- step status updates in place (`searching`, `reading`, `comparing`, `synthesising`)
- each step shows the concrete artifact produced: entities found, path found, chunks selected, contradiction resolved
- answer citations appear inline as the answer is assembled
- the graph updates with `graph_delta` frames during reasoning, not only after final answer

This is the intended spirit of `[[service-design-agent]]`, but the current implementation is still replay-based.

### D. Answer surface should become Perplexity-grade
The answer area should be:
- answer-first
- citation-first
- cleaner typography
- explicit inline source anchors
- expandable evidence panels
- better distinction between final answer and reasoning artifacts

Perplexity-like here means:
- readable prose first
- citations attached exactly where claims are made
- supporting sources shown as refined cards with title, institution, date, excerpt, and direct open action
- reasoning available, but secondary and legible

## Concrete redesign priorities

### Priority 1 — Remove layout indecision
- demote or remove `Radial / Hierarchy / Circle`
- keep one default investigative layout
- optionally keep alternate layouts behind a hidden advanced/debug control, not the main UX

### Priority 2 — Add persistent node inspector
On click/tap, open an inspector with:
- canonical label / type
- summary
- doc count / citation count
- neighboring entities grouped by relation
- exact supporting source excerpts
- direct source links

### Priority 3 — Split graph into two modes
Do **not** force one graph to do everything.

Use:
1. **Entity graph** — immersive, Neo4j-like, relationship-centric
2. **Evidence graph / provenance view** — navigational, path-centric, source-centric

### Priority 4 — Fix streaming contract
Agent WebSocket frames should include:
- `step_start`
- `step_update`
- `step_result`
- `graph_delta`
- `answer_chunk`
- `citation`
- `answer_final`

And each step frame must include `tool_name`, `observation_type`, and a structured summary fit for UI rendering.

### Priority 5 — Fix citation model before polishing citation UI
Citations should be answer-grounded objects with fields such as:
- `claim_span`
- `chunk_id`
- `doc_id`
- `doc_title`
- `institution`
- `date`
- `heading_path`
- `excerpt`
- `url`
- `score`

Without this, frontend citation polish will remain fake.

## Implementation implication

This is not a force-tuning problem. It is a **surface model problem** across frontend + agent streaming contract + citation/provenance schema.

The minimum serious implementation order is:
1. fix agent WebSocket/event contract
2. fix citation payloads and document metadata
3. add persistent inspector + direct source navigation
4. simplify graph to one primary Neo4j-like interaction model
5. separate provenance/evidence navigation from entity exploration

## Links
- [[frontend-v2-design]]
- [[graph-visualisation-diagnosis]]
- [[service-design-agent]]
- [[mobile-first-frontend-revamp-2026-05-06]]
- [[mobile-graph-usability-pass-2026-05-06]]

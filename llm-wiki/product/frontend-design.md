---
title: "Frontend Design — apps/web/"
created: 2026-05-04
updated: 2026-05-20
type: product
tags: [frontend, design, ux, visualization]
confidence: high
---

# Frontend Design — `apps/web/`

The frontend is the only user-facing surface. It hides the service topology behind a split-pane investigation UI: chat on the left, knowledge graph on the right.

---

## Implementation Status (verified 2026-05-20)

**Live and working:**
- `apps/web/index.html` — static single-page app (~5100 lines, inline JS + CSS)
- `apps/web/server.py` — FastAPI server that serves `/`, proxies `/api/*`, and proxies `/ws/chat/{session_id}`
- `apps/web/src/` — partial React component tree (`app/chat/`, `app/graph/`, `components/`) — coexists with the static page
- Split-pane layout: chat left, D3 graph canvas right with collapsible graph panel
- Named sessions sidebar with localStorage persistence (`graphknows_sessions` key)
- Session rename, session list, new session button
- D3 force-directed graph canvas (`#graph-canvas`) with node focus chip
- Graph toolbar with layout controls
- Mobile sidebar + graph panel toggle
- Per-message export button (`btn-export-msg`) — exports individual message text
- WebSocket integration for WS events (search_start, graph_update, brief_done, etc.)

**Not yet implemented (aspirational):**
- PDF export of full EPRS brief
- Markdown export of full brief
- Reasoning trace panel (was in earlier spec, removed)
- Dual graph tabs (KG + Lexical) — single graph canvas in current build
- Agent port: `8030` on host (NOT `8002` — that was a stale value from an earlier build)

---

## Core UX Flow

1. User enters investigation topic in chat
2. Agent streams progress: search events → graph_update (live D3 nodes/edges appear) → brief_done
3. User inspects the knowledge graph (click node → focus chip shows entity details)
4. User can start a new session or return to a past session from the sidebar

---

## Service Wiring

- Frontend runs on host port `8080`
- `server.py` proxies `/api/*` → `http://agent:8000` (container) = `http://localhost:8030` (host)
- WebSocket: `server.py` proxies `/ws/chat/{session_id}` → agent WS endpoint
- No direct calls to search or graphknows services from the browser

---

## Key Files

```
apps/web/
  index.html       # Full SPA — layout, styles, session management, D3 graph, WS client
  server.py        # FastAPI: serve static, proxy /api/* and /ws/*
  launch.py        # Optional: starts server + cloudflare tunnel
  start.sh         # Shell launcher
  src/             # Partial React component structure (future migration target)
  static/          # Static assets
```


The user should be able to:

1. ask one focused research question
2. watch the agent reveal its plan and evidence-gathering progress
3. inspect the evolving graph separately from the answer text
4. export the result for briefings or reporting

## Layout

The current layout is a split pane:

- left pane: investigations sidebar, chat history, input box, inline reasoning/answer cards
- right pane: graph canvas, graph controls, node-type filters, sources panel
- mobile: explicit Answer / Graph view switcher

## Current Live Behavior

The frontend now frames the experience as an investigation workflow rather than a single-step chat answer.

Verified live behavior:

- empty state prompts invite the user to start an investigation
- suggestion chips trigger agent investigations immediately
- the agent streams `plan_ready`, search progress, graph writes, and the final brief into the chat column
- the graph view updates during the run and remains inspectable after completion
- investigation names and local history persist in localStorage

## WebSocket Contract Used by the Frontend

The browser currently connects to the frontend proxy path:

```javascript
const ws = new WebSocket(`/ws/chat/${sessionId}`);
```

That route is proxied to the agent compatibility stream on `:8002`.

Observed live message pattern:

```json
{"type":"status","status":"thinking"}
{"type":"ws_event","payload":{"event_type":"plan_ready","payload":{}}}
{"type":"ws_event","payload":{"event_type":"search_start","payload":{}}}
{"type":"ws_event","payload":{"event_type":"graph_update","payload":{}}}
{"type":"ws_event","payload":{"event_type":"brief_done","payload":{}}}
{"type":"done","briefing":"..."}
```

The frontend unwraps `ws_event.payload.event_type` and maps it into:

- inline reasoning cards
- live answer body updates
- graph refreshes when `graph_update` arrives
- final answer rendering when `done` lands

## Graph UX

The graph panel is still a D3 force-directed view with:

- node colors by label/type
- layout controls
- filter-by-type dropdown
- reset and export controls
- click-to-focus node interactions

The frontend now initializes the graph only after the graph state is ready. This removed startup timing bugs where responsive layout code ran before graph sizing or label state existed.

## Session Behavior

Current persistence model:

- live graph state is maintained by the agent session backend
- the frontend stores investigation names and recent shells in localStorage
- restoring a session hydrates graph state through the frontend proxy

## Export Surface

The current SPA still supports export-oriented UX:

- answer export per response card
- graph PNG export
- share link generation

The original broader export vision remains valid, but the implemented surface is the static SPA above.

## Tech Stack

| Layer | Current choice | Why |
|---|---|---|
| App shell | Static HTML SPA | Zero build tooling, fast iteration |
| Frontend server | FastAPI proxy | Serves SPA and forwards HTTP/WS traffic |
| Graph viz | D3.js v7 | Full control over graph behavior |
| Styling | Inline CSS + CSS variables | Single-file deploy simplicity |
| WebSocket | Native browser WebSocket | Thin streaming layer |
| State | In-page JS state + localStorage | Adequate for the current SPA |

## Known Gaps

1. The investigative workflow can finish with a fully rendered answer and graph while the Sources side panel still shows `0`.
2. Long prompts can take 2-3 minutes to complete; UX depends on keeping the websocket and spinner alive for the full run.
3. The current frontend uses the compatibility websocket route `/ws/chat/{session_id}` rather than calling the primary `/sessions/{session_id}/stream` route directly.

## Links

[[architecture/reference]] [[architecture/service-agent]] [[product/roadmap]] [[ops/technical-debt]]

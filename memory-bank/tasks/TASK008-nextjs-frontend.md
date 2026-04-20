# TASK008 — Phase 5: Next.js 15 Frontend (apps/web)

**Status:** Pending  
**Added:** 2026-04-20  
**Updated:** 2026-04-20

## Original Request
Build the Next.js 15 App Router frontend with 4 views (Documents, Chat, Graph Explorer, Analytics), Kibo UI + shadcn/ui components, TanStack Query for server state, and Next.js Route Handlers as the unified `/api/v1/*` BFF.

## Thought Process
This is the largest single task. Breaking it into sub-phases:
1. **Scaffold** — Next.js project, Tailwind config, base layout, sidebar nav.
2. **BFF Route Handlers** — proxy layer first, so other views can work against real data.
3. **Document Manager** — most straightforward view; establishes the polling + upload patterns.
4. **Agent Chat** — most complex view due to SSE streaming + citations + reasoning trace.
5. **Graph Explorer** — react-force-graph-2d integration + inspector panel.
6. **Analytics** — recharts dashboards.

packages/ui is also set up here (Tailwind preset, design tokens, StatusBadge, CitationCard).

## SSE Consumption Pattern
```typescript
// lib/sse.ts - POST-based SSE
async function* streamAgentResponse(input: ChatRequest) {
  const res = await fetch('/api/v1/chat/stream', {
    method: 'POST', body: JSON.stringify(input),
    headers: {'Content-Type': 'application/json'}
  });
  const reader = res.body!.getReader();
  const decoder = new TextDecoder();
  // parse SSE lines: "event: xxx\ndata: {...}\n\n"
  for await (const chunk of readableStreamLines(reader, decoder)) {
    yield parseSSEEvent(chunk);  // { event, data }
  }
}
```

## BFF Endpoint Map
| Route Handler | Upstream | Method |
|---|---|---|
| `/api/v1/documents` | graphgen `/documents` | GET, POST (multipart) |
| `/api/v1/documents/[id]` | graphgen `/documents/{id}` | GET, DELETE |
| `/api/v1/documents/[id]/reprocess` | graphgen `/documents/{id}/reprocess` | POST |
| `/api/v1/documents/[id]/events` | graphgen WS → SSE proxy | GET (SSE) |
| `/api/v1/chat` | graphrag `/v1/chat` | POST |
| `/api/v1/chat/stream` | graphrag `/v1/chat/stream` | POST (SSE passthrough) |
| `/api/v1/conversations` | local SQLite | GET, POST |
| `/api/v1/conversations/[id]` | local SQLite | GET, PATCH, DELETE |
| `/api/v1/graph/query` | graphrag Cypher allow-list | POST |
| `/api/v1/graph/nodes/[id]/neighbors` | graphrag `/v1/nodes/{id}/neighbors` | GET |
| `/api/v1/graph/labels` | graphrag `/v1/schema` | GET |
| `/api/v1/analytics/kpis` | graphgen `/analytics/kpis` | GET |
| `/api/v1/analytics/entities/top` | graphgen `/analytics/entities/top` | GET |
| `/api/v1/analytics/timeline` | graphgen `/analytics/timeline` | GET |

## Implementation Plan
### Scaffold (5a prerequisite)
- [ ] Create `apps/web` Next.js 15 project with TypeScript, Tailwind, App Router
- [ ] Set up `packages/ui` with Tailwind preset + design tokens + base components
- [ ] Configure Turborepo pipeline for `web` depending on `types#build`
- [ ] Create root app layout + dashboard layout with sidebar

### BFF Layer
- [ ] Create all Route Handlers in `app/api/v1/**`
- [ ] Create `lib/api.ts` typed client (fetch wrapper with error handling)
- [ ] Create `lib/sse.ts` POST-based SSE stream parser
- [ ] Create `lib/db/conversations.ts` SQLite store

### View: Document Manager
- [ ] `/documents/page.tsx` — table + Kibo Dropzone
- [ ] Upload handler with per-file progress
- [ ] Status polling (TanStack Query, 3s interval)
- [ ] Document detail drawer (chunks/entities/logs tabs)
- [ ] `IngestionLogStream` component (SSE → live tail)

### View: Agent Chat
- [ ] `/chat/page.tsx` — full-width chat shell + sidebar
- [ ] `useAgentStream` hook (async generator from lib/sse.ts)
- [ ] `AssistantMessage` with inline `[n]` citation markers
- [ ] `CitationCard` hover card
- [ ] `ReasoningTimeline` (Kibo Timeline component)
- [ ] "Visualize Context" → Sheet with `GraphCanvas`

### View: Graph Explorer
- [ ] `/graph/page.tsx` — react-force-graph-2d canvas
- [ ] Filter sidebar (node labels, document, entity type)
- [ ] `InspectorPanel` with "Expand Neighbors" button
- [ ] Color tokens from `packages/ui` applied to node labels

### View: Analytics
- [ ] `/analytics/page.tsx`
- [ ] 4 KPI `StatsCard` components
- [ ] `EntityFrequencyChart` (recharts bar)
- [ ] `DocumentStatusDonut` (recharts pie)
- [ ] `IngestionTimelineChart` (recharts area)

### Final
- [ ] `apps/web/Dockerfile` (multi-stage, non-root, standalone output)
- [ ] `middleware.ts` auth stub

## Progress Tracking

**Overall Status:** Not Started — 0%

### Subtasks
| ID | Description | Status | Updated | Notes |
|----|-------------|--------|---------|-------|
| 8.1 | Next.js scaffold + Tailwind + layout | Not Started | 2026-04-20 | |
| 8.2 | packages/ui setup | Not Started | 2026-04-20 | |
| 8.3 | All BFF Route Handlers | Not Started | 2026-04-20 | |
| 8.4 | lib/api.ts + lib/sse.ts + SQLite store | Not Started | 2026-04-20 | |
| 8.5 | Document Manager view | Not Started | 2026-04-20 | |
| 8.6 | Agent Chat view + SSE hook | Not Started | 2026-04-20 | |
| 8.7 | Graph Explorer view | Not Started | 2026-04-20 | |
| 8.8 | Analytics view | Not Started | 2026-04-20 | |
| 8.9 | apps/web/Dockerfile | Not Started | 2026-04-20 | |
| 8.10 | middleware.ts auth stub | Not Started | 2026-04-20 | |

## Progress Log
### 2026-04-20
- Task created during planning session.

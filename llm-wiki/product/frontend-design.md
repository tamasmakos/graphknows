---
title: Frontend v2 Design — apps/web/
created: 2026-05-04
updated: 2026-05-04
type: concept
tags: [frontend, design, ux, visualization, export]
confidence: high
---

# Frontend v2 Design — `apps/web/`

> The frontend is the **only user-facing surface**. It communicates exclusively with `services/agent/`. All backend complexity is hidden. The core UX insight: political researchers need to argue and convince — every insight must be exportable and shareable.

---

## Core UX Principle

> **"Ask → Discover → Export → Convince"**

1. **Ask** — natural language query in the investigation session chat
2. **Discover** — live knowledge graph grows as the agent reasons, source citations appear
3. **Export** — one-click export of any graph state, insight, or source trail
4. **Convince** — sharable link or embeddable card that others can inspect

---

## Layout — Split Pane

```
┌─────────────────────────────────────────────────────────────────┐
│  📂 Investigations     [+ New]         ⚙️ Settings    👤 User   │
├──────────────────┬──────────────────────────────────────────────┤
│                  │                                               │
│  💬 Chat         │  🕸️ Graph Canvas (D3.js force-directed)     │
│  (left ~40%)     │  (right ~60%)                                │
│                  │                                               │
│  [query input]   │  ┌── node ──────────────────────────────┐   │
│  [chat history]  │  │ Macron ←──SUPPORTS── EU Defense Fund │   │
│                  │  │    ╰──OPPOSES── Sovereignty Clause   │   │
│  ▼ Reasoning     │  └──────────────────────────────────────┘   │
│  [collapsed DAG] │                                               │
│                  │  [ 🔍 Filter ] [ 📤 Export ] [ 🔗 Share ]   │
└──────────────────┴──────────────────────────────────────────────┘
```

---

## Component Breakdown

### 1. Investigation Sidebar (left panel, top)
- List of named investigations (persisted, resumable)
- Each item: name, last updated, node count, mini graph thumbnail
- Actions: open, duplicate, rename, delete

### 2. Chat Panel (left panel, main)
- Query input with keyboard shortcut (Cmd+Enter)
- Chat history: user messages + agent answers with inline citations
- Citations link to source documents (opens in overlay)
- Each answer includes: confidence badge, source count, export button

### 3. Reasoning Trace (left panel, collapsible)
- Collapsible DAG below the chat
- Nodes = reasoning steps (tool call, synthesis)
- Edges = "led to" 
- Color: confirmed=blue, uncertain=amber, contradicted=red
- On hover: shows tool name, query, result summary
- On click: expands to full tool result
- Subscribed to WebSocket stream — animates live as agent reasons

### 4. Graph Canvas (right panel, main)
- D3.js force-directed graph
- Nodes: colored by entity label (Person=blue, Org=purple, Law=orange, Topic=green)
- Edges: thickness = relation weight, label = relation type
- Session-discovered nodes: slightly highlighted (dashed border)
- Node click: shows detail panel (entity metadata + source chunks)
- Edge click: shows source chunk grounding the relationship
- Controls: zoom, pan, filter by label/date/source, search within graph

### 5. Export Suite (right panel, toolbar)
```
[ PNG ]  [ SVG ]  [ Interactive HTML ]  [ 🔗 Permalink ]  [ 📋 Citation Card ]
```
- **PNG/SVG**: current graph viewport, high-res, branded
- **Interactive HTML**: self-contained D3.js bundle — fully functional offline
- **Permalink**: shareable URL with graph state encoded (entity filter, date range, layout)
- **Citation Card**: structured summary for embedding in reports — entity, relationship, source, date

---

## D3.js Graph Spec

### Node Schema (frontend)
```typescript
interface GraphNode {
  id: string;
  name: string;
  label: string;          // PERSON | ORG | GEO | LAW | TOPIC | CONCEPT
  source: "neo4j" | "session" | "inferred";
  confidence: number;
  community_id?: string;
  x?: number; y?: number; // D3 layout positions
}
```

### Edge Schema (frontend)
```typescript
interface GraphEdge {
  id: string;
  source: string;   // node id
  target: string;   // node id
  relation_type: string;
  weight: number;
  source_chunks: string[];
}
```

### WebSocket Integration
```typescript
// Subscribe to live graph deltas from agent
const ws = new WebSocket(`/sessions/${sessionId}/stream`);
ws.onmessage = (event) => {
  const msg = JSON.parse(event.data);
  if (msg.type === "graph_delta") {
    addNodesToGraph(msg.nodes);
    addEdgesToGraph(msg.edges);
    restartSimulation(); // D3 force layout
  }
  if (msg.type === "step_start") {
    addReasoningStep(msg);
  }
};
```

---

## Investigation Session Persistence

Each session persists:
- `name`: user-assigned
- `created_at`, `last_updated`
- `chat_history`: list of user+agent messages
- `graph_snapshot`: last known graph state (nodes + edges as JSON)
- `exports`: list of exported files (paths or URLs)
- `session_id`: links to agent service session for live resumption

Storage: backend-persisted (agent service) with localStorage fallback for offline continuity.

---

## Export File Formats

### PNG/SVG
- Current D3 canvas → `html2canvas` or `d3-svg-export`
- Watermark: "Graphknows · graphknows.io · exported {date}"
- Filename: `{investigation-name}_{date}_{node-count}nodes.png`

### Interactive HTML
- Inline all D3.js + node/edge data as JSON
- Self-contained single-file bundle
- Opens in any browser, fully interactive (zoom, click, filter)
- No server required — works air-gapped

### Permalink
- Encode: entity filter, date range, layout positions, highlight set
- URL: `https://app.graphknows.io/share/{hash}`
- Share target can view without login (read-only)

### Citation Card
```markdown
**[Emmanuel Macron] SUPPORTS [EU Defence Fund]**
Source: EP Plenary Debate, 08 Jun 2022
Text: "We need a European defence fund that..."
Graph context: 3 supporting, 1 contradicting edge
Confidence: 0.91 | Export date: 2026-05-04
```

---

## Tech Stack

| Layer | Choice | Reason |
|---|---|---|
| Framework | Next.js 14 (existing) | SSR, App Router, TypeScript |
| Graph viz | D3.js v7 (force-directed) | Maximum control, export-friendly |
| Styling | Tailwind CSS (existing) | Rapid iteration |
| WebSocket | native browser WebSocket | Simple, no library needed |
| Export | html2canvas + d3-svg-export | Proven, no server needed |
| State | Zustand | Lightweight, no Redux overhead |

---

## Related Tasks

- t_ab74b2cf — frontend v2 split-pane
- t_948af8c4 — reasoning trace sidebar
- t_3094ed80 — investigation sessions
- t_cc183272 — export suite

## Links

[[product-architecture-v2]] [[service-design-agent]] [[business-plan]]

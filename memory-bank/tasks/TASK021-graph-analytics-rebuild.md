# TASK021 — Graph Explorer + Analytics Rebuild

**Status:** Pending  
**Added:** 2026-04-21  
**Updated:** 2026-04-21  
**Phase:** D — UI Rebuild (requires TASK017 complete)

---

## Original Request
The `/graph` page shows three separate lists (nodes, edges, layout as text) which is useless. The `/analytics` page is an empty stub. Both need to be rebuilt: Graph as an interactive force-directed explorer with an inspector panel; Analytics as cards + charts showing real ingestion metrics.

---

## Implementation Plan

### Graph Explorer

1. **Remove the three-list "schema" view** — delete current `/graph/page.tsx` content.

2. **Build `components/graph/GraphExplorer.tsx`**:
   - Full-screen `ForceGraph2D` (already installed: `react-force-graph-2d`)
   - Top filter bar: label filter (Combobox, Kibo), document scope filter (Combobox)
   - Node click → open `<NodeInspector>` Sheet (right side)
   - Right-click node → context menu (Expand neighbors, Copy ID)
   - Minimap (built into ForceGraph2D)

3. **Build `components/graph/NodeInspector.tsx`** — shadcn Sheet:
   - Tabs: Properties, Connected Nodes, Source Chunks
   - Properties: key-value table of node properties
   - Connected Nodes: list of neighbors with type badges + count
   - Source Chunks: which document chunks mention this entity
   - "Expand neighbors" button → calls `GET /api/v1/graph/expand?node_id=...&depth=1`

4. **New backend endpoint `GET /api/v1/graph/expand`**:
   - In `services/graphrag/src/main.py`
   - Params: `node_id`, `depth=1`, `limit=50`
   - Returns: `{nodes: [...], edges: [...]}`

5. **Lazy loading**: initial graph load fetches top-100 nodes by degree. "Load more" button adds next 100.

### Analytics Dashboard

6. **Install recharts**:
   ```bash
   pnpm --filter web add recharts
   ```

7. **New backend endpoints in `services/graphgen`**:
   - `GET /analytics/time-series` — ingestion jobs per day for last 30 days: `[{date, count}]`
   - `GET /analytics/entities/top` — top-20 entities by occurrence: `[{name, type, count}]`
   - `GET /analytics/summary` — document count, chunk count, entity count, edge count (already exists but extend)

8. **Build `app/(app)/analytics/page.tsx`**:
   - Row 1: 4 × stat Cards (Documents, Chunks, Entities, Edges) with Skeleton loading
   - Row 2: Area chart — ingestion over time (recharts AreaChart)
   - Row 3: Bar chart — top entities (recharts BarChart, horizontal)
   - Row 4: Donut — document status breakdown (recharts PieChart)
   - All use `useQuery` + Skeleton fallback while loading

---

## Progress Tracking

**Overall Status:** Not Started — 0%

### Subtasks
| ID | Description | Status | Updated | Notes |
|----|-------------|--------|---------|-------|
| 21.1 | Install recharts | Not Started | 2026-04-21 | |
| 21.2 | Build GraphExplorer.tsx (filter bar + full-screen force graph) | Not Started | 2026-04-21 | |
| 21.3 | Build NodeInspector.tsx Sheet (3 tabs) | Not Started | 2026-04-21 | |
| 21.4 | Add GET /graph/expand endpoint in graphrag | Not Started | 2026-04-21 | |
| 21.5 | Add GET /analytics/time-series + /entities/top in graphgen | Not Started | 2026-04-21 | |
| 21.6 | Build analytics page (4 cards + 3 charts) | Not Started | 2026-04-21 | |
| 21.7 | Add Skeleton loading states for all data | Not Started | 2026-04-21 | |
| 21.8 | Delete old graph/page.tsx three-list view | Not Started | 2026-04-21 | |

## Progress Log
### 2026-04-21
- Task created. Graph page shows three text lists. Analytics is empty stub. Both blocking to demo readiness.

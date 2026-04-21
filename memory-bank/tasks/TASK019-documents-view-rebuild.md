# TASK019 — Documents View Rebuild

**Status:** Pending  
**Added:** 2026-04-21  
**Updated:** 2026-04-21  
**Phase:** D — UI Rebuild (requires TASK017 + TASK012 complete)

---

## Original Request
The `/documents` page is a minimal stub. No drag-and-drop upload. No table with real columns. No per-file ingestion progress. No document detail view. Rebuild it using Kibo Dropzone + shadcn DataTable.

---

## Thought Process
The documents page is the primary data-ingestion interface. It needs:
1. **Upload area** — Kibo Dropzone for drag-and-drop with per-file progress bars (connected to `/jobs/{id}/stream` SSE)
2. **Data table** — sortable, filterable table showing all documents
3. **Detail drawer** — Sheet component with document details, chunks, entities, and ingestion log tabs

TASK012 provides the `/jobs/{id}/stream` SSE endpoint that drives the progress bars. TASK013 provides the `duplicate` status. This task is the frontend consumer of those backend features.

### Table Columns
| Column | Type | Notes |
|---|---|---|
| title | text | Document filename |
| type | badge | MIME type → icon (PDF, DOCX, MD, etc.) |
| status | badge | queued / processing / done / error / duplicate |
| chunks | number | `chunk_count` from API |
| entities | number | `entity_count` from API |
| created | relative | `formatDistanceToNow(created_at)` |
| actions | buttons | View, Reprocess, Delete |

---

## Implementation Plan

1. **`components/documents/Dropzone.tsx`** — Kibo Dropzone wrapper:
   - Accepts: all document types (PDF, DOCX, MD, TXT, PPTX, XLSX, HTML, images)
   - On file drop: `POST /api/v1/documents` → receive `{job_id, doc_id}`
   - Per-file: open SSE connection to `/api/v1/jobs/{job_id}/stream`
   - Show progress bar with percentage + status text
   - On done: invalidate React Query cache for documents list
   - On error: Sonner toast with error message

2. **`components/documents/DocumentsTable.tsx`** — TanStack Table v8:
   ```bash
   pnpm --filter web add @tanstack/react-table date-fns
   ```
   - Columns as defined above
   - Client-side sort + filter (server pagination if >500 docs — defer)
   - Row selection checkbox for bulk delete
   - `useQuery` to fetch `GET /api/v1/documents`

3. **`components/documents/DocumentDetail.tsx`** — shadcn Sheet (right-side):
   - Tabs: Overview, Chunks, Entities, Ingestion Log
   - Overview: metadata (name, type, size, created, hash, status)
   - Chunks: scrollable list with text preview + position
   - Entities: badge cloud grouped by type
   - Ingestion Log: timeline of job events (from job registry)

4. **`app/(app)/documents/page.tsx`** — composed page:
   - `<Dropzone onUpload={...} />`
   - `<DocumentsTable onRowClick={(doc) => setSelected(doc)} />`
   - `<DocumentDetail doc={selected} open={!!selected} onClose={() => setSelected(null)} />`

5. **Install `@tanstack/react-query`** for data fetching:
   ```bash
   pnpm --filter web add @tanstack/react-query @tanstack/react-query-devtools
   ```
   Wrap in `QueryClientProvider` in `(app)/layout.tsx`.

---

## Progress Tracking

**Overall Status:** Not Started — 0%

### Subtasks
| ID | Description | Status | Updated | Notes |
|----|-------------|--------|---------|-------|
| 19.1 | Install @tanstack/react-table + react-query + date-fns | Not Started | 2026-04-21 | |
| 19.2 | Build Dropzone.tsx with SSE progress tracking | Not Started | 2026-04-21 | Depends on TASK012 |
| 19.3 | Build DocumentsTable.tsx (TanStack Table, 6 cols) | Not Started | 2026-04-21 | |
| 19.4 | Build DocumentDetail.tsx Sheet with 4 tabs | Not Started | 2026-04-21 | |
| 19.5 | Compose documents page | Not Started | 2026-04-21 | |
| 19.6 | Add QueryClientProvider to (app)/layout.tsx | Not Started | 2026-04-21 | |
| 19.7 | Delete old documents/page.tsx stub | Not Started | 2026-04-21 | |
| 19.8 | vitest component tests for Dropzone + Table | Not Started | 2026-04-21 | |

## Progress Log
### 2026-04-21
- Task created. Current documents page is a minimal stub. No table columns, no dropzone, no detail view.

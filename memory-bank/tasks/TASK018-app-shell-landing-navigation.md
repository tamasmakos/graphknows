# TASK018 — App Shell + Landing + Navigation

**Status:** Pending  
**Added:** 2026-04-21  
**Updated:** 2026-04-21  
**Phase:** D — UI Rebuild (requires TASK017 complete)

---

## Original Request
The app has no consistent shell: no sidebar, no top bar, no command palette, and no landing page. Users land directly on `/` which shows a bare stub. Move app pages to a route group `(app)/` with a persistent sidebar layout; build a landing page at `(marketing)/`.

---

## Implementation Plan

### Route Structure Refactor
```
apps/web/src/app/
  (marketing)/
    page.tsx           ← landing page (currently empty page.tsx)
    layout.tsx         ← marketing layout (just header + footer, no sidebar)
  (app)/
    layout.tsx         ← app shell layout (sidebar + topbar)
    chat/
      page.tsx
    documents/
      page.tsx
    graph/
      page.tsx
    analytics/
      page.tsx
  api/                 ← route handlers (unchanged)
  layout.tsx           ← root layout (ThemeProvider, Toaster, fonts)
  globals.css
```

### Components to Build

1. **`components/shell/Sidebar.tsx`** — shadcn Sidebar (collapsible, icon-only when collapsed):
   - Nav items: Chat, Documents, Graph, Analytics
   - Active state based on `usePathname()`
   - Footer: theme toggle, user avatar stub
   - Mobile: Sheet overlay via `useSidebar()` hook

2. **`components/shell/TopBar.tsx`** — breadcrumb + ⌘K trigger + theme toggle (desktop):
   - Uses shadcn `Breadcrumb` + `SidebarTrigger`
   - `ThemeToggle` button (sun/moon icon)
   - Positioned sticky top-0 with backdrop blur

3. **`components/shell/CommandPalette.tsx`** — shadcn `Command` + `Dialog`:
   - Triggered by `⌘K` / `Ctrl+K`
   - Search groups: Pages (Chat, Docs, Graph, Analytics), Recent Documents (from API)
   - Keyboard navigation: `↑↓` to move, `↵` to navigate

4. **`(app)/layout.tsx`** — wraps all app pages:
   ```tsx
   <SidebarProvider>
     <Sidebar />
     <main>
       <TopBar />
       {children}
     </main>
   </SidebarProvider>
   ```

5. **`(marketing)/page.tsx`** — simple landing:
   - Hero: "GraphKnows — Turn documents into knowledge" + two CTA buttons (Try Demo, View on GitHub)
   - Feature cards: 3 × shadcn Card (Ingest, Explore, Query)
   - Tech stack badge row
   - No auth needed — just redirect to `/chat`

---

## Progress Tracking

**Overall Status:** Not Started — 0%

### Subtasks
| ID | Description | Status | Updated | Notes |
|----|-------------|--------|---------|-------|
| 18.1 | Restructure routes into (app)/ and (marketing)/ groups | Not Started | 2026-04-21 | Move existing pages |
| 18.2 | Create (app)/layout.tsx with SidebarProvider | Not Started | 2026-04-21 | |
| 18.3 | Build Sidebar.tsx (shadcn Sidebar, collapsible) | Not Started | 2026-04-21 | |
| 18.4 | Build TopBar.tsx (breadcrumb + ⌘K trigger) | Not Started | 2026-04-21 | |
| 18.5 | Build CommandPalette.tsx (shadcn Command in Dialog) | Not Started | 2026-04-21 | |
| 18.6 | Build (marketing)/page.tsx landing | Not Started | 2026-04-21 | Simple, no auth |
| 18.7 | Mobile sidebar (Sheet) | Not Started | 2026-04-21 | |
| 18.8 | Route redirects (/ → /chat if logged in, or → landing) | Not Started | 2026-04-21 | |

## Progress Log
### 2026-04-21
- Task created. No shell exists. All pages are at top-level routes. No sidebar, no command palette.

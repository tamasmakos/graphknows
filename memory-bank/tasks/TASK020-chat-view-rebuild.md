# TASK020 — Chat View Rebuild

**Status:** Pending  
**Added:** 2026-04-21  
**Updated:** 2026-04-21  
**Phase:** D — UI Rebuild (requires TASK017 + TASK014 complete)

---

## Original Request
The chat page SSE works but uses no Kibo UI components, has no conversation persistence, no reasoning timeline, no citation hover-cards, and no conversation history sidebar. Rebuild using Kibo AI Input + AI Message components.

---

## Thought Process
The chat page already has the hardest part working: SSE streaming, message bubbles, citation tags, GraphVisualizer. The rebuild is about:
1. Replacing raw HTML/Tailwind with Kibo AI chat components
2. Adding conversation persistence (SQLite via Route Handler)
3. Rendering the new SSE event types from TASK014 (reasoning timeline, tool calls)
4. HoverCard citations showing the source chunk text

### Persistence: SQLite via Route Handler
No auth needed for MVP. Use `better-sqlite3` (sync, zero config) in Next.js Route Handlers:
- `conversations` table: `id, title, created_at, updated_at`
- `messages` table: `id, conversation_id, role, content, metadata (JSON), created_at`

This runs in Node.js (not edge runtime), which is fine for a self-hosted app.

### Left Rail: Conversation History
- Scrollable list of past conversations (from SQLite)
- New conversation button
- Click to load — replace main panel content
- Title auto-generated from first user message (LLM call, debounced, optional)

---

## Implementation Plan

1. **Install `better-sqlite3`** + types:
   ```bash
   pnpm --filter web add better-sqlite3
   pnpm --filter web add -D @types/better-sqlite3
   ```

2. **Create `app/api/conversations/route.ts`** — GET (list) + POST (create):
   ```typescript
   // GET /api/conversations → list conversations
   // POST /api/conversations → create new conversation, return {id, title}
   ```

3. **Create `app/api/conversations/[id]/messages/route.ts`** — GET (history) + POST (save message):
   ```typescript
   // GET /api/conversations/{id}/messages → list messages
   // POST /api/conversations/{id}/messages → save message, return {id}
   ```

4. **Create `lib/db.ts`** — SQLite singleton:
   ```typescript
   import Database from 'better-sqlite3'
   
   const db = new Database('./data/chat.db')
   db.exec(`CREATE TABLE IF NOT EXISTS conversations (...)`)
   db.exec(`CREATE TABLE IF NOT EXISTS messages (...)`)
   export default db
   ```

5. **Build `components/chat/ConversationHistory.tsx`**:
   - `useQuery` → `GET /api/conversations`
   - shadcn ScrollArea with conversation list items
   - New conversation button at top
   - Active conversation highlighted

6. **Build `components/chat/ChatView.tsx`** using Kibo AI components:
   - `<AIInput>` (Kibo) replacing the raw textarea
   - `<AIMessage>` (Kibo) for each message bubble (role: user/assistant)
   - Citation `[n]` markers → shadcn `<HoverCard>` showing chunk text + source
   - Reasoning steps from `reasoning` + `tool_call` events → Kibo `<Timeline>`
   - GraphVisualizer panel (existing, wired to `subgraph` SSE events)
   - Sonner toast on `error` SSE events

7. **SSE consumer update** — handle all 8 event types from TASK014:
   ```typescript
   switch (event.type) {
     case "reasoning":   setReasoningSteps(...)
     case "tool_call":   setToolCalls(...)
     case "tool_result": updateToolCall(...)
     case "token":       appendToCurrentMessage(...)
     case "citation":    addCitation(...)
     case "subgraph":    updateGraph(...)
     case "done":        finalizeMessage(); saveToSQLite(...)
     case "error":       toast.error(event.message)
   }
   ```

8. **Compose `app/(app)/chat/page.tsx`**:
   ```tsx
   <div className="flex h-full">
     <ConversationHistory />
     <ChatView conversationId={activeConversation} />
   </div>
   ```

---

## Progress Tracking

**Overall Status:** Not Started — 0%

### Subtasks
| ID | Description | Status | Updated | Notes |
|----|-------------|--------|---------|-------|
| 20.1 | Install better-sqlite3 + types | Not Started | 2026-04-21 | |
| 20.2 | Create lib/db.ts (SQLite singleton + migrations) | Not Started | 2026-04-21 | |
| 20.3 | Create /api/conversations Route Handler | Not Started | 2026-04-21 | |
| 20.4 | Create /api/conversations/[id]/messages Route Handler | Not Started | 2026-04-21 | |
| 20.5 | Build ConversationHistory.tsx | Not Started | 2026-04-21 | |
| 20.6 | Build ChatView.tsx with Kibo AI Input + AI Message | Not Started | 2026-04-21 | |
| 20.7 | Update SSE consumer for 8 event types | Not Started | 2026-04-21 | Depends on TASK014 |
| 20.8 | Citations → HoverCard | Not Started | 2026-04-21 | |
| 20.9 | Reasoning → Kibo Timeline | Not Started | 2026-04-21 | |
| 20.10 | Persist messages to SQLite after done event | Not Started | 2026-04-21 | |
| 20.11 | vitest tests for SSE consumer state machine | Not Started | 2026-04-21 | |

## Progress Log
### 2026-04-21
- Task created. Current chat page: works for basic streaming but no persistence, no Kibo components, no reasoning timeline.

# TASK024 — CONTRIBUTING Recipes + One-Command Dev + Architecture Docs

**Status:** Pending  
**Added:** 2026-04-21  
**Updated:** 2026-04-21  
**Phase:** F — Documentation & DX (final task; run after all others)

---

## Original Request
Developer experience is poor: `pnpm dev` doesn't start the full stack, the README port table is wrong, there are no "how to extend" recipes, and there's no architecture diagram. Fix all of it.

---

## Implementation Plan

### One-Command Dev

1. **Update root `package.json`**:
   ```json
   {
     "scripts": {
       "dev": "concurrently --kill-others-on-fail \"docker compose -f docker-compose.dev.yaml up neo4j\" \"turbo run dev\"",
       "dev:full": "docker compose -f docker-compose.dev.yaml up",
       "test": "turbo run test",
       "build": "turbo run build",
       "lint": "turbo run lint"
     }
   }
   ```

2. **Install `concurrently`**:
   ```bash
   pnpm add -w concurrently
   ```

3. **`turbo.json` `dev` pipeline** — ensure all services have a `dev` script:
   - `apps/web`: `next dev` (already there)
   - `services/graphgen`: `uvicorn src.main:app --reload --port 8020`
   - `services/graphrag`: `uvicorn src.main:app --reload --port 8010`
   - Python services need `package.json` `dev` scripts that call uv.

4. **`docker-compose.dev.yaml`** — should only run infrastructure (Neo4j, no app services); verify this is already correct.

### CONTRIBUTING.md — Extension Recipes

5. **Update `CONTRIBUTING.md`** with 4 extension recipes:

   **Recipe 1: Adding a new document parser**:
   - Create `services/graphgen/src/kg/parser/your_format.py`
   - Subclass `BaseParser`, implement `parse(path) -> ParsedDocument`
   - Auto-discovered on startup — no registration needed
   - Test: add fixture file to `tests/fixtures/`, add one test in `test_parsers.py`

   **Recipe 2: Adding a new graph node type**:
   - Add to `services/graphgen/src/kg/schema.py`: new label + properties
   - Add UNIQUE constraint + optional vector index to `kg/neo4j/schema_bootstrap.py`
   - Add extractor logic in the relevant pipeline step
   - Update `packages/types/src/graph.ts` to include the new type in the frontend

   **Recipe 3: Adding a new agent tool**:
   - Create function in `services/graphrag/src/agent/tools.py`
   - Decorate with `@llm_tool` (LlamaIndex)
   - Add to `tools` list in `workflow.py`
   - Write unit test with mock Neo4j

   **Recipe 4: Adding a new UI page/view**:
   - Create `apps/web/src/app/(app)/your-view/page.tsx`
   - Add nav item to `components/shell/Sidebar.tsx`
   - Use `@graphknows/ui` components; do not add raw style={{ }} attributes
   - Add vitest component test

### Architecture Docs

6. **Create `docs/ARCHITECTURE.md`**:
   - System overview diagram (Mermaid)
   - Component table: service, port, responsibility, key deps
   - Data flow: document ingest pipeline (file → parser → chunker → embedder → Neo4j)
   - Data flow: agent query pipeline (query → decompose → retrieve → rerank → synthesize → verify)
   - SSE event taxonomy table (all 8 types from TASK014)
   - Neo4j schema diagram (Document, Chunk, Entity, Relationship node/edge types)
   - Environment variables reference table

7. **Fix README.md**:
   - Correct port table (currently may be stale)
   - Add "Quick start" section (5 commands: clone, cp .env, docker compose up, navigate to localhost:3000)
   - Add screenshot placeholder (add real screenshots once UI is built in TASK017–021)
   - Remove any mentions of FalkorDB

### Final Memory Bank Refresh

8. After all other tasks complete, do a final memory bank update:
   - Mark all tasks complete in `_index.md`
   - Update `activeContext.md`: next focus is user feedback + performance tuning
   - Update `progress.md`: all systems working
   - Update `systemPatterns.md`: add any patterns discovered during implementation

---

## Progress Tracking

**Overall Status:** Not Started — 0%

### Subtasks
| ID | Description | Status | Updated | Notes |
|----|-------------|--------|---------|-------|
| 24.1 | Install concurrently at workspace root | Not Started | 2026-04-21 | |
| 24.2 | Update root package.json dev scripts | Not Started | 2026-04-21 | |
| 24.3 | Add dev script to Python service package.json files | Not Started | 2026-04-21 | |
| 24.4 | Verify turbo.json dev pipeline | Not Started | 2026-04-21 | |
| 24.5 | Write 4 extension recipes in CONTRIBUTING.md | Not Started | 2026-04-21 | |
| 24.6 | Create docs/ARCHITECTURE.md | Not Started | 2026-04-21 | Mermaid diagrams |
| 24.7 | Fix README.md (ports, quick start, remove FalkorDB) | Not Started | 2026-04-21 | |
| 24.8 | Final memory bank refresh | Not Started | 2026-04-21 | After all else done |

## Progress Log
### 2026-04-21
- Task created. Current state: pnpm dev only starts Next.js. No extension recipes. No ARCHITECTURE.md. README has stale info.

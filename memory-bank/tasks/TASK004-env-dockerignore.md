# TASK004 — Phase 1d: .dockerignore, .env.example, .gitignore

**Status:** Pending  
**Added:** 2026-04-20  
**Updated:** 2026-04-20

## Original Request
Create `.dockerignore` at the repo root (currently absent), create `.env.example` with the full namespaced env var convention, and update `.gitignore` for the new monorepo structure.

## Thought Process
`.dockerignore` is currently missing — confirmed by the third exploration agent. This means Docker builds send `.git`, `.venv`, `node_modules`, and `output/` to the build context, making builds slow and potentially leaking git history. This is a security concern (OWASP: sensitive data exposure via build context).

`.env.example` uses the `{SERVICE}_*` namespacing decided in the plan.

## Environment Variable Naming Convention
```
LLM_*          shared: provider, keys, model names
GRAPHGEN_*     ETL service
GRAPHRAG_*     agent service  
WEB_*          Next.js (NEXT_PUBLIC_* for client-side)
NEO4J_*        graph DB
POSTGRES_*     pgvector
LANGFUSE_*     observability
```

## Files to Create
- `.dockerignore`
- `.env.example`

## Files to Modify
- `.gitignore` — add `.turbo/`, `.next/`, `apps/**/node_modules`, `packages/**/node_modules`, `apps/**/dist`

## Progress Tracking

**Overall Status:** Not Started — 0%

### Subtasks
| ID | Description | Status | Updated | Notes |
|----|-------------|--------|---------|-------|
| 4.1 | Create .dockerignore | Not Started | 2026-04-20 | Security: prevent .git + .venv in context |
| 4.2 | Create .env.example (full convention) | Not Started | 2026-04-20 | |
| 4.3 | Update .gitignore for monorepo | Not Started | 2026-04-20 | |

## Progress Log
### 2026-04-20
- Task created during planning session.

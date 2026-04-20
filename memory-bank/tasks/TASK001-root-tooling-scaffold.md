# TASK001 — Phase 1a: Root Tooling Scaffold

**Status:** Pending  
**Added:** 2026-04-20  
**Updated:** 2026-04-20

## Original Request
Set up Turborepo + pnpm workspaces at the monorepo root so that `pnpm dev` becomes the single entrypoint for all services.

## Thought Process
The root `pyproject.toml` already has a `uv` workspace covering Python services. We need to layer pnpm workspaces on top for TypeScript packages (`apps/`, `packages/`). Turborepo orchestrates the task graph across both language ecosystems via `turbo.json`. Python services are NOT in the pnpm workspace — they run via Docker Compose. Turborepo's `dev` task for Python services will delegate to `docker compose`.

The key insight: language-native package managers own their lanes. pnpm owns TS, uv owns Python. Turborepo coordinates at the dev-loop level only.

## Implementation Plan
- [ ] Create `pnpm-workspace.yaml` declaring `apps/*` and `packages/*` globs
- [ ] Create root `package.json` with `turbo`, `pnpm` version constraints, and scripts: `dev`, `build`, `lint`, `typecheck`, `test`, `generate-types`
- [ ] Create `turbo.json` with task pipeline: `build`, `dev`, `lint`, `typecheck`, `test`
- [ ] Create placeholder `apps/web/package.json` (so pnpm workspace resolves)
- [ ] Create placeholder `packages/ui/package.json` and `packages/types/package.json`
- [ ] Create `scripts/generate-types.sh` skeleton (openapi-typescript invocation)
- [ ] Run `pnpm install` to generate `pnpm-lock.yaml`

## Files to Create
- `pnpm-workspace.yaml`
- `package.json` (root)
- `turbo.json`
- `apps/web/package.json` (placeholder, full in TASK008)
- `packages/ui/package.json`
- `packages/types/package.json`
- `scripts/generate-types.sh`

## Files to Modify
- None (root `pyproject.toml` already correct)

## Progress Tracking

**Overall Status:** Not Started — 0%

### Subtasks
| ID | Description | Status | Updated | Notes |
|----|-------------|--------|---------|-------|
| 1.1 | Create pnpm-workspace.yaml | Not Started | 2026-04-20 | |
| 1.2 | Create root package.json | Not Started | 2026-04-20 | |
| 1.3 | Create turbo.json | Not Started | 2026-04-20 | |
| 1.4 | Create package stubs (web, ui, types) | Not Started | 2026-04-20 | |
| 1.5 | Create scripts/generate-types.sh | Not Started | 2026-04-20 | |
| 1.6 | Run pnpm install | Not Started | 2026-04-20 | |

## Progress Log
### 2026-04-20
- Task created during planning session.

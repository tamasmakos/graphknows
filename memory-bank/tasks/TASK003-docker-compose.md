# TASK003 — Phase 1c: Docker Compose Cleanup + Dev Overlay

**Status:** Pending  
**Added:** 2026-04-20  
**Updated:** 2026-04-20

## Original Request
Clean up `docker-compose.yaml` to remove FalkorDB and unused services, add Neo4j properly with pinned tags and healthchecks, add the `web` service, pin all image tags. Create `docker-compose.dev.yaml` as a hot-reload overlay.

## Thought Process
The current compose is a mix of prod and dev concerns. The split strategy:
- `docker-compose.yaml` — declarative baseline: infra + built images, healthchecks, no bind mounts, no --reload.
- `docker-compose.dev.yaml` — overlay: adds bind mounts, --reload flags, `pnpm dev` for web.

Dev start: `docker compose -f docker-compose.yaml -f docker-compose.dev.yaml up`
This is wrapped by `pnpm dev` at the root.

Pinned images (as of plan date):
- `neo4j:5.21-community`
- `pgvector/pgvector:pg16.4`
- `postgres:16.4`
- `langfuse/langfuse:2.76`

## Implementation Plan
- [ ] Rewrite `docker-compose.yaml`:
  - Remove: `falkordb`, `workspace` dev container
  - Add: `neo4j:5.21-community` with auth, data volume, healthcheck
  - Update: pin all image tags
  - Update: graphgen port 8020, graphrag port 8010
  - Add: `web` service (port 3000) once Dockerfile exists
  - Add: healthchecks on graphgen, graphrag, web
  - Add: `condition: service_healthy` in depends_on
- [ ] Create `docker-compose.dev.yaml`:
  - graphgen: bind mount src, `--reload`
  - graphrag: bind mount src, `--reload`
  - web: bind mount app + packages, `pnpm dev` command

## Files to Create
- `docker-compose.dev.yaml`

## Files to Modify
- `docker-compose.yaml`

## Progress Tracking

**Overall Status:** Not Started — 0%

### Subtasks
| ID | Description | Status | Updated | Notes |
|----|-------------|--------|---------|-------|
| 3.1 | Remove falkordb + workspace from compose | Not Started | 2026-04-20 | |
| 3.2 | Add neo4j with pinned tag + healthcheck | Not Started | 2026-04-20 | |
| 3.3 | Pin all other image tags | Not Started | 2026-04-20 | |
| 3.4 | Add web service stub to compose | Not Started | 2026-04-20 | Full Dockerfile in TASK009 |
| 3.5 | Add service healthchecks + condition:service_healthy | Not Started | 2026-04-20 | |
| 3.6 | Create docker-compose.dev.yaml | Not Started | 2026-04-20 | |

## Progress Log
### 2026-04-20
- Task created during planning session.

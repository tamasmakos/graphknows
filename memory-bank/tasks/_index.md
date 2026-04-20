# Tasks Index

## In Progress
_(none yet — planning complete, implementation starting)_

## Pending

- [TASK001] Root Tooling Scaffold — Turborepo + pnpm-workspace.yaml + turbo.json + root package.json + package stubs
- [TASK002] Cleanup Dead Code — resolve 3 merge conflicts, delete frontend/, simulation/, life-log parsers, old scripts
- [TASK003] Docker Compose — strip falkordb, add neo4j with pinned tags, add web service, create docker-compose.dev.yaml
- [TASK004] .dockerignore / .env.example / .gitignore — missing dockerignore (security risk), full env convention
- [TASK005] DocumentParser Abstraction (graphgen Phase 2) — 8 format parsers, HeadingAwareChunker, REST endpoints
- [TASK006] Schema + Plugin System + Neo4j Uploader (graphgen Phase 3) — declarative schema.py, GraphPlugin ABC, neo4j/ replaces falkordb/
- [TASK007] AgentWorkflow Rewrite (graphrag Phase 4) — iterative agent loop, 4 tools, structured citations, SSE, Langfuse spans
- [TASK008] Next.js 15 Frontend (Phase 5) — apps/web, 4 views, BFF Route Handlers, packages/ui, packages/types
- [TASK009] Dockerfiles + CONTRIBUTING.md (Phase 6) — multi-stage, non-root, pinned, healthcheck; 3-recipe CONTRIBUTING

## Completed
_(none yet)_

## Abandoned
_(none)_

---

## Recommended Implementation Order

Phase 1 tasks should be done first (in order) as they set up the scaffolding all other work depends on:

```
TASK002 (cleanup)
  → TASK001 (tooling scaffold)
    → TASK004 (env files)
      → TASK003 (compose)
        → TASK006 (graphgen schema + neo4j)   ──┐
          → TASK005 (document parsers)          │
                                                ├─→ TASK009 (Dockerfiles)
        → TASK007 (graphrag agent)            ──┤
                                                │
        → TASK008 (frontend)                  ──┘
```

TASK002 must precede everything (fixes merge conflicts).
TASK001 must precede TASK008 (pnpm workspace needed by web scaffold).
TASK006 must precede TASK005 (schema.py needed by pipeline/core.py).
TASK006 and TASK007 can run in parallel.
TASK008 can start in parallel with TASK005/006/007 once TASK001 is done.
TASK009 (Dockerfiles) is best done last — depends on final service structure.

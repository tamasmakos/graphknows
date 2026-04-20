# TASK009 — Phase 6: Dockerfile Upgrades + CONTRIBUTING.md

**Status:** Pending  
**Added:** 2026-04-20  
**Updated:** 2026-04-20

## Original Request
Upgrade both Python service Dockerfiles to multi-stage, non-root, pinned base images with HEALTHCHECK. Create CONTRIBUTING.md documenting the three main extension points.

## Thought Process
This is the final phase but the Dockerfiles should be done alongside (or before) any deployment testing. The CONTRIBUTING.md is critical for the template's stated purpose — a developer should be able to extend the system without reading source code.

### Dockerfile Pattern (Python services)
```dockerfile
# Stage 1: builder (deps, model downloads)
FROM ghcr.io/astral-sh/uv:0.4-python3.11-bookworm-slim AS builder
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*
COPY pyproject.toml uv.lock ./
COPY services/<svc>/pyproject.toml ./services/<svc>/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --package <svc> --frozen --no-install-project --no-dev
# graphgen only: NLP models
RUN uv run python -m spacy download en_core_web_lg
RUN uv run python -c "import nltk; [nltk.download(p) for p in ['punkt','punkt_tab','stopwords']]"

# Stage 2: runtime (slim, non-root)
FROM python:3.11.10-slim-bookworm AS runtime
RUN groupadd -r app && useradd -r -g app -u 10001 app \
    && apt-get update && apt-get install -y --no-install-recommends curl tini \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY --from=builder --chown=app:app /app/.venv /app/.venv
COPY --chown=app:app services/<svc>/src /app/src
USER app
ENV PATH="/app/.venv/bin:$PATH" PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=20s --retries=3 \
  CMD curl -fsS http://127.0.0.1:8000/health || exit 1
ENTRYPOINT ["/usr/bin/tini","--"]
CMD ["uvicorn","src.main:app","--host","0.0.0.0","--port","8000"]
```

### Dockerfile Pattern (Next.js web)
- Base: `node:22.11-alpine3.20`
- Stage 1: deps install (`pnpm install --frozen-lockfile`)
- Stage 2: builder (`pnpm build`, Next.js `output: 'standalone'`)
- Stage 3: runner (alpine, non-root `nextjs` user, copy `.next/standalone` only)
- HEALTHCHECK on `/api/v1/health`

### CONTRIBUTING.md Extension Points
Three main sections with concrete interface examples:
1. **Adding a new document parser** — copy template, implement `BaseParser`, set `supported_extensions`.
2. **Adding a new node type** — create a `GraphPlugin` subclass in `plugins/`, call `schema.register_node(NodeSpec(...))` in `register()`.
3. **Adding a new agent tool** — subclass `AgentTool` in `agent/tools.py`, define `input_schema`, implement `async run()`.

## Implementation Plan
- [ ] Rewrite `services/graphgen/Dockerfile` — 2-stage pattern
- [ ] Rewrite `services/graphrag/Dockerfile` — 2-stage pattern
- [ ] Create `apps/web/Dockerfile` — 3-stage Next.js standalone
- [ ] Create `CONTRIBUTING.md` with 3 extension-point recipes
- [ ] Update `README.md` — 30-minute quickstart, architecture overview, links

## Progress Tracking

**Overall Status:** Not Started — 0%

### Subtasks
| ID | Description | Status | Updated | Notes |
|----|-------------|--------|---------|-------|
| 9.1 | graphgen Dockerfile — 2-stage | Not Started | 2026-04-20 | |
| 9.2 | graphrag Dockerfile — 2-stage | Not Started | 2026-04-20 | |
| 9.3 | apps/web Dockerfile — 3-stage standalone | Not Started | 2026-04-20 | |
| 9.4 | CONTRIBUTING.md (3 extension recipes) | Not Started | 2026-04-20 | |
| 9.5 | README.md rewrite | Not Started | 2026-04-20 | |

## Progress Log
### 2026-04-20
- Task created during planning session.

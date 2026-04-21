# TASK023 — Observability + Evals + Supply Chain

**Status:** Pending  
**Added:** 2026-04-21  
**Updated:** 2026-04-21  
**Phase:** E — Quality

---

## Original Request
Structured logging with correlation IDs, a Ragas evaluation harness, Dependabot configuration, spaCy removal from Dockerfiles, and SHA-pinned base images. These are hygiene + quality items that should be done before the system is shown to users.

---

## Implementation Plan

### Structured Logging

1. **Install `structlog`** in both Python services:
   ```bash
   uv add structlog --workspace services/graphgen
   uv add structlog --workspace services/graphrag
   ```

2. **Configure structlog in `main.py`** of each service:
   ```python
   import structlog
   structlog.configure(
       processors=[
           structlog.contextvars.merge_contextvars,
           structlog.processors.TimeStamper(fmt="iso"),
           structlog.stdlib.add_log_level,
           structlog.processors.JSONRenderer(),
       ],
       wrapper_class=structlog.BoundLogger,
       context_class=dict,
       logger_factory=structlog.PrintLoggerFactory(),
   )
   ```

3. **Correlation ID middleware** in both services:
   ```python
   @app.middleware("http")
   async def add_correlation_id(request, call_next):
       correlation_id = request.headers.get("X-Correlation-ID", str(uuid4()))
       structlog.contextvars.bind_contextvars(correlation_id=correlation_id)
       response = await call_next(request)
       response.headers["X-Correlation-ID"] = correlation_id
       return response
   ```

4. **Replace all `logging.*` and `print()` calls** with `structlog.get_logger()` calls.

### Ragas Evaluation Harness

5. **Create `evals/` directory** at repo root:
   ```
   evals/
     conftest.py
     fixtures/
       qa_pairs.json        ← 20 ground-truth question-answer pairs
       documents/           ← source documents for the QA pairs
     test_ragas.py          ← pytest eval suite
   ```

6. **`evals/test_ragas.py`**:
   ```python
   from ragas import evaluate
   from ragas.metrics import faithfulness, answer_relevancy, context_precision
   
   @pytest.mark.eval
   async def test_ragas_metrics_above_threshold(live_agent):
       dataset = load_qa_fixtures()
       results = await evaluate(dataset, metrics=[faithfulness, answer_relevancy, context_precision])
       assert results["faithfulness"] >= 0.80
       assert results["answer_relevancy"] >= 0.75
   ```
   Marked with `@pytest.mark.eval` so CI runs them separately (they require live services + LLM credits).

### Dependabot

7. **Create `.github/dependabot.yml`**:
   ```yaml
   version: 2
   updates:
     - package-ecosystem: npm
       directory: /
       schedule: { interval: weekly }
       groups:
         dev-dependencies: { patterns: ["@types/*", "eslint*", "vitest*"] }
     - package-ecosystem: pip
       directory: /services/graphgen
       schedule: { interval: weekly }
     - package-ecosystem: pip
       directory: /services/graphrag
       schedule: { interval: weekly }
     - package-ecosystem: docker
       directory: /
       schedule: { interval: weekly }
     - package-ecosystem: github-actions
       directory: /
       schedule: { interval: weekly }
   ```

### Docker Image Hygiene

8. **Remove spaCy from `services/graphgen/Dockerfile`**:
   - Find `RUN python -m spacy download en_core_web_lg` or similar — remove it.
   - Remove `spacy` from `requirements.txt` / `pyproject.toml` if present.
   - Saves ~40–80MB from image.

9. **Pin Node.js SHA in `apps/web/Dockerfile`**:
   - Replace `FROM node:18-alpine` with a digest-pinned version:
     `FROM node:20-alpine@sha256:<digest>`
   - Use `docker pull node:20-alpine && docker inspect node:20-alpine --format '{{index .RepoDigests 0}}'` to get current digest.

10. **Remove `falkordb` from `pyproject.toml`** of both services (if not already done in TASK010).

---

## Progress Tracking

**Overall Status:** Not Started — 0%

### Subtasks
| ID | Description | Status | Updated | Notes |
|----|-------------|--------|---------|-------|
| 23.1 | Add structlog to both services | Not Started | 2026-04-21 | |
| 23.2 | Configure structlog JSON renderer in main.py | Not Started | 2026-04-21 | |
| 23.3 | Add correlation ID middleware | Not Started | 2026-04-21 | |
| 23.4 | Replace logging.* + print() with structlog | Not Started | 2026-04-21 | |
| 23.5 | Create evals/ directory + 20 QA fixtures | Not Started | 2026-04-21 | |
| 23.6 | Create evals/test_ragas.py with 3 metrics | Not Started | 2026-04-21 | |
| 23.7 | Create .github/dependabot.yml | Not Started | 2026-04-21 | 5 ecosystems |
| 23.8 | Remove spaCy from graphgen Dockerfile | Not Started | 2026-04-21 | ~40MB savings |
| 23.9 | Pin Node.js SHA in apps/web/Dockerfile | Not Started | 2026-04-21 | |
| 23.10 | Confirm falkordb removed from pyproject.toml | Not Started | 2026-04-21 | May be TASK010 |

## Progress Log
### 2026-04-21
- Task created. Confirmed: structlog not configured, no Ragas harness, no dependabot.yml, spaCy still in Dockerfile, Node.js not SHA-pinned.

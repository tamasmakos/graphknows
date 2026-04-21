# TASK022 — CI/CD + Tests

**Status:** Pending  
**Added:** 2026-04-21  
**Updated:** 2026-04-21  
**Phase:** E — Quality (can be run in parallel with Phase D)

---

## Original Request
`.github/workflows/` is empty. There are zero test files despite pytest and vitest being declared. No quality gate exists. Add a full CI pipeline and a baseline test suite covering the 70/20/10 pyramid.

---

## Target Coverage
- **Unit (70%)**: fast, in-process, no external services — pytest for Python, vitest for TypeScript
- **Integration (20%)**: real Neo4j via testcontainers — pytest with `testcontainers-neo4j`
- **E2E (10%)**: critical paths via Playwright

---

## Implementation Plan

### CI Workflow: `.github/workflows/ci.yml`

```yaml
on: [push, pull_request]
jobs:
  lint-typecheck:   # pnpm lint + tsc --noEmit + ruff + mypy
  test-python:      # pytest graphgen + pytest graphrag
  test-frontend:    # pnpm vitest run
  build:            # pnpm build (turbo)
  docker:           # docker buildx bake (validate only, no push on PRs)
```

All jobs use `actions/cache` for pnpm store + pip/uv cache + Docker layer cache.

### Python Unit Tests

**`services/graphgen/tests/unit/`**:
- `test_parsers.py`: one test per parser format (fixture files in `tests/fixtures/`)
  - `test_markdown_parser_creates_heading_aware_chunks()`
  - `test_pdf_parser_extracts_text()`
  - `test_image_parser_calls_vision_model(mock_llm)`
- `test_pipeline.py`:
  - `test_pipeline_calls_progress_callback(mock_neo4j)`
  - `test_pipeline_batches_embeddings_in_chunks_of_64(mock_embeddings)`
- `test_jobs.py`:
  - `test_job_registry_creates_and_retrieves_job()`
  - `test_queue_raises_on_full()`
  - `test_dedup_returns_existing_job_for_same_hash()`

**`services/graphrag/tests/unit/`**:
- `test_workflow.py`:
  - `test_decompose_returns_subquestions(mock_llm)`
  - `test_stream_emits_all_8_event_types(mock_llm, mock_neo4j)`
  - `test_verifier_flags_unsupported_claim(mock_llm)`
- `test_retrieval.py`:
  - `test_rrf_merges_correctly()`
  - `test_reranker_returns_top_k(mock_cross_encoder)`

### Python Integration Tests

**`services/graphgen/tests/integration/test_pipeline_e2e.py`**:
- Uses `testcontainers.neo4j.Neo4jContainer`
- Fixture: uploads `tests/fixtures/sample.md` to real in-memory Neo4j
- Asserts: `Document` node created, `Chunk` nodes created, vector index populated
- TDD: red first, then implement

**`services/graphrag/tests/integration/test_agent_e2e.py`**:
- Uses `testcontainers.neo4j.Neo4jContainer` pre-seeded with fixture data
- Sends a query → asserts `AgentResponse` has non-empty `answer` + `citations`

### Frontend Tests

**`apps/web/src/`**:
- `__tests__/chat-sse-consumer.test.ts`: vitest unit — SSE state machine handles all 8 event types
- `__tests__/DocumentsTable.test.tsx`: vitest + testing-library — renders correctly with mock data
- `__tests__/ConversationHistory.test.tsx`: vitest — creates + loads conversations

**`e2e/`** (Playwright):
- `chat.spec.ts`: open app → type query → verify streaming response appears
- `documents.spec.ts`: upload fixture file → verify job status → verify document in table

### Test Infrastructure
- `services/graphgen/tests/conftest.py`: shared `mock_neo4j`, `mock_llm`, `mock_embeddings` fixtures
- `services/graphrag/tests/conftest.py`: shared `mock_neo4j_with_data`, `mock_llm` fixtures
- `apps/web/vitest.config.ts`: configure jsdom, testing-library
- `playwright.config.ts` at root: configure `baseURL`, `webServer` (starts compose)

---

## Progress Tracking

**Overall Status:** Not Started — 0%

### Subtasks
| ID | Description | Status | Updated | Notes |
|----|-------------|--------|---------|-------|
| 22.1 | Create .github/workflows/ci.yml | Not Started | 2026-04-21 | 5 jobs |
| 22.2 | Create graphgen unit tests (parsers, pipeline, jobs) | Not Started | 2026-04-21 | ~10 tests |
| 22.3 | Create graphrag unit tests (workflow, retrieval) | Not Started | 2026-04-21 | ~8 tests |
| 22.4 | Create graphgen integration test (pipeline e2e) | Not Started | 2026-04-21 | testcontainers |
| 22.5 | Create graphrag integration test (agent e2e) | Not Started | 2026-04-21 | testcontainers |
| 22.6 | Create frontend vitest tests | Not Started | 2026-04-21 | 3 test files |
| 22.7 | Create Playwright e2e tests | Not Started | 2026-04-21 | 2 specs |
| 22.8 | Create shared test conftest.py fixtures | Not Started | 2026-04-21 | |
| 22.9 | Configure vitest.config.ts for web | Not Started | 2026-04-21 | |

## Progress Log
### 2026-04-21
- Task created. Confirmed: zero test files exist. CI workflows exist but are empty.

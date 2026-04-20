---
description: "Use when writing any production code, tests, new features, bug fixes, or refactoring. Enforces Test-Driven Development: Red → Green → Refactor cycle, Uncle Bob's Three Laws, FIRST principles, and the Testing Pyramid across Python (pytest) and TypeScript (vitest/Jest)."
applyTo: "**/*.py, **/tests/**, **/*.test.ts, **/*.test.tsx, **/*.spec.ts, **/*.spec.tsx, **/*.test.js"
---
# Test-Driven Development (TDD)

TDD is non-negotiable on this project. Every code change follows the Red → Green → Refactor cycle.

## Uncle Bob's Three Laws (The Non-Negotiable Core)

1. **Do not write production code** unless it is to make a failing unit test pass.
2. **Do not write more of a unit test** than is sufficient to fail — compilation errors count as failures.
3. **Do not write more production code** than is sufficient to pass the one failing test.

Never generate or suggest production code without a corresponding test written first. If asked to implement a feature, write the test first, then the minimal implementation.

## Red → Green → Refactor

```
RED    — Write a failing test that describes the desired behavior
GREEN  — Write the minimum production code to make it pass (fake it if needed)
REFACTOR — Clean up code and tests while all tests stay green
```

Never refactor on red. Never skip the refactor step — technical debt compounds.

---

## The Testing Pyramid

Target ratio: **70% unit / 20% integration / 10% E2E**

| Layer | What it tests | Speed | Tools |
|---|---|---|---|
| **Unit** | Single function / class in isolation; all deps mocked | < 2 s total | pytest, vitest |
| **Integration** | Multiple components communicating; real DB in test container | seconds | pytest + httpx, testcontainers |
| **E2E** | Critical user journeys through the running system | minutes | playwright, pytest |

The inverted pyramid ("ice cream cone") — many E2E, few unit tests — is an antipattern. Do not add E2E tests for paths already covered by unit + integration tests.

---

## FIRST Principles

Every unit test must be:

- **Fast** — runs in milliseconds. If a test needs a real network, DB, or LLM call, it is not a unit test.
- **Isolated** — no shared mutable state between tests; each arranges its own world.
- **Repeatable** — same result in any environment; no clock, no randomness, no network.
- **Self-validating** — the test asserts its own pass/fail; no human interpretation needed.
- **Timely** — written before or alongside production code, never weeks later.

---

## Writing Tests: Core Rules

### Arrange–Act–Assert (AAA)

Every test has exactly three sections. Never interleave them.

```python
def test_extract_entities_returns_person_nodes():
    # Arrange
    text = "Alice met Bob at OpenAI."
    extractor = EntityExtractor(model=MockLLM())

    # Act
    nodes = extractor.extract(text)

    # Assert
    names = {n.label for n in nodes if n.type == "PERSON"}
    assert names == {"Alice", "Bob"}
```

### Test naming — tests are specifications

Names must be sentences that describe behavior. A failing test name must tell you what regressed.

```python
# Good
def test_upload_returns_422_when_file_is_empty(): ...
def test_community_detection_groups_connected_nodes(): ...

# Bad
def test_upload(): ...
def test1(): ...
```

### One reason to fail per test

A test can have multiple `assert` statements as long as they all verify the same behavior. Do not test two unrelated behaviors in one test.

### Test positive and negative cases

```python
def test_graph_node_accepts_valid_label(): ...
def test_graph_node_rejects_empty_label(): ...
```

---

## Python (pytest) — Stack-Specific Rules

**Run tests with uv:**
```bash
uv run pytest                          # all tests
uv run pytest tests/unit/              # unit tests only
uv run pytest -x -q                    # stop on first failure, quiet
uv run pytest --cov=src --cov-report=term-missing
```

**`asyncio_mode = "auto"` is already configured** — do not add `@pytest.mark.asyncio` decorators.

### FastAPI endpoint tests (integration layer)

Use `httpx.AsyncClient` with `ASGITransport` — never spin up a real server:

```python
import pytest
from httpx import AsyncClient, ASGITransport
from src.main import app

@pytest.fixture
async def client():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c

async def test_health_returns_200(client):
    response = await client.get("/health")
    assert response.status_code == 200
```

### Mocking rules

- **Mock at the boundary**: mock LLM clients, FalkorDB clients, Postgres connections — never mock internal application logic.
- Use `unittest.mock.AsyncMock` for async callables; `MagicMock` for sync.
- Prefer `pytest.fixture` + `monkeypatch` over module-level patches.

```python
@pytest.fixture
def mock_llm(monkeypatch):
    mock = AsyncMock(return_value="mocked response")
    monkeypatch.setattr("src.kg.llm.LLMClient.complete", mock)
    return mock
```

### Shared fixtures go in `conftest.py`

Place `conftest.py` at the test directory root. Fixtures used across multiple test files live there. Do not duplicate fixture logic across test files.

### Test directory layout

```
services/graphgen/
  src/
  tests/
    conftest.py
    unit/
      test_extraction.py
      test_community_detection.py
    integration/
      test_pipeline_api.py
```

---

## TypeScript / Next.js — Stack-Specific Rules

*(Applies when the `apps/web/` frontend is scaffolded — TASK008)*

Use **vitest** for unit and component tests; **Playwright** for E2E.

**Run tests:**
```bash
pnpm test              # vitest watch
pnpm test --run        # CI mode (no watch)
pnpm playwright test   # E2E
```

### Component tests

```typescript
import { render, screen } from '@testing-library/react'
import { describe, it, expect } from 'vitest'
import { GraphView } from '@/components/GraphView'

describe('GraphView', () => {
  it('renders node count label', () => {
    render(<GraphView nodes={[{ id: '1', label: 'Alice' }]} />)
    expect(screen.getByText('1 node')).toBeInTheDocument()
  })
})
```

### Mocking rules (TypeScript)

- Mock API clients and external services with `vi.mock()`; never mock React hooks or component internals.
- Use `msw` (Mock Service Worker) for HTTP boundary mocks in integration tests.

---

## What NOT to Do

- Do not write production code first and add tests afterward.
- Do not mock internal application logic — only mock at the infrastructure boundary.
- Do not write tests that require live infrastructure (FalkorDB, Postgres, LLMs) in the unit layer.
- Do not leave `test_*` functions that call real services in the pytest-collected test suite — those belong in `smoke_test.py` or a dedicated `e2e/` folder excluded from the default pytest run.
- Do not duplicate tests that cover the same behavior — delete redundant tests.
- Tests are first-class code: apply the same naming, readability, and refactoring discipline as production code.

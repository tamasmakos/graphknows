# MCP Bug Fix Plan — Round 2

> Generated from live file inspection of `/root/graphknows`.
> Three confirmed root-cause bugs; fixes are precise and non-speculative.

---

## Shared-File Analysis (read this first)

| File | Bug 1 | Bug 2 | Bug 3 |
|---|---|---|---|
| `graphknows/builders/stores.py` | ✅ primary edit | — | — |
| `graphknows/memory/workflows/corpus_ingest.py` | ✅ primary edit | — | — |
| `graphknows/recall.py` | — | ✅ primary edit | — |
| `graphknows/retrieval/stores/embedder.py` | — | — | ✅ read-only (no change needed) |
| `graphknows/mcp/stdio_server.py` | — | — | ✅ primary edit |
| `graphknows/mcp/_app.py` | — | — | ✅ possible edit (lifespan hook) |

**Conclusion: No files are shared between any two bugs. All three fixes can be developed and reviewed in parallel by different contributors.** Merge order is irrelevant from a conflict standpoint.

---

## Bug 1 — `corpus_ingest`: asyncpg "attached to a different loop"

### Root cause (confirmed)

`build_corpus_workflow()` in `graphknows/builders/stores.py` (lines 96–116) calls
`_run(pg.connect())` at line 105. `_run` (from `graphknows/memory/ingestor.py` line 36)
submits the coroutine to `_BG_LOOP` — a daemon thread's private event loop. The
`PgVectorStore`'s asyncpg connection pool is therefore bound to `_BG_LOOP`.

`CorpusIngestWorkflow.run()` (corpus_ingest.py line 65) is `async` and runs on the
MCP server's anyio loop. It calls `STMService.ingest()` → `_STMIngestHandler.ingest_text()`
which calls `_run(self._pg.write_stm_chunk(...))` (stm/ingest.py line 206–224). This
re-uses the same `_BG_LOOP` submission, but asyncpg internally compares the pool's
creation loop with the current running loop and raises **"attached to a different loop"**
because the pool was initialized on `_BG_LOOP` while `write_stm_chunk` uses the same
`_BG_LOOP` for execution — but the corpus workflow's `await flusher.flush(...)` at line
133 runs asyncpg operations directly on the anyio loop without `_run()`, causing the
cross-loop conflict there.

### Precise fix

**File: `graphknows/builders/stores.py`** — lines 96–116

Remove the `_run(pg.connect())` call. The `pg` store will be connected lazily or
explicitly inside `CorpusIngestWorkflow.run()` on the correct loop.

```python
# BEFORE (line 99, 105):
from graphknows.memory.ingestor import _run
...
_run(pg.connect())   # ← DELETE THIS LINE

# AFTER — remove the import of _run and the connect call entirely:
def build_corpus_workflow(settings: GraphKnowsSettings) -> Any:
    """Build a CorpusIngestWorkflow from settings."""
    from graphknows.memory.extractor import GLiNER2EntityExtractor
    from graphknows.memory.workflows.corpus_ingest import CorpusIngestWorkflow
    from graphknows.retrieval.stores.embedder import embed_one

    mg = build_memgraph_store(settings)
    pg = build_pgvector_store(settings)
    # DO NOT call pg.connect() here — pool must be created on the calling loop
    neo = build_neo4j_store(settings)
    extractor = GLiNER2EntityExtractor()
    key = settings.openrouter_api_key or ""
    return CorpusIngestWorkflow(
        mg_store=mg,
        pg_store=pg,
        neo_store=neo,
        extractor=extractor,
        embed_fn=embed_one,
        openrouter_key=key,
    )
```

**File: `graphknows/memory/workflows/corpus_ingest.py`** — `run()` method, line 65

Add `await self._pg.connect()` at the top of `run()`, guarded against double-connect:

```python
async def run(self, request: CorpusIngestRequest) -> CorpusIngestResult:
    # Ensure the asyncpg pool is created on THIS event loop (not _BG_LOOP)
    if not getattr(self._pg, '_pool', None):
        await self._pg.connect()

    from graphknows.memory.flush import FlushOptions, SessionFlusher
    from graphknows.memory.stm.service import STMService
    ...  # rest unchanged
```

> **Why this guard:** `PgVectorStore` stores its pool as `self._pool` (confirmed at
> `retrieval/stores/pgvector.py` line 107 — `async def connect(self)`). Checking
> `getattr(self._pg, '_pool', None)` is safe because `build_pgvector_store` returns
> a fresh `PgVectorStore(dsn=...)` with `_pool = None` until `connect()` is awaited.
> If the attribute name changes, use a `try/except` or add a `is_connected` property
> to `PgVectorStore`.

**Alternative (simpler, lower risk):** Move the `await pg.connect()` call into the
MCP tool in `graphknows/mcp/tools/corpus.py` before `build_corpus_workflow`:

```python
# corpus.py lines 44–50 — ALTERNATIVE approach
s = get_settings()
workflow = build_corpus_workflow(s)
await workflow._pg.connect()   # ensure pool on MCP event loop
request = CorpusIngestRequest(...)
result = await workflow.run(request)
```

**Recommended:** Use the `CorpusIngestWorkflow.run()` approach (option 1) so
the workflow is self-contained and doesn't rely on callers to manage lifecycle.

### Test strategy

- **`tests/test_corpus_ingest.py`** (create or extend):
  - Mock `PgVectorStore.connect` as an async coroutine spy.
  - Assert it is called exactly once during `workflow.run()`.
  - Assert it is NOT called inside `build_corpus_workflow()`.
  - Regression test: patch `asyncio.get_event_loop` to confirm the pool is
    created on the same loop as `run()`.
- **Integration test** (if CI has Postgres): run `corpus_ingest` with a 2-doc
  payload and assert `errors == []` and `chunks_written > 0`.

### Risks and edge cases

| Risk | Mitigation |
|---|---|
| `_pool` attribute name changes in `PgVectorStore` | Add `is_connected: bool` property to `PgVectorStore`; use it as the guard |
| Double-`connect()` if `run()` called twice on same workflow | The `_pool` guard prevents double-init |
| `build_memory_ingestor()` at line 79 also calls `_run(pg.connect())` — same pattern | **Not changed here** — `MemoryIngestor` uses `_BG_LOOP` consistently for all its ops, so it is self-consistent. Do not change `build_memory_ingestor`. |
| If `pg.connect()` raises, Phase 1 will fail with a clear exception | Acceptable; `run()` already catches per-doc exceptions and appends to `errors` |

---

## Bug 2 — `recall_by_topic`: `entities: []` and `score: 0.0`

### Root cause (confirmed)

`recall_by_topic()` in `graphknows/recall.py` lines 204–221. The Cypher at line 206 is:

```cypher
MATCH (e:ENTITY)
WHERE toLower(e.name) CONTAINS token
```

Live schema: `:IN_TOPIC` edges exist only on `ENTITY_CONCEPT` nodes (10 total), not
on the 176 `ENTITY` nodes. The initial `MATCH (e:ENTITY)` never finds nodes that have
`:IN_TOPIC` edges, so `matched_names` is populated with ENTITY names that are useless
for the `OPTIONAL MATCH (e2)-[:IN_TOPIC]->(t)` join. Result: `entities: []`, `score: 0`.

### Precise fix

**File: `graphknows/recall.py`** — lines 204–221

Replace the `cypher` string assignment entirely:

```python
# BEFORE (lines 204–221):
    cypher = """
    UNWIND $tokens AS token
    MATCH (e:ENTITY)
    WHERE toLower(e.name) CONTAINS token
    WITH collect(DISTINCT e.name) AS matched_names
    MATCH (t:TOPIC)
    OPTIONAL MATCH (e2)-[:IN_TOPIC]->(t)
    WHERE e2.name IN matched_names
    WITH t,
         collect(DISTINCT e2.name) AS matched_entities,
         count(DISTINCT e2) AS overlap
    ORDER BY overlap DESC, t.entity_count DESC
    LIMIT $top_k
    RETURN t.community_id AS topic_id,
           t.summary AS summary,
           matched_entities AS entities,
           overlap AS score
    """

# AFTER:
    cypher = """
    UNWIND $tokens AS token
    MATCH (e)
    WHERE (e:ENTITY OR e:ENTITY_CONCEPT) AND toLower(e.name) CONTAINS token
    WITH collect(DISTINCT e.name) AS matched_names
    MATCH (t:TOPIC)
    OPTIONAL MATCH (e2)-[:IN_TOPIC]->(t)
    WHERE e2.name IN matched_names
    WITH t,
         collect(DISTINCT e2.name) AS matched_entities,
         count(DISTINCT e2) AS overlap
    ORDER BY overlap DESC, t.entity_count DESC
    LIMIT $top_k
    RETURN t.community_id AS topic_id,
           t.summary AS summary,
           matched_entities AS entities,
           overlap AS score
    """
```

**Diff summary:** Change line 206 from `MATCH (e:ENTITY)` + `WHERE toLower(e.name) CONTAINS token` to `MATCH (e)` + `WHERE (e:ENTITY OR e:ENTITY_CONCEPT) AND toLower(e.name) CONTAINS token`. Everything else stays identical.

### Test strategy

- **`tests/test_recall.py`** (create or extend):
  - Unit test with a mock Neo4j driver that returns rows with both `ENTITY` and
    `ENTITY_CONCEPT` names in `matched_names`.
  - Assert that `entities` is non-empty and `score > 0` when `ENTITY_CONCEPT` nodes
    match the query tokens.
  - Snapshot test: given a fixed list of tokens, assert the Cypher string contains
    `(e:ENTITY OR e:ENTITY_CONCEPT)` (simple string assertion guards against regression).
- **Integration test** (if CI has Neo4j with data): call `recall_by_topic(driver, "memory")`
  and assert `len(hits) > 0` and `hits[0].score > 0`.

### Risks and edge cases

| Risk | Mitigation |
|---|---|
| `ENTITY_CONCEPT` nodes with `:IN_TOPIC` edges but no name that matches query tokens → score still 0 | This is correct behaviour; fix only widens the candidate set |
| Performance: `MATCH (e)` without label is a full node scan if no index | Add index hint or keep `(e:ENTITY OR e:ENTITY_CONCEPT)` — Neo4j 5+ optimizes label unions via index intersection |
| Future schema adds a third label needing `:IN_TOPIC` traversal | Centralise the label set in a constant at top of `recall.py`, e.g. `_ENTITY_LABELS = ["ENTITY", "ENTITY_CONCEPT"]` |
| `WHERE e2.name IN matched_names` still only matches by name, not by node identity | Acceptable given current schema; if entity names are not unique, use `elementId` join |

---

## Bug 3 — `ltm_query`: cold-start 60s timeout

### Root cause (confirmed)

`embed_one()` in `graphknows/retrieval/stores/embedder.py` (lines 74–88) makes an HTTP
call to OpenRouter on every invocation — there is **no local model load** and therefore
no cold-start model loading delay in this codebase. The bug is not a local sentence-
transformer model. Instead the 60s timeout is caused by:

1. The OpenRouter HTTP call timeout is 30s (line 60: `timeout=30`).
2. On a cold server, the asyncpg pool is not yet connected when `ltm_query` runs,
   adding latency.
3. The embedding call runs synchronously inside `run_in_executor` (wherever `embed_one`
   is called in the retrieval path), but if the executor queue is saturated or the
   OpenRouter API is slow, cumulative latency can hit the MCP client's 60s limit.

**The fix direction in the task brief (warm a local model) does not apply** — this
codebase uses an HTTP API (OpenRouter), not a local sentence-transformer. The correct
fix is to:

a. **Pre-warm the asyncpg connection pool** for the LTM retrieval path at server
   startup, so the first `ltm_query` doesn't pay connection setup latency.
b. **Make `embed_one` timeout visible and surfaceable** so cold OpenRouter calls
   fail fast rather than silently consuming the MCP timeout budget.

### Precise fix

**File: `graphknows/mcp/stdio_server.py`** — `main()` at line 132

The `main()` function calls `asyncio.run(app.run_stdio_async())`. FastMCP does not
expose a lifespan hook in its public API as of the current version (no `lifespan=`
kwarg in `FastMCP(...)` call — confirmed in `_app.py`). The startup warmup must be
injected either:

**Option A — Warmup task in `main()` before `app.run_stdio_async()`:**

```python
# graphknows/mcp/stdio_server.py — replace main() (lines 132–138)

async def _startup_warmup() -> None:
    """Pre-warm connections and validate embedder before first tool call."""
    import asyncio
    import logging
    log = logging.getLogger("graphknows.startup")

    # 1. Probe the embedder (validates OPENROUTER_API_KEY and API reachability)
    try:
        from graphknows.retrieval.stores.embedder import embed_one
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, embed_one, "warmup")
        log.warning("Embedder warmup OK")
    except Exception as exc:
        log.warning("Embedder warmup failed (non-fatal): %s", exc)

    # 2. Pre-connect the LTM pgvector store
    try:
        from graphknows.mcp._state import get_pg_store
        await get_pg_store()
        log.warning("pgvector warmup OK")
    except Exception as exc:
        log.warning("pgvector warmup failed (non-fatal): %s", exc)


async def _run_server() -> None:
    await _startup_warmup()
    await app.run_stdio_async()


def main() -> None:
    """Run the graphknows MCP stdio server."""
    asyncio.run(_run_server())
```

**Option B — FastMCP lifespan (if supported by installed version):**

Check if installed `mcp` package supports `FastMCP(lifespan=...)`. If yes, pass an
async context manager. If not, use Option A (safer, no version dependency).

**Additional fix — `graphknows/retrieval/stores/embedder.py`** line 56:

Increase the httpx timeout or make it configurable so slow OpenRouter responses
don't silently eat into the MCP 60s budget:

```python
# BEFORE line 60:
        timeout=30,

# AFTER:
        timeout=httpx.Timeout(connect=10.0, read=45.0, write=10.0, pool=5.0),
```

This surfaces a clear `httpx.ReadTimeout` at 45s rather than letting the call hang
and trigger the MCP client's opaque 60s cutoff.

### Test strategy

- **`tests/test_startup_warmup.py`** (create):
  - Mock `embed_one` and `get_pg_store` as async-safe mocks.
  - Call `_startup_warmup()` and assert both were called exactly once.
  - Assert exceptions in warmup do not propagate (non-fatal).
- **`tests/test_embedder.py`** (create or extend):
  - Assert `embed_one` raises `httpx.ReadTimeout` if the HTTP call exceeds timeout.
  - Assert `embed_one` raises `RuntimeError` if `OPENROUTER_API_KEY` is unset.
- **Integration/timing test**: time the second `ltm_query` call (after warmup) and
  assert it completes in < 5s (excluding OpenRouter latency).

### Risks and edge cases

| Risk | Mitigation |
|---|---|
| `_startup_warmup` adds latency to server startup (OpenRouter call) | Warmup runs concurrently; if it fails, server still starts (non-fatal) |
| `get_pg_store` may not be idempotent (double-connect) | Confirmed: `_state.py` uses a cached singleton via `_pg` global — safe |
| OpenRouter API key not set in CI → warmup logs a warning but doesn't block | Warmup is try/except; CI can set a dummy key to skip the API call |
| `httpx.Timeout` object form is httpx ≥ 0.20 | Pin `httpx>=0.20` in `pyproject.toml` (likely already satisfied) |
| Warmup probe string `"warmup"` generates a real embedding API call | Use an env-guard: `if os.environ.get("GRAPHKNOWS_SKIP_WARMUP"): return` |

---

## CI Checklist (all 3 bugs)

```
[ ] Bug 1: Unit test — assert pg.connect() called in run(), NOT in build_corpus_workflow()
[ ] Bug 1: Integration test — corpus_ingest 2-doc payload → errors == []
[ ] Bug 2: Unit test — Cypher string contains "(e:ENTITY OR e:ENTITY_CONCEPT)"
[ ] Bug 2: Unit test — mock Neo4j returns ENTITY_CONCEPT hit → score > 0
[ ] Bug 2: Integration test — recall_by_topic("memory") returns score > 0
[ ] Bug 3: Unit test — _startup_warmup() calls embed_one and get_pg_store
[ ] Bug 3: Unit test — warmup exception does not bubble up
[ ] Bug 3: Embedder timeout test — httpx mock raises ReadTimeout at 45s boundary
[ ] All: ruff/mypy clean on changed files
[ ] All: no regression in existing test suite
```

---

## Sequencing Recommendation

**All three bugs are fully independent** — no shared files, no shared state.

```
┌────────────────────────┐   ┌────────────────────────┐   ┌────────────────────────┐
│  Bug 1 (corpus_ingest) │   │  Bug 2 (recall_topic)  │   │  Bug 3 (ltm cold-start)│
│  stores.py             │   │  recall.py             │   │  stdio_server.py       │
│  corpus_ingest.py      │   │                        │   │  embedder.py (minor)   │
└────────────────────────┘   └────────────────────────┘   └────────────────────────┘
        PR #1                         PR #2                         PR #3
        (can merge in any order — no conflicts)
```

**If only one developer is available**, recommended order by risk:
1. **Bug 2** first — pure Cypher string change, lowest blast radius, easiest to verify.
2. **Bug 3** next — additive startup code, no existing logic changed.
3. **Bug 1** last — requires careful lifecycle reasoning around asyncpg pool ownership.

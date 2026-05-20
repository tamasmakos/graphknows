---
title: "Technical Debt Register"
created: 2026-05-03
updated: 2026-05-20
type: ops
tags: [debt, architecture, admin, testing]
confidence: high
---

# Technical Debt Register

Re-triaged 2026-05-20. Extraction quality sprint completed 2026-05-20 (6 fixes shipped).
Items referencing the pre-pivot shared Neo4j KG model are marked [PRE-PIVOT] and deprioritised — they are real but only matter when re-activating the corpus pipeline.

---

## HIGH — Blocks correctness or reliability of current system

**1. HybridRetriever uses Neo4j for entity traversal, not Memgraph [HIGH]**
`retrieval/retriever.py` uses `Neo4jStore` for the entity-first graph path. Per-session entities are written to Memgraph, not Neo4j, so the graph traversal step reads from an unrelated corpus (or is empty). The retriever should use the Memgraph session graph for the entity path.
Fix: add `MemgraphStore` to `HybridRetriever`, filter by `session_id`. Track via test.

**2. Session TTL / Memgraph cleanup not wired [HIGH]**
`session/end.py` exists but there is no scheduled or triggered TTL cleanup of Memgraph nodes after session expiry. Long-running deployments will accumulate unbounded session data in Memgraph (in-memory → OOM risk).
Fix: wire `session/end.py` into the session lifecycle on a configurable TTL (default 24h).

**3. pgvector session_id filter not enforced in retriever [HIGH]**
`PgVectorStore.search()` accepts `session_id` but the `HybridRetriever` semantic path may not consistently pass it, risking cross-session context bleed.
Fix: audit all `PgVectorStore.search()` call sites; add integration test asserting session isolation.

---

## MEDIUM — Degrades quality but doesn't break the system

**4. Stub doc fallback fires too broadly [MEDIUM]**
`investigate.py:_make_stub_doc()` fires when the search service is unavailable. Stub docs produce generic entity-rich text that pollutes the session graph with irrelevant named entities. There is no alerting when stubs fire.
Fix: log a WARNING and emit a `search_error` WS event; add a circuit-breaker so > N consecutive stub docs abort the investigation loop with a user-facing error.

**5. Novelty signal uses stub embedding [MEDIUM]**
`dup_ratio` is computed over BAAI/bge-small-en-v1.5 embeddings loaded fresh in each `investigate_node` call. On cold starts this adds 2–4s latency and could OOM on RAM-constrained hosts.
Fix: move embedder to a module-level singleton initialised at service startup.

**6. No rate-limit on WebSocket broadcast [MEDIUM]**
The agent broadcasts every event immediately. High-frequency `thought_token` events (500+ per session) overwhelmed the Cloudflare tunnel (WS frame limit). The `router.py` 350ms batching was an emergency patch.
Fix: formalise the batching as a configurable parameter; add a pressure-relief valve that drops events when the WS queue depth exceeds a threshold.

**7. EPRS system prompt is hard-coded in communicate.py [MEDIUM]**
The 7-section EPRS prompt is a string literal. Difficult to A/B test, version, or swap.
Fix: move to `graph/prompts/eprs.md`, load at startup, expose as a config parameter.

**8. No integration test for end-to-end investigate loop [MEDIUM]**
`test_workflow_wired.py` covers wiring but doesn't run a real search + write + retrieve cycle against live Memgraph + pgvector.
Fix: add a docker-compose-based integration test that spins up infra and runs one iteration.

---

## LOW — Technical improvements, not urgent

**9. `orchestrator.py` is dead code [LOW]**
`services/agent/orchestrator.py` is the pre-LangGraph ReAct orchestrator. It's not in any import path. Leaving it creates confusion for new contributors.
Fix: delete or move to `_archive/`.

**10. Ontology assets duplicated [LOW]**
`services/agent/ontology/assets/` and `services/graphknows/graphknows/ontology/assets/` contain the same files. This happened during the ADR-003 services refactor.
Fix: make `services/agent/ontology/` a symlink or thin wrapper that delegates to `services/graphknows`.

**11. `graphknows_mode=http` path untested [LOW]**
`GRAPHKNOWS_MODE=http` routes extraction through the graphknows HTTP API instead of in-process. The in-process path is the default and is tested; the HTTP path has no dedicated tests.
Fix: add smoke test for the HTTP mode path.

---

## PRE-PIVOT — Only relevant if reactivating corpus pipeline

These items refer to the old shared Neo4j KG pipeline (`services/graphknows/graphknows/graphgen/`). Not blocking for the per-session agent. Deprioritised until corpus pipeline work resumes.

- **Label Collapse** — pipeline falls back to UNKNOWN label instead of inducing compact label set. Fix in `entity_relation/extraction.py`.
- **community_id Propagation** — not propagated through graph; breaks downstream community queries in Neo4j.
- **get_chunks_for_entity not implemented** — retrieval layer missing method; blocks GraphRAG chunk grounding.
- **Entity Resolution Quality** — hub node over/under-merge heuristics weak in `graph_cleaning/resolution.py`.
- **LLM Extraction Quality Metrics** — silhouette score is the only structural metric; no entity-level precision/recall.
- **Patch runner Neo4j scope** — `graphgen patch` targets legacy Neo4j corpus only; per-session Memgraph graphs are built fresh and don't need patching.

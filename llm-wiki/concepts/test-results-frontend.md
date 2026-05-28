---
title: Test Results — Frontend v2 Smoke Test
created: 2026-05-05
updated: 2026-05-05
type: concept
tags: [testing, frontend, smoke-test, qa]
confidence: high
---

# Frontend v2 Smoke Test Results

Tested against: `http://localhost:8080` (cloudflare tunnel: `https://exposure-steel-corpus-parents.trycloudflare.com`)

## Results

| #   | Test                            | Result | Notes                                    |
| --- | ------------------------------- | ------ | ---------------------------------------- |
| 1   | Page loads with session UUID    | ✅ PASS | UUID auto-generated, displayed in header |
| 2   | Ask query → answer appears      | ✅ PASS | After Memgraph fix — answer in ~7s       |
| 3   | Graph nodes appear after query  | ✅ PASS | 26 nodes, 21 edges on Ukraine query      |
| 4   | Share Link → toast              | ✅ PASS | "Link copied!" toast fires               |
| 5   | Export PNG → download           | ✅ PASS | graphknows-graph.png downloaded          |
| 6   | Filter dropdown → PERSON only   | ✅ PASS | Non-person nodes hidden correctly        |
| 7   | Reset → graph clears            | ✅ PASS | Empty state returns                      |
| 8   | ?session= param → graph restore | ✅ PASS | GET /api/session/{id}/graph called       |

## Bugs Found & Fixed

### Bug 1 — 502 on every query
- **Root cause**: Agent defaulted `GRAPHRAG_URL=http://graphrag:8001` (Docker DNS, unresolvable outside compose)
- **Fix**: Changed default to `http://localhost:8001` in `services/agent/main.py`
- **Commit**: `f4741e2`

### Bug 2 — Graph always empty (0 nodes)
- **Root cause**: Entity extractor only looked for `entities_mentioned` key, which graphrag never returns
- **Fix**: Now mines `reasoning_steps[].observation[]` + `[BRACKETED]` names from answer text
- **Commit**: `7b1b68a`

### Bug 3 — 500 Internal Server Error when Memgraph is unreachable
- **Root cause**: Agent tried Memgraph Bolt connection with default 30s timeout, blocking the request thread
- **Fix**: Set `connection_timeout=2` on Memgraph driver; full in-memory fallback when unreachable
- **Commit**: `1a207a7`

## Confirmed Working
- All 3 services running: graphrag (:8001), agent (:8002), frontend (:8080)
- Real answers sourced from Neo4j KG (8 chunks, 129 entities, 6 communities)
- D3.js graph populates with real entities from reasoning trace
- Cloudflare tunnel serving the app publicly

## Related
- [[frontend-v2-design]]
- [[architecture-reference]]
- [[product-roadmap-v2]]

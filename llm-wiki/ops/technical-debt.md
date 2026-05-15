---
title: Technical Debt Register
type: concept
tags: [pipeline, architecture, admin]
confidence: high
---

# Technical Debt Register

Register of known technical debt items with severity (HIGH/MEDIUM/LOW), description, and remediation approach.

## HIGH Severity

1. **Label Collapse [HIGH]**
   Pipeline falls back to UNKNOWN label instead of inducing compact label set from input text.
   Fix: implement label induction step before extraction.

2. **community_id Propagation [HIGH]**
   community_id not properly propagated through graph, breaks downstream retrieval.
   Fix: trace through detection.py and neo4j_adapter.py

3. **pgvector Retrieval [HIGH]**
   Not verified end-to-end. BAAI/bge-small-en-v1.5 embeddings stored but retrieval path unconfirmed.
   Fix: write integration test.

## MEDIUM Severity

4. **get_chunks_for_entity [MEDIUM]**
   Method not implemented in retrieval layer, blocking GraphRAG chunk-level grounding.

5. **RAGAS Evaluation [MEDIUM]**
   No formal evaluation of GraphRAG answer quality. Need faithfulness + answer relevancy + context precision metrics.

6. **Entity Resolution Quality [MEDIUM]**
   Political entity classification and merge eligibility heuristics are weak. Hub nodes may be over-merged or under-merged.

7. **LLM Extraction Quality Metrics [MEDIUM]**
   Silhouette score is the only structural quality metric. Need entity-level precision/recall against known entities.

8. **Frontend Sources Panel Hydration [MEDIUM]**
   The investigative workflow renders final answers and graph updates, but retrieval hits are not yet mapped into the frontend Sources side panel contract. Result: the answer can cite `[Doc-N]` while the side panel still shows `0`.
   Fix: propagate retrieval/source payloads through the websocket and `/chat` answer path, then hydrate the side panel in `apps/web/index.html`.

## LOW Severity

9. **Multi-language Support [LOW]**
   Pipeline only tested on English (EP debates provided pre-translated). German Bundestag debates will require multilingual GLiNER2.

10. **KGE Weighting [LOW - DEPRECATED]**
   Knowledge Graph Embedding weighting was tried and deprecated in favor of Node2Vec. Code may still have dead paths.

11. **Large Chunk Handling [LOW - FIXED]**
    2048/128 chunking caused silent truncation. Now using 512/64. Old configs may still use large chunks.

## Links

[[project-roadmap]] [[graphgen-architecture]] [[administration]]

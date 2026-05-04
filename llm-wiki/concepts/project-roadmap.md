---
title: Project Roadmap
type: concept
tags: [roadmap, planning, architecture]
sources: []
confidence: high
---

# Project Roadmap

## Phase 0 - COMPLETED (Thesis Baseline)

- Full corpus run: 13 EP debates, 311 segments, 2085 chunks
- KG: 6377 nodes, 20661 edges, 3557 entities, 70 topics, 341 subtopics
- Node2Vec modularity improvement: 0.748 -> 0.823
- LLM summarization of all communities
- Neo4j upload pipeline
- LDA baseline comparison

## Phase 1 - IN PROGRESS (GraphKnows/GraphRAG Agent)

Dependencies: T1 (pgvector fix) + T2 (label fix) must complete before T4 (full ingest)

- T1: Verify pgvector retrieval end-to-end (t_66aa8b14) - HIGH priority
- T2: Fix label collapse - use induced label set not UNKNOWN fallback (t_930b0b51) - HIGH
- T3: Fix community_id propagation in graph (t_d5f42600) - MEDIUM
- T4: Full corpus ingest into Neo4j+pgvector (t_b267bf42) - blocked by T1+T2
- T5: Implement get_chunks_for_entity in retrieval layer (t_40558195) - MEDIUM
- T6: RAGAS evaluation of GraphRAG agent (t_639d6b60) - blocked by T4

Effort: 4-6 weeks

## Phase 2 - NEXT (Data Expansion)

- T7: Bundestag scraper (t_e842525d) - scrape + parse German parliament debates
- T8: Multi-language entity extraction (GLiNER2 multilingual model)
- T9: Entity resolution improvements - political entity classification and merge eligibility
- T10: Frontend UI - reasoning transparency interface showing agent steps
- T11: Performance benchmarking vs LDA baseline (formal evaluation)

Effort: 2-3 months

## Phase 3 - FUTURE (Product)

- T12: Multi-country coverage (FR, UK, European Commission)
- T13: SaaS API design and implementation
- T14: Monitoring/alerting for new debate sessions
- T15: Documentation site
- T16: Business development / pilot customers

Effort: 6+ months

## Links

[[graphgen-project]] [[technical-debt]] [[kanban-plan]] [[business-plan]]

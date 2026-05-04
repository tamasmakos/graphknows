---
title: Kanban Task Plan
type: concept
tags: [planning, roadmap, admin]
confidence: high
---

# Kanban Task Plan

## IN PROGRESS

| ID | Title | Priority | Effort | Blocked By |
|---|---|---|---|---|
| t_66aa8b14 | Verify pgvector retrieval end-to-end | HIGH | 2d | - |
| t_930b0b51 | Fix label collapse (UNKNOWN fallback) | HIGH | 3d | - |

## BLOCKED

| ID | Title | Priority | Effort | Blocked By |
|---|---|---|---|---|
| t_b267bf42 | Full corpus ingest (Neo4j + pgvector) | HIGH | 1d | t_66aa8b14 + t_930b0b51 |
| t_639d6b60 | RAGAS re-evaluation | HIGH | 3d | t_b267bf42 |

## IN PROGRESS / MEDIUM

| ID | Title | Priority | Effort | Blocked By |
|---|---|---|---|---|
| t_d5f42600 | Fix community_id propagation | MEDIUM | 2d | - |
| t_40558195 | Implement get_chunks_for_entity | MEDIUM | 2d | - |

## BACKLOG (Phase 2)

| ID | Title | Priority | Effort | Notes |
|---|---|---|---|---|
| t_e842525d | Bundestag scraper | HIGH | 1w | First expansion beyond EP |
| new_01 | Multi-language GLiNER2 | MEDIUM | 2w | Needed for Bundestag |
| new_02 | Entity resolution improvements | MEDIUM | 2w | Political entity classification |
| new_03 | Frontend reasoning transparency UI | MEDIUM | 4w | Show agent steps in real time |
| new_04 | Performance benchmark vs LDA | LOW | 1w | Formal comparison |

## BACKLOG (Phase 3 / Product)

| ID | Title | Priority | Effort | Notes |
|---|---|---|---|---|
| new_05 | SaaS API design | MEDIUM | 3w | REST + GraphQL |
| new_06 | Monitoring/alerting for new debates | MEDIUM | 2w | Scheduled scraping + diff |
| new_07 | Documentation site | LOW | 1w | MkDocs or Docusaurus |
| new_08 | Multi-country coverage (FR, UK) | LOW | 2mo | Depends on scraper pattern |

## DONE

- Thesis full corpus run (6377 nodes, 20661 edges)
- Node2Vec integration (modularity 0.748->0.823)
- Leiden hierarchical community detection
- LLM summarization pipeline
- IterativeOrchestrator refactor (CC 63-><=10)
- LDA baseline comparison
- Neo4j adapter
- pgvector schema setup

## Links

[[project-roadmap]] [[technical-debt]] [[administration]] [[graphgen-project]]

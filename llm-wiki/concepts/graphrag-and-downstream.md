---
title: GraphRAG and Downstream Applications
type: concept
tags: [pipeline, graph, architecture, neo4j, pgvector]
created: 2026-05-03
updated: 2026-05-03
confidence: medium
---

# GraphRAG and Downstream Applications

## GraphKnows Project

GraphKnows is the downstream application layer built on top of the GraphGen knowledge graph. It implements a hybrid GraphRAG (Graph Retrieval-Augmented Generation) agent that combines structured graph traversal with semantic vector search to answer political intelligence queries.

## Dual Retrieval Architecture

Two retrieval backends operate in parallel:

- Neo4j graph database at bolt://172.18.0.1:7687 -- used for structured graph traversal, entity lookup, community queries, and relationship path finding
- pgvector extension on PostgreSQL at postgresql://172.18.0.1:5432/vectordb -- used for semantic similarity search over chunk embeddings

This hybrid approach means queries can be answered by either structured graph paths (good for "how are X and Y connected?") or by semantic similarity (good for "what was said about topic Z?"), with results fused by the orchestrator.

## IterativeOrchestrator

The query orchestration layer was refactored to decompose complex queries. The original implementation had a cyclomatic complexity of CC=63, which was decomposed to components of CC<=10. The pattern is query decomposition: a top-level question is broken into sub-queries, each answered independently, then synthesized.

## Pending Tasks

| Task ID     | Description                                              | Status  |
|-------------|----------------------------------------------------------|---------|
| t_66aa8b14  | Verify pgvector retrieval end-to-end                     | pending |
| t_930b0b51  | Fix label collapse bug in entity normalization           | pending |
| t_d5f42600  | Fix community_id propagation in graph queries            | pending |
| t_b267bf42  | Full corpus ingest into Neo4j + pgvector                 | pending |
| t_639d6b60  | RAGAS re-evaluation after fixes                          | pending |
| t_40558195  | Implement get_chunks_for_entity retrieval function       | pending |

## Corpus Expansion

A Bundestag parliamentary debate scraper is planned (t_e842525d) to extend coverage beyond EU Parliament debates. This is the first step in the Phase 2 national parliament expansion strategy.

## Evaluation: RAGAS

Once the pending fixes are applied and full corpus is ingested, RAGAS evaluation will be run across three metrics:
- Faithfulness: are claims in generated answers supported by retrieved context?
- Answer relevancy: does the answer address the query?
- Context precision: is the retrieved context relevant and not noisy?

---

Links: [[graphgen-project]] [[graphgen-architecture]] [[project-roadmap]]

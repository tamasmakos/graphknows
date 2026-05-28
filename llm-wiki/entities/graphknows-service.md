---
title: "services/graphknows — Consolidated Graph Package"
created: 2026-05-20
updated: 2026-05-20
type: entity
tags: [graphgen, graphrag, retrieval, pipeline, service]
confidence: high
---

# `services/graphknows` — Consolidated Graph Package

## Overview

`services/graphknows` is the consolidated Python package that was formerly split between separate `graphgen` and `graphrag` microservices. It is now the single home for:

- **KG extraction pipeline** — GLiNER2 NER + DSPy relation extraction (`graphgen/`)
- **Retrieval** — HybridRetriever, pgvector store, Memgraph store, Neo4j store (`retrieval/`)
- **Ontology** — domain-agnostic ontology loader, router, and relation extractor (`ontology/`)
- **Models** — shared Pydantic data models (`models/`)
- **MCP server** — tool exposure via MCP protocol (`mcp/`)
- **Ingestion** — document ingestion pipeline glue (`ingestion/`)

## Key Directories

```
services/graphknows/graphknows/
  graphgen/           # NER + relation extraction + community detection + analytics
    pipeline/
      entity_relation/    # GLiNER2 + DSPy extraction
      community/          # Leiden community detection
      lexical_graph_building/
      graph_cleaning/     # canonicalization, entity resolution, pruning
    patch/              # Ontology patch runner (3-phase: structural/legacy/unknown)
    evaluation/         # Entity resolution eval, LDA baseline
    analytics/          # Centrality, coherence, diversity metrics
  retrieval/
    retriever.py        # HybridRetriever: pgvector dense + 2-hop Memgraph + RRF
    stores/
      pgvector.py
      memgraph.py
      neo4j.py
      embedder.py
  ontology/
    loader.py           # OWL/JSONLD loading
    router.py           # embedding-similarity + keyword + LLM classifier
    rel_extractor.py    # relation extraction with ontology schema as context
    assets/             # legal, politics, finance, science ontology files
  models/               # shared Pydantic schemas (Entity, Relation, Session, etc.)
  mcp/                  # MCP server for tool exposure
  ingestion/pipeline.py # document-to-graph pipeline entry point
```

## Role in System

The `services/graphknows` package is called by `services/agent/` as an in-process library via `tools/extraction/in_process.py` — NOT as a separate HTTP microservice. This is a deliberate simplification from the earlier 4-microservice design.

## History

- Originally `graphgen` was a standalone GitHub repo (`tamasmakos/graphgen`) and a separate microservice.
- During the services refactor (ADR-003), `graphgen` and `graphrag` were folded into a single `graphknows` package.
- The old `services/graphgen/` directory no longer exists.

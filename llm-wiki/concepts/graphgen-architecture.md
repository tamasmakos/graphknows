---
title: GraphGen Architecture
created: 2026-05-03
updated: 2026-05-03
type: concept
tags: [pipeline, architecture, graph, extraction]
sources: []
confidence: high
---

# GraphGen Architecture

## Overview

GraphGen is structured as a modular pipeline where each stage transforms data and passes it forward via a shared context object. The design separates application logic (config.yaml) from infrastructure secrets (.env).

## Processing Hierarchy

Documents flow through the following abstraction levels:

  DOC -> SEGMENT -> CHUNK -> ENTITY -> TOPIC

- DOC: raw input document
- SEGMENT: logical section of a document (e.g. paragraph block)
- CHUNK: fixed-size text window (512 chars / 64 overlap) fed to extraction models
- ENTITY: extracted named entity or concept node in the graph
- TOPIC: community of related entities discovered by Leiden clustering

## PipelineContext Bus Object

The central data carrier passed between all pipeline stages. Fields include:

- graph: the live NetworkX DiGraph being constructed
- stats: running counters (nodes, edges, entities, etc.)
- errors: structured error log per stage
- diagnostics: per-stage timing and throughput info

## Config System

- config.yaml: application logic — chunk size, overlap, model names, Leiden resolutions, min community size, etc.
- .env: secrets and infrastructure — DB connection strings, API keys, model endpoints

## Module Structure

```
graphgen/
  entity_relation/
    dspy_module.py       # DSPy signature + predictor for relation extraction
    extraction.py        # orchestrates extraction across chunks
    extractors.py        # GLiNER2 entity detection wrapper
  graph_cleaning/
    resolution.py        # entity resolution logic
    canonicalization.py  # name normalization, deduplication
    pruning.py           # removes low-confidence or isolated nodes
  community/
    detection.py         # Leiden community detection, hierarchical
    subcommunities.py    # SUBCOMMUNITY level processing
  summarization/
    dspy_module.py       # DSPy signature for community summarization
    summarizer.py        # orchestrates summarization
    core.py              # label generation, report writing
  embeddings/
    node2vec_wrapper.py  # Node2Vec training + edge weight update
    rag.py               # pgvector upload + similarity search
  lexical_graph_building/
    chunking.py          # text segmentation logic
    builder.py           # graph construction from extracted data
  visualization/         # graph rendering utilities
```

## Entry Points

- graphgen CLI: main command-line interface
- run_with_config.py: programmatic entry point, loads config.yaml and runs full pipeline

## Core Data Types

| Type                 | Description                                          |
|----------------------|------------------------------------------------------|
| PipelineContext      | shared state object passed through all stages        |
| ChunkExtractionTask  | unit of work: one text chunk + metadata              |
| SegmentData          | structured representation of a document segment      |

## Related Pages

- [[graphgen-project]]
- [[pipeline-stages]]

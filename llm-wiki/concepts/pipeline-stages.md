---
title: Pipeline Stages
created: 2026-05-03
updated: 2026-05-03
type: concept
tags: [pipeline, extraction, embedding, community]
sources: []
confidence: high
---

# Pipeline Stages

## Overview

GraphGen's pipeline is composed of 10 sequential stages. Each stage reads from and writes to the shared PipelineContext object. Stages are designed to be independently testable.

---

## Stage 1: OntologyLabelExtractor

- Parses OWL/RDF ontology files (CDM ontology)
- Extracts class labels and hierarchy from the ontology
- Produces a constrained label set used by downstream extraction stages
- Ensures extracted entities and relations align with the domain schema

---

## Stage 2: Text Chunking

- Splits input documents into overlapping text windows
- Chunk size: 512 characters, overlap: 64 characters
- Rationale: 2048/128 settings caused silent truncation in GLiNER2 (model silently dropped entities near chunk boundaries); 512/64 resolved this
- Implemented in: lexical_graph_building/chunking.py

---

## Stage 3: GLiNER2 Entity Detection

- Model: knowledgator/gliner-multitask-large-v0.5
- Input: text chunk + ontology-derived label set
- Strategy: precision-first (minimize false positives at cost of some recall)
- Output: list of (span, entity_type) pairs per chunk
- Ontology-guided labels constrain detection to domain-relevant entity types

---

## Stage 4: DSPy Relation Extraction

- Model orchestrated via DSPy signatures and predictors
- Input: chunk text + detected entities from Stage 3
- Constrained to ontology classes and detected entity hints
- Drops relation triplets where head/tail entities are not aligned to ontology types
- Output: list of (head_entity, relation_type, tail_entity) triplets

---

## Stage 5: Graph Assembly

- Constructs a NetworkX DiGraph from extracted entities and relations
- Entities -> nodes (with type, source, confidence attributes)
- Relations -> directed edges (with relation type, weight attributes)
- Multiple extractions of the same entity are merged (pre-resolution)

---

## Stage 6: Entity Resolution

- Files: graph_cleaning/canonicalization.py, graph_cleaning/resolution.py
- Canonicalization: normalizes entity names (lowercasing, punctuation, abbreviation expansion)
- Resolution: merges nodes deemed to refer to the same real-world entity
- Political entity classification: tags nodes as PERSON, ORGANIZATION, LOCATION, CONCEPT, EVENT, etc.
- Reduces node count and improves graph density / modularity

---

## Stage 7: Node2Vec Edge Weighting

- Trains Node2Vec embeddings on the current graph
- Parameters: dims=64, walk_length=16, num_walks=20
- Computes pairwise node similarities from embedding space
- Updates edge weights in the graph based on embedding similarity
- Effect: modularity improves from 0.748 to 0.823

---

## Stage 8: Leiden Community Detection

- Algorithm: Leiden (improvement over Louvain)
- Resolution parameters tested: [0.3, 0.5, 0.7, 1.0]
- Hierarchical output: COMMUNITY level (70 topics) + SUBCOMMUNITY level (341 subtopics)
- min_size=3: communities with fewer than 3 nodes are discarded
- Results stored back into PipelineContext and annotated on graph nodes

---

## Stage 9: LLM Summarization

- Orchestrated via DSPy signatures
- Input: per-community subgraph (nodes + edges) for each COMMUNITY and SUBCOMMUNITY
- Output: human-readable label + narrative report per community
- Labels are stored as node attributes; reports are written to output files

---

## Stage 10: Neo4j Upload + pgvector Embeddings

- Uploads final graph to Neo4j at bolt://172.18.0.1:7687
- Generates sentence embeddings for entity descriptions using BAAI/bge-small-en-v1.5
- Stores embeddings in pgvector at postgresql://172.18.0.1:5432/vectordb
- Enables downstream RAG (retrieval-augmented generation) queries over the political knowledge base

---

## Related Pages

- [[graphgen-architecture]]
- [[knowledge-graph-construction]]

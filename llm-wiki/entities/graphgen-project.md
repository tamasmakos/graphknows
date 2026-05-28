---
title: GraphGen Project
created: 2026-05-03
updated: 2026-05-03
type: entity
tags: [project, pipeline, architecture]
sources: []
confidence: high
---

# GraphGen Project

## Overview

GraphGen is a Python package for automated knowledge graph (KG) generation from political text. It implements a full NLP pipeline — from raw document ingestion to structured graph storage — designed to support political intelligence research and analysis.

The project is developed as part of an academic thesis and serves as the primary technical artifact demonstrating graph-based approaches to political document understanding.

Repo: https://github.com/tamasmakos/graphgen

## Current State

Active development. Core pipeline is functional end-to-end. Key pipeline stages (chunking, entity extraction, relation extraction, graph assembly, entity resolution, community detection, summarization, embeddings) are implemented and tested on a real political corpus.

## Tech Stack

| Component         | Technology                                      |
|-------------------|-------------------------------------------------|
| Language          | Python                                          |
| Graph library     | NetworkX (DiGraph)                              |
| Graph DB          | Neo4j                                           |
| LLM orchestration | DSPy                                            |
| Entity extraction | GLiNER2 (knowledgator/gliner-multitask-large-v0.5) |
| Graph embeddings  | Node2Vec                                        |
| Community detect. | Leiden algorithm                                |
| Vector store      | pgvector (PostgreSQL)                           |
| Sentence embeds   | BAAI/bge-small-en-v1.5                          |
| Containerization  | Docker                                          |

## Infrastructure

- Neo4j: bolt://172.18.0.1:7687
- pgvector: postgresql://172.18.0.1:5432/vectordb

## Thesis Context

GraphGen is the technical core of an academic thesis on political intelligence and knowledge graph construction. The pipeline was evaluated on a real political document corpus, yielding the results below.

## Key Results

| Metric              | Value              |
|---------------------|--------------------|
| Nodes               | 6,377              |
| Edges               | 20,661             |
| Distinct entities   | 3,557              |
| Topics (communities)| 70                 |
| Subtopics           | 341                |
| Silhouette score    | 0.160              |
| Modularity (before) | 0.748              |
| Modularity (after)  | 0.823 (post-Node2Vec edge weighting) |

The graph topology is scale-free (power law degree distribution), consistent with real-world political networks exhibiting preferential attachment dynamics.

## Related Pages

- [[graphgen-architecture]]
- [[thesis-findings-and-learnings]]

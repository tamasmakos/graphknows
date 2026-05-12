---
title: Knowledge Graph Construction
created: 2026-05-03
updated: 2026-05-03
type: concept
tags: [graph, extraction, nlp, thesis]
sources: []
confidence: high
---

# Knowledge Graph Construction

## Overview

Knowledge graph (KG) construction in GraphGen is an ontology-guided, multi-stage process that transforms unstructured political text into a structured, queryable graph of entities and their relationships. The methodology prioritizes schema alignment and precision over raw recall.

## Ontology-Guided Schema

The CDM (Common Data Model) ontology defines the permitted entity types and relation types in the graph. This schema constraint:

- Prevents extraction of irrelevant or noisy entities
- Ensures consistent typing across the corpus
- Enables meaningful community detection by keeping the graph semantically coherent
- Allows downstream reasoning to operate within a well-defined domain

Entity types include: PERSON, ORGANIZATION, LOCATION, EVENT, CONCEPT, POLICY, DOCUMENT, and others defined in the CDM ontology.

## Entity Extraction

Entities are extracted using GLiNER2 with ontology-derived label prompts. Each extracted entity is associated with:

- Surface form (as it appears in text)
- Normalized canonical form (post-resolution)
- Ontology type
- Source document and chunk reference
- Confidence score

Entity resolution merges co-referent mentions across chunks and documents, reducing noise and improving graph density.

## Relation Triplets

Relations are extracted by a DSPy-orchestrated LLM in the form:

  (head_entity, relation_type, tail_entity)

Constraints:
- Head and tail must be entities detected by GLiNER2 in the same chunk
- Relation type must be consistent with ontology-defined relation classes
- Triplets failing these constraints are dropped

This results in a directed, typed, multi-relational graph.

## Graph Topology

The resulting graph exhibits scale-free properties:

- Degree distribution follows a power law: P(k) ~ k^(-gamma)
- A small number of highly connected hub nodes dominate (key political figures, major organizations)
- The majority of nodes have low degree (peripheral entities)
- This is consistent with Barabasi-Albert preferential attachment dynamics — new entities in political discourse tend to attach to already well-connected entities

Scale-free topology has implications for community detection: hub nodes act as bridges between communities and can significantly affect modularity scores.

## Entity Resolution Challenges

Key challenges in the political domain:

- Name variation: "PM Johnson", "Boris Johnson", "the Prime Minister" all refer to the same entity
- Acronyms and abbreviations: "EU", "European Union", "the Union"
- Role-based references: "the Chancellor", "the Minister"
- Transliterations and multilingual variants

GraphGen addresses these via canonicalization.py (rule-based normalization) and resolution.py (similarity-based merging with political entity type awareness).

## Canonicalization Strategy

1. Lowercase and strip punctuation
2. Expand known abbreviations (domain-specific dictionary)
3. Compute string similarity between candidate entity pairs (e.g. Jaro-Winkler or embedding cosine)
4. Merge entities above similarity threshold, preserving the most frequent surface form as canonical
5. Re-classify merged nodes with political entity type labels

## Related Pages

- [[pipeline-stages]]
- [[graphgen-architecture]]

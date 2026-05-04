---
source_url: internal/thesis
ingested: 2026-05-03
sha256: 7a6deb69f130a70e947f0ae4f60565dd445b1c0a503a207ecf4aaa8e1d2193bb
---

# GraphGen Thesis Synthesis

## Overview

This document synthesizes the key findings from the GraphGen master thesis, which applies knowledge graph (KG) construction and community detection to the 'This is Europe' European Parliament debate series (2022-2024).

## Corpus

The study used transcripts from the 'This is Europe' EP debate series spanning 2022 to 2024. The corpus covers speeches by 13 EU leaders, segmented into 311 segments and further split into 2085 chunks for processing.

## Pipeline

The processing pipeline proceeds as follows:

1. Ontology-guided entity and relation extraction using GLiNER2 + DSPy
2. Knowledge graph construction in NetworkX
3. Node embedding via Node2Vec
4. Community detection via the Leiden algorithm
5. LLM-based summarization of detected communities (topics and subtopics)

Ontology guidance was critical: providing typed labels to the extraction model reduced entity type drift significantly compared to open extraction. A precision-first policy was applied to triplet dropping -- when uncertain, triplets were dropped rather than kept. Chunking was set at 512 tokens with 64-token overlap.

## Results

The resulting knowledge graph contains:
- 6377 nodes total, of which 3557 are entity nodes
- 20661 edges
- 70 detected topic communities
- 341 subtopic communities

## Evaluation

Community quality was measured using silhouette scores:
- Community-level silhouette: 0.160 (p<0.001) -- weak but statistically significant positive separation
- Subcommunity-level silhouette: -0.220 (p<0.001) -- negative, indicating subcommunity structure is not well-separated in embedding space

Global topic separation was 0.793 vs overlap of 0.207, suggesting the top-level communities are meaningfully distinct.

Node2Vec embeddings improved modularity from 0.748 to 0.823 compared to raw graph structure, demonstrating that embedding-guided community detection is beneficial.

The degree distribution follows a power law P(k) ~ k^-gamma, confirming scale-free topology consistent with the Barabasi-Albert preferential attachment model. This is typical of real-world discourse networks.

## Expert Ground Truth (EPRS)

The European Parliamentary Research Service (EPRS) provides an expert-curated thematic structure of six themes:
1. EU membership value
2. EU values defense
3. Main challenges
4. Delivering for citizens
5. EU integration
6. EU unity

Top recurring topics identified: Ukraine, enlargement, energy. The KG-derived communities showed substantial alignment with these themes.

## Baseline Comparison

The LDA bag-of-words baseline provides important context. Galke and Scherp (2022) warn that graph-based approaches do not always outperform simpler bag-of-words methods, and this finding held here -- the silhouette scores reveal that KG community detection is competitive but not definitively superior on this corpus.

## What Worked

- Ontology-guided labels for GLiNER2: significantly reduced entity type drift
- Precision-first triplet dropping: maintained graph quality over recall
- 512/64 chunking strategy: balanced context and granularity

## What Did Not Work

- Knowledge Graph Embedding (KGE) weighting: deprecated due to poor signal
- Large chunk sizes: caused silent truncation of entities at chunk boundaries
- Subcommunity silhouette: weak negative score suggests the hierarchical decomposition is not well-grounded in embedding geometry

## Conclusion

KG-based community detection is a viable complementary approach to topic modeling, not a replacement for established methods like LDA. It offers a structured, interpretable graph of political discourse but requires careful engineering to extract signal. The approach is particularly valuable for its ability to surface entity relationships and trace how ideas propagate across debates, which bag-of-words models cannot capture.

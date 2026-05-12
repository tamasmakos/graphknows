---
title: Thesis Findings and Learnings
created: 2026-05-03
updated: 2026-05-03
type: concept
tags: [thesis, topic-modeling, nlp, evaluation, corpus]
sources: [raw/papers/graphgen-thesis-synthesis.md]
confidence: high
---

# Thesis Findings and Learnings

## Core Claim

Knowledge graph community detection is a viable complementary approach to topic modeling in political discourse analysis. It is not a replacement for established methods like LDA, but it uniquely surfaces entity relationships and idea propagation across debates -- something bag-of-words models fundamentally cannot do.

## Empirical Results

| Metric                        | Value                        | Notes                                     |
|-------------------------------|------------------------------|-------------------------------------------|
| Corpus                        | 311 segments, 2085 chunks    | 13 EU leaders, 'This is Europe' 2022-2024 |
| Graph nodes                   | 6377 (3557 entity nodes)     |                                           |
| Graph edges                   | 20661                        |                                           |
| Topics (communities)          | 70                           | Top-level Leiden communities              |
| Subtopics                     | 341                          | Hierarchical sub-communities              |
| Silhouette (community)        | 0.160 (p<0.001)              | Weak positive separation                  |
| Silhouette (subcommunity)     | -0.220 (p<0.001)             | Negative -- poor sub-level geometry       |
| Global separation             | 0.793                        | vs overlap 0.207                          |
| Modularity (before Node2Vec)  | 0.748                        |                                           |
| Modularity (after Node2Vec)   | 0.823                        | +0.075 improvement from embeddings        |
| Degree distribution           | Power law P(k) ~ k^-gamma    | Scale-free, validates Barabasi-Albert     |

EPRS expert themes for ground truth: EU membership value, EU values defense, main challenges, delivering for citizens, EU integration, EU unity. Top topics by frequency: Ukraine, enlargement, energy.

## What Worked

- Ontology-guided labels for GLiNER2: constraining entity types with an ontology substantially reduced type drift compared to open extraction
- Precision-first triplet dropping: when extraction confidence was low, dropping the triplet rather than retaining it kept graph quality high
- 512/64 chunking: 512-token chunks with 64-token overlap balanced context window usage and boundary coverage
- Node2Vec embeddings: improved Leiden community modularity from 0.748 to 0.823

## What Did Not Work

- KGE (Knowledge Graph Embedding) weighting: deprecated; the added complexity did not produce meaningful signal
- Large chunk sizes: caused silent entity truncation at boundaries, reducing extraction completeness without obvious error signals
- Subcommunity silhouette score (-0.220): the hierarchical topic decomposition does not align with the Node2Vec embedding geometry, meaning subcommunity boundaries are not semantically crisp

## Open Questions

- Can ontology coverage be extended to non-EU parliamentary corpora without manual rework?
- What is the minimum corpus size for Leiden communities to be stable and meaningful?
- Is the weak silhouette a property of this corpus (short political speeches) or of the approach generally?
- How does KG topic modeling compare to BERTopic or LLM-based clustering at scale?
- Can subcommunity quality be improved with different embedding strategies (e.g., GraphSAGE, spectral)?

## Academic Framing

The work sits at the intersection of information extraction, graph-based topic modeling, and political science NLP. Key references in context:
- Galke and Scherp (2022): caution that graph complexity does not automatically beat bag-of-words; this finding is borne out by the silhouette results
- Barabasi-Albert model: the scale-free topology of the resulting KG confirms the graph exhibits preferential attachment dynamics typical of real discourse networks
- Leiden algorithm: chosen over Louvain for its resolution limit improvements and guaranteed well-connected communities

The EPRS expert ground truth provides a rare external validation anchor for an otherwise unsupervised pipeline.

---

Links: [[graphgen-project]] [[graphgen-architecture]] [[knowledge-graph-construction]]

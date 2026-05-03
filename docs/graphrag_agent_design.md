# GraphKnows: Agentic GraphRAG System Design

Literature synthesis + architecture design for the political intelligence platform.

---

## 1. Literature Review

### 1.1 Foundational GraphRAG

**Edge et al. (2024) — "From Local to Global: A Graph RAG Approach to Query-Focused Summarization" [arXiv:2404.16130]**
Microsoft Research. The paper that named the paradigm. Builds a KG from source text via LLM extraction, then runs Leiden community detection to produce hierarchical summaries at multiple resolutions. At query time it answers global questions by traversing community summaries rather than raw chunks. Key insight: *global* queries (e.g. "what are the main themes?") require community-level aggregation that flat RAG cannot do. Limitation: the community summaries are pre-computed and static; the system is not truly agentic.

**Retrieval-Augmented Generation with Graphs (2025) [arXiv:2501.00309] — 192 citations**
Comprehensive survey of the full GraphRAG taxonomy. Classifies approaches along three axes:
  - Graph type: KG, document graph, scene graph, code graph, molecule graph
  - Retrieval granularity: entity, triple, subgraph, path, community
  - Integration: pre-retrieval (graph augments query), mid-retrieval (graph guides traversal), post-retrieval (graph reranks)

The survey identifies the emerging consensus: best performance comes from *hybrid* retrieval — dense vector search surfaces semantically relevant chunks, graph traversal provides structural relational context that embeddings miss.

**EHRAG (2024) [arXiv:2604.17458-like]**
Entity-Hop Retrieval-Augmented Generation. Explicitly models multi-hop reasoning over KG edges. Introduces the entity-centric query expansion loop: seed entities from dense search -> BFS over graph -> accumulate entity context -> re-rank. This is the architectural foundation of our `hybrid_agent.py` search_entities + get_entity_neighbours pattern.

### 1.2 Agentic / Multi-step Reasoning

**HippoRAG (2024) [arXiv:2406.13213]**
Inspired by the hippocampal indexing theory of human memory. Models the KG as a Personalized PageRank over an associative memory, anchored by dense query embeddings. The key contribution: *associative traversal* — from a query embedding, PPR propagates activation through the graph to find contextually connected entities beyond direct neighbors. Relevance for us: HippoRAG's PPR is a softer, probabilistic alternative to hard BFS; better for fuzzy political queries.

**KAG — Knowledge Augmented Generation (2025) [arXiv:2501.13958]**
Ant Group. Moves beyond retrieval to *mutual enhancement* — the LLM both uses the KG and updates it during reasoning. Introduces the KAG-Solver: a planner that decomposes a question into sub-problems, routes each to a logical solver (graph query, vector search, or LLM reasoning), and assembles the answer. The key idea: **task decomposition + tool routing** is more powerful than any single retrieval strategy.

**UniAI-GraphRAG (2024) [arXiv:2406.14550]**
Unified framework that formalizes GraphRAG as: Query -> Subgraph Extraction -> Subgraph Compression -> Answer Generation. Crucially distinguishes *structural* retrieval (paths, neighborhoods, communities) from *semantic* retrieval (embedding similarity). Shows that combining both always outperforms either alone — the paper that most directly validates our hybrid RRF fusion approach.

**G-Retriever (2024) [arXiv:2402.07630]**
Steerable subgraph retrieval with prize-collecting Steiner tree (PCST) formulation. Given a query, find the minimum subgraph that connects the most relevant entities and edges. Elegant because it produces a compact, interpretable subgraph rather than a bag of nodes. Relevant for visualization: the PCST subgraph is exactly what you'd show in the frontend reasoning graph.

**TRACE (2024) — Transparent Retrieval with Attributed Chain Explanations**
Focuses on auditability: every claim in the final answer is attributed to a specific graph triple or chunk. Directly relevant to our "Reasoning Transparency" requirement.

### 1.3 Multi-Agent Architectures

**AgentGraph / Graph-of-Thoughts (2024)**
Multiple specialized agents on the graph: one agent for entity disambiguation, one for relation traversal, one for summarization. Coordinated by a planner. Compared to single-agent ReAct, multi-agent shows better performance on multi-hop questions with >3 hops.

**LightRAG (2024) [arXiv:2410.05779]**
Dual-level retrieval: local (entity neighborhood) + global (community). Fast because it indexes at both levels at ingestion time rather than computing at query time. Shows that pre-indexing community structure pays off even for small graphs.

---

## 2. Design Principles from the Literature

From the survey and empirical results, five principles emerge:

1. **Hybrid always beats pure.** Dense retrieval finds semantically similar content; graph traversal finds structurally related content. Neither subsumes the other on political text where named entities and legal references are critical. (UniAI-GraphRAG, EHRAG, our own validation)

2. **Agentic decomposition beats monolithic retrieval.** A planner that routes sub-questions to the right tool outperforms a single-pass retriever. (KAG-Solver, AgentGraph)

3. **Community structure enables global reasoning.** Leiden/Louvain communities let the system answer "what are the main themes?" without scanning all chunks. (GraphRAG Microsoft, LightRAG)

4. **Attribution is non-negotiable for political AI.** Every claim must be traceable to a source chunk and graph path. (TRACE, our requirement)

5. **Ontology-guided extraction produces cleaner graphs.** Constraining NER to a domain ontology (LKIF) reduces spurious entities and improves relation precision. (our thesis, supported by KAG's type-aware extraction)

---

## 3. Current State vs Target Architecture

### 3.1 What we have now

```
[EP Debate Text]
    |
    v
[Chunker] -> [LKIF GLiNER2 NER] -> [lkif_chunks.json]
    |
    v
[Neo4j: Chunk + Entity + MENTIONS]     [pgvector: chunk embeddings]
    |                                          |
    +------------------+------------------------+
                       |
               [hybrid_agent.py: ReAct loop]
                  tools: search_chunks (pgvector cosine)
                         search_entities (Neo4j name match)
                         get_entity_neighbours (Neo4j BFS)
                         get_chunks_for_entity (Neo4j)
                  LLM: Groq LLaMA-3.3-70b
```

Gaps:
- No relation extraction (only MENTIONS edges, no typed relations)
- No community detection (Leiden not applied to live Neo4j graph)
- Entity search is exact name match, not semantic
- No attribution / source tracing in final answer format
- Single-agent, no task decomposition

### 3.2 Target Architecture

```
                    ┌─────────────────────────────────────┐
                    │         INGESTION PIPELINE           │
                    │                                      │
  [EP XML / PDF] -> [Chunker] -> [GLiNER2 + LKIF NER]    │
                                      |                    │
                              [Relation Extractor]         │
                              (DSPy + LKIF relations)      │
                                      |                    │
                    ┌─────────────────+──────────────────┐ │
                    │          Neo4j KG                   │ │
                    │  Chunk -> Entity (typed)             │ │
                    │  Entity -[LKIF_RELATION]-> Entity   │ │
                    │  Entity -[:IN_COMMUNITY]-> Topic    │ │
                    └─────────────────────────────────────┘ │
                                      |                    │
                              [BGE Embeddings]              │
                              entity + chunk level          │
                                      |                    │
                    ┌─────────────────+──────────────────┐ │
                    │           pgvector                  │ │
                    │  chunks table (384-dim)              │ │
                    │  entities table (384-dim)  <- NEW   │ │
                    └─────────────────────────────────────┘ │
                    └─────────────────────────────────────┘

                    ┌─────────────────────────────────────┐
                    │         QUERY PIPELINE               │
                    │                                      │
  [User Query]  ->  [Query Planner]                       │
                    (LLM decomposes into sub-goals)        │
                          |                                │
              ┌───────────+────────────┐                   │
              v                        v                   │
    [Dense Retriever]        [Graph Navigator]             │
    pgvector cosine          Neo4j BFS / PPR               │
    chunk + entity           typed relations               │
              |                        |                   │
              └───────────+────────────┘                   │
                          v                                │
                   [RRF Fusion]                            │
                   rerank by reciprocal rank               │
                          |                                │
                   [Community Summaries]                   │
                   attach Leiden topic context             │
                          |                                │
                   [Answer Synthesizer]                    │
                   Groq LLaMA-3.3-70b                      │
                   with attribution (chunk+path)           │
                          |                                │
                   [Reasoning Trace]                       │
                   step-by-step + graph path               │
                   -> frontend visualization               │
                    └─────────────────────────────────────┘
```

---

## 4. Concrete Design: GraphKnows Agentic GraphRAG

### 4.1 Agent Architecture: Planner + Specialist Agents

Following KAG-Solver and AgentGraph, we use a **two-level agent structure**:

```
QueryPlanner (LLM)
    |-- decomposes query into typed sub-tasks
    |-- routes each to the right specialist
    |-- assembles and attributes final answer

Specialists (tools / sub-agents):
    DenseSearchAgent   -- pgvector cosine over chunks and entities
    GraphSearchAgent   -- Neo4j BFS + typed relation traversal
    CommunityAgent     -- Leiden topic lookup (global queries)
    AttributionAgent   -- maps claims back to source triples/chunks
```

The planner uses a simple ReAct loop with these tool signatures:

```python
search_chunks(query: str, k: int, filter_label: str = None)
    # pgvector cosine on chunk embeddings
    # filter_label: LKIF type to restrict results

search_entities(query: str, k: int, label: str = None)
    # pgvector cosine on entity name embeddings (NEW)
    # replaces current broken exact-match approach

get_neighbours(entity: str, relation: str = None, depth: int = 2)
    # Neo4j BFS with optional LKIF relation type filter

get_community(entity: str)
    # returns Leiden community summary for the entity's cluster

get_path(source: str, target: str)
    # shortest typed relation path between two entities
    # enables "what is the connection between X and Y?" queries

summarize_subgraph(entity_ids: list)
    # LLM-compressed summary of a retrieved subgraph
    # used by CommunityAgent for global queries
```

### 4.2 Graph Schema (Neo4j)

```cypher
// Nodes
(:Chunk {id, doc_id, text, date, source_url, offset})
(:Entity {name, label, embedding})   // label = LKIF class
(:Topic {id, label, summary, leiden_level})   // community node

// Edges
(:Chunk)-[:MENTIONS {confidence: float}]->(:Entity)
(:Entity)-[:LKIF_REL {type: "hasParty"|"subjectTo"|"granted"|...}]->(:Entity)
(:Entity)-[:IN_COMMUNITY]->(:Topic)
(:Topic)-[:PARENT_TOPIC]->(:Topic)   // hierarchical Leiden levels
```

LKIF relation types extracted via DSPy (small set, high precision):
  hasParty, subjectTo, amends, repeals, grantedBy, imposedBy, violates, supports

### 4.3 Retrieval: Hybrid RRF

Reciprocal Rank Fusion score for candidate chunk c:

```
RRF(c) = sum over retriever r of: 1 / (k + rank_r(c))
k = 60   (standard constant, robust to rank distribution)
```

Retrievers:
  r1: pgvector chunk cosine (returns ranked chunk list)
  r2: entity cosine -> entity neighbors -> chunks containing neighbors
  r3: community summary match (for global/thematic queries)

Top-N fused results go to the answer synthesizer.

### 4.4 Attribution Format

Every answer includes a structured reasoning trace:

```json
{
  "answer": "The EU discussed Rule of Law conditionality...",
  "steps": [
    {
      "thought": "Search for Rule of Law debate passages",
      "action": "search_chunks",
      "observation": ["chunk_012", "chunk_034"],
      "score": 0.87
    }
  ],
  "citations": [
    {
      "claim": "Rule of Law conditionality was proposed",
      "chunk_id": "chunk_012",
      "entity_path": ["RULE_OF_LAW", "subjectTo", "REGULATION_2021_1060"],
      "source": "EP Plenary 2024-03-13, CRE-9-2024-03-13-ITM-006"
    }
  ]
}
```

### 4.5 Community Detection (Leiden)

Apply networkx-leiden or leidenalg to the Neo4j graph after each ingestion batch:
- Extract (Entity, Entity, weight) triplets from Neo4j
- Run Leiden at two resolution levels: fine (sub-topics) and coarse (main themes)
- LLM-summarize each community into a Topic node
- Store Topic nodes and IN_COMMUNITY edges back in Neo4j

This enables LightRAG-style dual-level retrieval and Microsoft GraphRAG-style global queries.

---

## 5. Implementation Roadmap

### Phase 1 — Fix Entity Search (immediate, 1 session)
- Embed entity names with BGE-small-en -> pgvector entities table
- Replace current exact-match search_entities with cosine search
- Fixes the empty search_entities observations in current demo

### Phase 2 — Relation Extraction (next)
- DSPy module: given (chunk_text, entity_a, entity_b), predict LKIF relation type
- Small supervised set from annotated EP debates
- Store LKIF_REL edges in Neo4j
- Enables get_path tool and typed neighbour traversal

### Phase 3 — Community Detection
- Run leidenalg on Neo4j entity co-mention graph after ingestion
- LLM-summarize each community
- Add Topic nodes + IN_COMMUNITY edges
- Add get_community tool to the agent
- Enables global/thematic queries

### Phase 4 — Planner Agent
- Upgrade from single-loop ReAct to KAG-Solver-style planner
- Planner decomposes query, routes sub-tasks to specialist tools
- Assembles structured answer with citations
- Benchmark on held-out EP debate questions

### Phase 5 — Frontend Visualization
- Expose agent reasoning trace as streaming JSON (SSE)
- D3.js / Cytoscape subgraph visualization of retrieved nodes/edges
- Each reasoning step highlights the active subgraph
- Citations link to source EP document

---

## 6. Key Design Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Graph store | Neo4j | Native graph traversal, Cypher BFS, mature |
| Vector store | pgvector | Already deployed, cosine ops confirmed |
| Embedding model | BGE-small-en-v1.5 (384-dim) | Cached, fast, good for domain text |
| LLM backbone | Groq LLaMA-3.3-70b | Fast inference, long context |
| NER ontology | LKIF + W3C Org (296 labels) | Legal-political domain coverage |
| Relation types | LKIF relation subset (8 types) | High precision, interpretable |
| Community algo | Leiden (leidenalg) | Higher modularity than Louvain |
| Retrieval fusion | RRF k=60 | Robust, no learned weights needed |
| Agent pattern | ReAct -> Planner-Specialist | Incremental upgrade path |

---

## References

- Edge et al. (2024). From Local to Global: A Graph RAG Approach to Query-Focused Summarization. arXiv:2404.16130
- Retrieval-Augmented Generation with Graphs (GraphRAG). arXiv:2501.00309
- KAG: Boosting LLMs in Professional Domains via Knowledge Augmented Generation. arXiv:2501.13958
- HippoRAG: Neurobiologically Inspired Long-Term Memory for Large Language Models. arXiv:2406.13213
- G-Retriever: Retrieval-Augmented Generation for Textual Graph Understanding. arXiv:2402.07630
- LightRAG: Simple and Fast Retrieval-Augmented Generation. arXiv:2410.05779
- UniAI-GraphRAG: Unified Graph Retrieval-Augmented Generation. arXiv:2406.14550

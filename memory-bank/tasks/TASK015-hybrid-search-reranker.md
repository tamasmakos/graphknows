# TASK015 — Hybrid Search + BGE Reranker

**Status:** Pending  
**Added:** 2026-04-21  
**Updated:** 2026-04-21  
**Phase:** C — Agent Intelligence (requires TASK011 complete; TASK014 preferred)

---

## Original Request
The current `search_chunks` tool uses only a vector similarity query (top-k=5). This misses lexically-specific terms, exact names, and numeric codes. Add a fulltext (BM25) path, merge both result sets, and rerank with a cross-encoder before returning to the agent.

---

## Thought Process
"Hybrid search" = vector ANN (semantic recall) + fulltext BM25 (lexical precision) → merge (Reciprocal Rank Fusion) → rerank (cross-encoder). The reranker is the expensive step; it's why we retrieve top-20 from each path but only return top-5 after reranking — the cross-encoder scores the query+passage pairs directly.

**Why RRF instead of score fusion?**
Vector cosine scores and BM25 scores are on different scales and can't be added directly. RRF (`1/(k + rank)`) is rank-based, numerically stable, and well-studied in information retrieval research.

**Reranker model**: `BAAI/bge-reranker-base` (125M params, 512-token context). Small enough to run on CPU. Alternative: `BAAI/bge-reranker-v2-m3` for multilingual. Both load via `sentence-transformers` `CrossEncoder`.

**ENTITY hybrid search**: `search_entities` currently does only vector search. Add fulltext fuzzy match (`CALL db.index.fulltext.queryNodes('entity_fulltext', $query) YIELD node, score`).

---

## Implementation Plan

1. **New `services/graphrag/src/agent/reranker.py`**:
   ```python
   from sentence_transformers import CrossEncoder
   
   class BGEReranker:
       def __init__(self, model_name: str = "BAAI/bge-reranker-base"):
           self._model = CrossEncoder(model_name, max_length=512)
       
       def rerank(self, query: str, passages: list[str], top_k: int = 5) -> list[int]:
           """Returns indices of top_k passages sorted by relevance."""
           pairs = [(query, p) for p in passages]
           scores = self._model.predict(pairs)
           ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
           return ranked[:top_k]
   ```

2. **Update `search_chunks` tool** (`agent/tools.py`):
   - Step 1: Vector query → top-20 (Cypher: `CALL db.index.vector.queryNodes(...)`)
   - Step 2: Fulltext query → top-20 (Cypher: `CALL db.index.fulltext.queryNodes(...)`)
   - Step 3: RRF merge → deduplicate → top-40
   - Step 4: Reranker → top-5
   - Emit `tool_call` + `tool_result` SSE events with duration

3. **Update `search_entities` tool** (`agent/tools.py`):
   - Add fulltext branch: `CALL db.index.fulltext.queryNodes('entity_fulltext', $query) YIELD node, score LIMIT 20`
   - Merge with vector results via RRF
   - No reranker needed for entity names (short strings, lexical match is often sufficient)

4. **RRF helper** (`agent/tools.py` or new `agent/retrieval.py`):
   ```python
   def reciprocal_rank_fusion(result_lists: list[list[str]], k: int = 60) -> list[str]:
       scores: dict[str, float] = {}
       for results in result_lists:
           for rank, doc_id in enumerate(results):
               scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank + 1)
       return sorted(scores, key=scores.get, reverse=True)
   ```

5. **Singleton reranker** — initialize `BGEReranker` once in lifespan (slow first load), reuse per request via `app.state.reranker`.

6. **Langfuse span** on reranker call: `with langfuse.span(name="rerank", input={"query": q, "n_passages": len(passages)}): ...`

7. **Unit tests**:
   - `test_rrf_merges_and_deduplicates()`
   - `test_reranker_returns_top_k()`
   - `test_search_chunks_calls_both_indexes(mock_neo4j)`

---

## Progress Tracking

**Overall Status:** Not Started — 0%

### Subtasks
| ID | Description | Status | Updated | Notes |
|----|-------------|--------|---------|-------|
| 15.1 | Add BAAI/bge-reranker-base to pyproject.toml | Not Started | 2026-04-21 | sentence-transformers dep |
| 15.2 | Create agent/reranker.py (BGEReranker) | Not Started | 2026-04-21 | |
| 15.3 | Initialize reranker singleton in lifespan | Not Started | 2026-04-21 | app.state.reranker |
| 15.4 | Implement RRF helper | Not Started | 2026-04-21 | |
| 15.5 | Update search_chunks: vector+fulltext+RRF+rerank | Not Started | 2026-04-21 | |
| 15.6 | Update search_entities: add fulltext branch | Not Started | 2026-04-21 | |
| 15.7 | Add Langfuse span on reranker | Not Started | 2026-04-21 | |
| 15.8 | Unit tests | Not Started | 2026-04-21 | |

## Progress Log
### 2026-04-21
- Task created. Current state: vector-only search_chunks with top_k=5, no fulltext, no reranker.

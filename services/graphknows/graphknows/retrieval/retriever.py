"""HybridRetriever — entity-first graph retrieval + pgvector semantic search + RRF fusion.

Algorithm
---------
1. Embed the query with BGE-small-en-v1.5 (shared stores/embedder singleton).
2. Entity-first graph path:
   a. ANN search entities table → top k_entity entities by cosine similarity.
   b. For each entity, traverse MENTIONS edges in Neo4j → linked Chunk nodes.
   c. Collect (chunk_id, entity_rank) pairs for RRF scoring.
3. Semantic path:
   a. ANN search chunks table filtered to session_id → top k_semantic chunks.
4. RRF fusion (K=60):
   - Graph path contributes: 1 / (60 + entity_rank + 1) per chunk.
   - Semantic path contributes: 1 / (60 + semantic_rank + 1) per chunk.
   - Sum scores per unique chunk_id, sort descending, take top k_final.
5. Fetch full text + metadata for fused chunk IDs from pgvector.
6. Return a RetrievalContext with all three views: graph_matches, semantic_matches, fused_results.
"""
from __future__ import annotations

import asyncio
import logging
from collections import defaultdict

from graphknows.retrieval.context import ChunkMatch, EntityMatch, RetrievalContext
from graphknows.stores.embedder import embed
from graphknows.stores.neo4j import Neo4jStore
from graphknows.stores.pgvector import PgVectorStore

logger = logging.getLogger(__name__)

RRF_K: int = 60
DEFAULT_K_ENTITY: int = 10
DEFAULT_K_SEMANTIC: int = 20
DEFAULT_K_FINAL: int = 10
MAX_CHUNKS_PER_ENTITY: int = 5


class HybridRetriever:
    """Stateless retriever: entity-graph traversal + dense vector search + RRF fusion.

    Attributes:
        pg: Connected PgVectorStore.
        neo: Connected Neo4jStore.
        embed_model: Name of the sentence-transformer model.
    """

    def __init__(
        self,
        pg: PgVectorStore,
        neo: Neo4jStore,
        embed_model: str = "BAAI/bge-small-en-v1.5",
    ) -> None:
        self.pg = pg
        self.neo = neo
        self.embed_model = embed_model

    async def retrieve(
        self,
        query: str,
        session_id: str,
        k_entity: int = DEFAULT_K_ENTITY,
        k_semantic: int = DEFAULT_K_SEMANTIC,
        k_final: int = DEFAULT_K_FINAL,
    ) -> RetrievalContext:
        """Run the full hybrid retrieval pipeline.

        Args:
            query: Natural-language query string.
            session_id: Session namespace for pgvector chunk search.
            k_entity: Number of top entities retrieved by cosine similarity.
            k_semantic: Number of chunks retrieved by direct cosine similarity.
            k_final: Maximum fused results to return.

        Returns:
            RetrievalContext with all retrieval views serialisable to XML.
        """
        # ── Step 1: embed query ────────────────────────────────────────────
        emb_array = await asyncio.to_thread(embed, [query], self.embed_model)
        emb = emb_array[0].tolist()

        # ── Step 2 & 3: run entity + semantic search concurrently ──────────
        entity_rows_task = asyncio.create_task(
            self.pg.search_entities(emb, k=k_entity)
        )
        sem_rows_task = asyncio.create_task(
            self.pg.search_chunks(emb, session_id=session_id, k=k_semantic)
        )
        entity_rows, sem_rows = await asyncio.gather(entity_rows_task, sem_rows_task)

        # ── Step 4: entity → chunks via Neo4j MENTIONS traversal ──────────
        graph_chunk_info: dict[str, dict] = {}
        entity_matches: list[EntityMatch] = []

        neo_tasks = [
            self.neo.get_chunks_for_entity(ent["name"], top_k=MAX_CHUNKS_PER_ENTITY)
            for ent in entity_rows
        ]
        neo_results = await asyncio.gather(*neo_tasks, return_exceptions=True)

        for i, (ent, neo_chunks) in enumerate(zip(entity_rows, neo_results)):
            if isinstance(neo_chunks, Exception):
                logger.warning("Neo4j lookup failed for %r: %s", ent["name"], neo_chunks)
                neo_chunks = []

            entity_similarity = ent["score"]
            chunk_matches: list[ChunkMatch] = []

            for j, nc in enumerate(neo_chunks):
                mention_weight = entity_similarity * (1.0 / (j + 1))
                cid = nc["chunk_id"]

                if cid not in graph_chunk_info or mention_weight > graph_chunk_info[cid]["mention_weight"]:
                    graph_chunk_info[cid] = {
                        "entity_rank": i,
                        "mention_weight": mention_weight,
                        "doc_id": nc.get("doc_id", ""),
                        "text": nc.get("text", ""),
                    }

                chunk_matches.append(
                    ChunkMatch(
                        chunk_id=cid,
                        doc_id=nc.get("doc_id", ""),
                        text=nc.get("text", ""),
                        metadata={"mention_weight": mention_weight},
                        entities=[{"name": ent["name"], "label": ent["label"]}],
                    )
                )

            entity_matches.append(
                EntityMatch(
                    name=ent["name"],
                    label=ent["label"],
                    similarity=entity_similarity,
                    chunks=chunk_matches,
                )
            )

        # ── Step 5: build semantic_matches list ────────────────────────────
        semantic_matches: list[ChunkMatch] = [
            ChunkMatch(
                chunk_id=r["chunk_id"],
                doc_id=r.get("doc_id", ""),
                text=r.get("text", ""),
                metadata={"cosine_score": r.get("cosine_score", 0.0), **r.get("metadata", {})},
                entities=[],
            )
            for r in sem_rows
        ]

        # ── Step 6: RRF fusion ─────────────────────────────────────────────
        rrf_scores: dict[str, float] = defaultdict(float)
        sem_ids: set[str] = set()

        for cid, info in graph_chunk_info.items():
            rrf_scores[cid] += 1.0 / (RRF_K + info["entity_rank"] + 1)

        for rank, r in enumerate(sem_rows):
            cid = r["chunk_id"]
            rrf_scores[cid] += 1.0 / (RRF_K + rank + 1)
            sem_ids.add(cid)

        ranked = sorted(rrf_scores.items(), key=lambda x: -x[1])[:k_final]

        # ── Step 7: fetch full text + metadata for fused chunks ────────────
        all_ids = [cid for cid, _ in ranked]
        full_text_map = await self.pg.get_chunks_by_ids(all_ids)

        chunk_entities: dict[str, list[dict]] = defaultdict(list)
        for em in entity_matches:
            for cm in em.chunks:
                chunk_entities[cm.chunk_id].append({"name": em.name, "label": em.label})

        fused: list[dict] = []
        for rank, (cid, rrf_score) in enumerate(ranked):
            fc = full_text_map.get(cid, {})
            sources = []
            if cid in graph_chunk_info:
                sources.append("graph")
            if cid in sem_ids:
                sources.append("semantic")

            fused.append(
                {
                    "chunk_id": cid,
                    "doc_id": fc.get("doc_id", graph_chunk_info.get(cid, {}).get("doc_id", "")),
                    "text": fc.get("text", graph_chunk_info.get(cid, {}).get("text", "")),
                    "metadata": fc.get("metadata", {}),
                    "rrf_score": round(rrf_score, 6),
                    "sources": ",".join(sources),
                    "rank": rank + 1,
                    "entities": chunk_entities.get(cid, []),
                }
            )

        return RetrievalContext(
            query=query,
            session_id=session_id,
            graph_matches=entity_matches,
            semantic_matches=semantic_matches,
            fused_results=fused,
        )

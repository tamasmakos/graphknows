"""HybridRetriever: pgvector dense + Memgraph 2-hop walk + RRF fusion."""
from __future__ import annotations

import asyncio
import re
from collections import defaultdict
from typing import Any

from services.agent.graph.state import Chunk
from services.agent.infra.pgvector_client import PgVectorClient


# Regex for simple entity extraction: capitalised phrases (2+ words or single
# capitalised word preceded by stop).  Same pattern used in keyword graph fallback.
_ENTITY_RE = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b")


def _extract_entity_ids(texts: list[str]) -> list[str]:
    """Extract candidate entity IDs from chunk texts via simple regex.

    Args:
        texts: List of chunk text strings.

    Returns:
        Deduplicated list of entity name strings.
    """
    entities: set[str] = set()
    for t in texts:
        for match in _ENTITY_RE.finditer(t):
            entities.add(match.group(1))
    return list(entities)


class HybridRetriever:
    """Hybrid retriever combining dense pgvector search with graph 2-hop walk.

    Attributes:
        session_id: Active session identifier for namespacing.
        pg: PgVectorClient instance for dense retrieval.
        mg: Memgraph (Neo4j) driver instance for graph traversal.
    """

    RRF_K: int = 60

    def __init__(
        self, session_id: str, pg_client: PgVectorClient, mg_driver: Any
    ) -> None:
        """Initialise HybridRetriever.

        Args:
            session_id: Active session identifier.
            pg_client: PgVectorClient for dense vector search.
            mg_driver: Memgraph/Neo4j driver for graph traversal.
        """
        self.session_id = session_id
        self.pg = pg_client
        self.mg = mg_driver

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def retrieve(
        self, query: str, k_dense: int = 20, k_final: int = 8
    ) -> list[Chunk]:
        """Run hybrid retrieval: dense search, entity extraction, 2-hop walk,
        RRF fusion.

        Args:
            query: Natural-language query string.
            k_dense: Number of top dense hits to retrieve (default 20).
            k_final: Number of final fused results to return (default 8).

        Returns:
            Top-k_final Chunk instances after RRF fusion.
        """
        # 1. Dense search
        dense_hits = await self.pg.search_chunks(
            self.session_id, query, k=k_dense
        )

        # 2. Entity extraction from dense hit texts
        dense_texts = [h["text"] for h in dense_hits]
        entity_ids = _extract_entity_ids(dense_texts)

        # 3. 2-hop Memgraph walk
        graph_hits: list[dict[str, Any]] = []
        if entity_ids:
            graph_hits = await self._two_hop_walk(entity_ids)

        # 4. Map doc_ids back to chunks
        graph_doc_ids = list({h["doc_id"] for h in graph_hits if h.get("doc_id")})
        graph_chunks: list[dict[str, Any]] = []
        if graph_doc_ids:
            graph_chunks = await self.pg.get_chunks_by_doc_ids(
                self.session_id, graph_doc_ids
            )

        # 5. Convert to Chunk dataclasses
        dense_chunks = [
            Chunk(id=h["id"], doc_id=h["doc_id"], text=h["text"])
            for h in dense_hits
        ]
        graph_chunk_objs = [
            Chunk(id=c["id"], doc_id=c["doc_id"], text=c["text"])
            for c in graph_chunks
        ]

        # 6. RRF merge
        return self._rrf(dense_chunks, graph_chunk_objs, k_final=k_final)

    # ------------------------------------------------------------------
    # Graph walk
    # ------------------------------------------------------------------

    async def _two_hop_walk(
        self, entity_ids: list[str]
    ) -> list[dict[str, Any]]:
        """Run 2-hop Cypher walk over Memgraph entities.

        Args:
            entity_ids: List of entity name strings to start from.

        Returns:
            List of dicts with keys ``id`` and ``doc_id``.
        """
        if not entity_ids:
            return []

        def _sync_walk() -> list[dict[str, Any]]:
            try:
                with self.mg.session() as session:
                    result = session.run(
                        """
                        MATCH (e:Entity {session_id: $sid})-[*1..2]-(n:Entity {session_id: $sid})
                        WHERE e.id IN $entity_ids
                        RETURN DISTINCT n.id AS id, n.doc_ids AS doc_ids
                        """,
                        sid=self.session_id,
                        entity_ids=entity_ids,
                    )
                    rows = result.data()
            except Exception:
                return []
            # doc_ids may be a list or a single string
            out: list[dict[str, Any]] = []
            for r in rows:
                nid = r.get("id")
                raw = r.get("doc_ids")
                if not nid:
                    continue
                if isinstance(raw, list):
                    for did in raw:
                        out.append({"id": nid, "doc_id": did})
                elif isinstance(raw, str):
                    out.append({"id": nid, "doc_id": raw})
            return out

        return await asyncio.to_thread(_sync_walk)

    # ------------------------------------------------------------------
    # RRF fusion
    # ------------------------------------------------------------------

    @staticmethod
    def _rrf(
        dense: list[Chunk],
        graph: list[Chunk],
        k_final: int = 8,
        k: int = 60,
    ) -> list[Chunk]:
        """Reciprocal Rank Fusion of dense and graph result lists.

        Args:
            dense: Ranked chunks from dense vector search.
            graph: Ranked chunks from graph 2-hop walk.
            k_final: Number of top results to return (default 8).
            k: RRF constant (default 60).

        Returns:
            Top-k_final Chunk instances fused by reciprocal rank.
        """
        scores: dict[str, float] = defaultdict(float)
        for rank, chunk in enumerate(dense, 1):
            scores[chunk.id] += 1.0 / (k + rank)
        for rank, chunk in enumerate(graph, 1):
            scores[chunk.id] += 1.0 / (k + rank)
        all_chunks = {c.id: c for c in dense + graph}
        sorted_ids = sorted(scores, key=scores.__getitem__, reverse=True)
        return [all_chunks[cid] for cid in sorted_ids[:k_final]]

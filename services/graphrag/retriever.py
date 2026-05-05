"""Pure retrieval layer for the GraphRAG service.

No LLM synthesis here — this is a dumb retrieval primitive called by the agent service.
All retrieval methods that were embedded in agent.py tool implementations live here.
"""
from __future__ import annotations

import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

# Ensure the graphrag package directory is on sys.path so relative imports work
_SVC_DIR = Path(__file__).parent
if str(_SVC_DIR) not in sys.path:
    sys.path.insert(0, str(_SVC_DIR))

from graph_store import NeoStore, PgStore, embed

log = logging.getLogger("graphrag.retriever")


class HybridRetriever:
    """Pure retrieval primitive that combines pgvector and Neo4j for hybrid search.

    Attributes:
        cfg: Configuration dictionary with Neo4j and pgvector connection params.
        _pg: Lazy-loaded PgStore instance.
        _neo: Lazy-loaded NeoStore instance.
    """

    def __init__(self, cfg: dict) -> None:
        """Initialize the retriever with configuration.

        Args:
            cfg: Dictionary containing neo4j_uri, neo4j_user, neo4j_password,
                 neo4j_database, pgvector_dsn, and embed_model.
        """
        self.cfg = cfg
        self._pg: Optional[PgStore] = None
        self._neo: Optional[NeoStore] = None

    def _pg_store(self) -> PgStore:
        """Lazy-load the pgvector store.

        Returns:
            PgStore instance connected to the configured DSN.
        """
        if self._pg is None:
            self._pg = PgStore(self.cfg["pgvector_dsn"])
        return self._pg

    def _neo_store(self) -> NeoStore:
        """Lazy-load the Neo4j store.

        Returns:
            NeoStore instance connected to the configured Neo4j instance.
        """
        if self._neo is None:
            self._neo = NeoStore(
                self.cfg["neo4j_uri"],
                self.cfg["neo4j_user"],
                self.cfg["neo4j_password"],
                self.cfg["neo4j_database"],
            )
        return self._neo

    def _embed(self, text: str) -> list[float]:
        """Generate embedding for the given text.

        Args:
            text: The text to embed.

        Returns:
            List of floats representing the normalized embedding.
        """
        return embed([text], self.cfg["embed_model"])[0].tolist()

    def retrieve(self, query: str, k: int = 10) -> dict:
        """RRF fusion of pgvector chunk search + entity search.

        Args:
            query: The search query string.
            k: Number of top results to return after RRF fusion.

        Returns:
            Dictionary with keys chunks, entities, and scores.
        """
        query_emb = self._embed(query)

        # Dense retrieval: pgvector chunk search
        chunks = self._pg_store().search_chunks(query_emb, k=k)

        # Entity retrieval: pgvector entity search
        entities = self._pg_store().search_entities(query_emb, k=k)

        # RRF fusion
        K = 60
        scores: dict[str, float] = defaultdict(float)
        chunk_texts: dict[str, str] = {}

        for rank, c in enumerate(chunks):
            cid = c["chunk_id"]
            scores[cid] += 1.0 / (K + rank + 1)
            chunk_texts[cid] = c.get("text", "")

        # Also add entity-linked chunks via graph traversal
        for ent in entities[:5]:
            ent_name = ent["name"]
            try:
                nbr_chunks = self._neo_store().get_chunks_for_entity(ent_name)
                for rank, nc in enumerate(nbr_chunks):
                    cid = nc["chunk_id"]
                    scores[cid] += 1.0 / (K + rank + 1)
                    chunk_texts.setdefault(cid, nc.get("text", ""))
            except Exception as exc:
                log.warning("Failed to get chunks for entity %r: %s", ent_name, exc)

        # Sort by RRF score
        ranked = sorted(scores.items(), key=lambda x: -x[1])[:k]
        fused_chunks = [
            {"chunk_id": cid, "score": round(score, 4), "text": chunk_texts.get(cid, "")[:400]}
            for cid, score in ranked
        ]

        return {
            "chunks": fused_chunks,
            "entities": entities,
            "scores": [{"chunk_id": cid, "rrf_score": round(score, 4)} for cid, score in ranked],
        }

    def get_path(self, entity_a: str, entity_b: str, max_hops: int = 3) -> dict:
        """BFS over LKIF_REL in Neo4j between two entities.

        Args:
            entity_a: Name of the source entity.
            entity_b: Name of the target entity.
            max_hops: Maximum BFS depth.

        Returns:
            Dictionary with keys path (list of node dicts) and found (bool).
        """
        neo = self._neo_store()
        # Neo4j shortestPath does not accept parameters for hop depth — use fixed max.
        rows = neo._run(
            """
            MATCH (a:Entity {name: $name_a}), (b:Entity {name: $name_b}),
                  p = shortestPath((a)-[:LKIF_REL*1..6]-(b))
            RETURN [n IN nodes(p) | {name: n.name, label: labels(n)[0]}] AS path_nodes
            LIMIT 1
            """,
            name_a=entity_a.upper(),
            name_b=entity_b.upper(),
        )
        if rows:
            return {"path": rows[0]["path_nodes"], "found": True}
        return {"path": [], "found": False}

    def get_chunks(self, entity_id: str, top_k: int = 5) -> list[dict]:
        """Entity → source chunks with full text from pgvector.

        This fixes the incomplete get_chunks_for_entity in graph_store.py by
        joining Neo4j chunk IDs with pgvector for full text.

        Args:
            entity_id: The entity name to look up chunks for.
            top_k: Maximum number of chunks to return.

        Returns:
            List of chunk dicts with chunk_id, doc_id, text, and score.
        """
        neo = self._neo_store()
        pg = self._pg_store()

        # Get chunk IDs from Neo4j
        rows = neo._run(
            """
            MATCH (e:Entity {name: $name})<-[:MENTIONS]-(c:Chunk)
            RETURN c.id AS chunk_id
            ORDER BY c.id
            LIMIT $top_k
            """,
            name=entity_id.upper(),
            top_k=top_k,
        )
        chunk_ids = [r["chunk_id"] for r in rows]
        if not chunk_ids:
            return []

        # Fetch full text from pgvector
        cur = pg.conn.cursor()
        cur.execute(
            """SELECT id, doc_id, text FROM chunks WHERE id = ANY(%s)""",
            (chunk_ids,),
        )
        rows = cur.fetchall()
        cur.close()

        return [
            {"chunk_id": r[0], "doc_id": r[1], "text": r[2][:400] if r[2] else ""}
            for r in rows
        ]

    def get_community(self, entity_id: str) -> dict:
        """Leiden community lookup for entity.

        Args:
            entity_id: The entity name to look up.

        Returns:
            Dictionary with community_id, label, members, or community_id None if not found.
        """
        neo = self._neo_store()
        result = neo.get_entity_community(entity_id)
        if result is None:
            return {"community_id": None}

        # Fetch community members
        rows = neo._run(
            """
            MATCH (t:Topic {id: $tid})<-[:IN_COMMUNITY]-(e:Entity)
            RETURN e.name AS name, e.label AS label
            ORDER BY e.name
            """,
            tid=result["id"],
        )
        members = [{"name": r["name"], "label": r["label"]} for r in rows]

        return {
            "community_id": result["id"],
            "label": result["label"],
            "members": members,
        }

    def close(self) -> None:
        """Close all open store connections."""
        if self._pg:
            self._pg.close()
        if self._neo:
            self._neo.close()

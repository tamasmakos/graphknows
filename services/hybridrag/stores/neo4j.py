"""Unified Neo4j store for HybridRAG.

Merges:
  - NeoStore (corpus Chunk/Entity/Topic operations)
  - Neo4jGDSBackend (session-scoped SessionEntity writes)

All sync Neo4j calls are wrapped in asyncio.to_thread for non-blocking operation.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any

logger = logging.getLogger(__name__)


class Neo4jStore:
    """Async-friendly Neo4j store for corpus + session entity graph.

    Attributes:
        uri: Bolt URI for Neo4j, e.g. bolt://neo4j:7687.
        user: Neo4j username.
        password: Neo4j password.
        database: Target database (default: neo4j).
    """

    def __init__(
        self,
        uri: str,
        user: str,
        password: str,
        database: str = "neo4j",
    ) -> None:
        from neo4j import GraphDatabase

        self._driver = GraphDatabase.driver(uri, auth=(user, password))
        self._db = database

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run(self, cypher: str, **params: Any) -> list[Any]:
        with self._driver.session(database=self._db) as s:
            return list(s.run(cypher, **params))

    async def _arun(self, cypher: str, **params: Any) -> list[Any]:
        return await asyncio.to_thread(self._run, cypher, **params)

    def close(self) -> None:
        self._driver.close()

    # ------------------------------------------------------------------
    # Indexes
    # ------------------------------------------------------------------

    async def ensure_indexes(self) -> None:
        """Create corpus + session indexes idempotently."""

        def _sync() -> None:
            for stmt in [
                "CREATE INDEX chunk_id IF NOT EXISTS FOR (c:Chunk) ON (c.id)",
                "CREATE INDEX entity_name IF NOT EXISTS FOR (e:Entity) ON (e.name)",
                "CREATE INDEX topic_id IF NOT EXISTS FOR (t:Topic) ON (t.id)",
                "CREATE INDEX sess_entity_session IF NOT EXISTS FOR (n:SessionEntity) ON (n.session_id)",
                "CREATE INDEX sess_entity_id IF NOT EXISTS FOR (n:SessionEntity) ON (n.entity_id)",
            ]:
                try:
                    with self._driver.session(database=self._db) as s:
                        s.run(stmt).consume()
                except Exception as exc:  # index already exists → safe to ignore
                    logger.debug("Index creation note: %s", exc)

        await asyncio.to_thread(_sync)
        logger.info("Neo4j indexes ensured")

    # ------------------------------------------------------------------
    # Corpus chunk writes
    # ------------------------------------------------------------------

    async def upsert_chunks(self, chunks: list[dict[str, Any]]) -> None:
        """Write Chunk nodes to Neo4j (text stored for quick lookup)."""
        if not chunks:
            return

        def _sync() -> None:
            for c in chunks:
                self._run(
                    "MERGE (ch:Chunk {id:$id}) SET ch.doc_id=$doc_id, ch.text=$text",
                    id=c["id"],
                    doc_id=c.get("doc_id", ""),
                    text=(c.get("text") or "")[:3000],
                )

        await asyncio.to_thread(_sync)
        logger.debug("Upserted %d Chunk nodes", len(chunks))

    # ------------------------------------------------------------------
    # Corpus entity + MENTIONS writes
    # ------------------------------------------------------------------

    async def upsert_entities_and_mentions(self, chunks: list[dict[str, Any]]) -> None:
        """Write Entity nodes and MENTIONS edges derived from chunk entity lists."""
        if not chunks:
            return

        def _sync() -> None:
            for c in chunks:
                for ent in c.get("entities", []):
                    name = ent.get("text", "").strip().upper()
                    if not name or len(name) < 2:
                        continue
                    label = (
                        ent.get("label")
                        or ent.get("ontology_label")
                        or ent.get("coarse_bucket")
                        or "Entity"
                    )
                    self._run(
                        "MERGE (e:Entity {name:$name}) SET e.label=$label",
                        name=name,
                        label=label,
                    )
                    self._run(
                        """MATCH (ch:Chunk {id:$cid})
                           MATCH (e:Entity {name:$name})
                           MERGE (ch)-[:MENTIONS]->(e)""",
                        cid=c["id"],
                        name=name,
                    )

        await asyncio.to_thread(_sync)

    # ------------------------------------------------------------------
    # Topic / community writes
    # ------------------------------------------------------------------

    async def upsert_topics(self, topics: list[dict[str, Any]]) -> None:
        """Write Topic nodes and IN_COMMUNITY edges.

        Each topic dict: {id, community_id, label, summary, members: [entity_names]}.
        """
        if not topics:
            return

        def _sync() -> None:
            for t in topics:
                self._run(
                    """MERGE (tp:Topic {id:$id})
                       SET tp.label=$label, tp.summary=$summary,
                           tp.community_id=$community_id""",
                    id=t["id"],
                    label=t.get("label", ""),
                    summary=t.get("summary", ""),
                    community_id=t.get("community_id"),
                )
                for name in t.get("members", []):
                    self._run(
                        """MATCH (e:Entity {name:$name})
                           MATCH (tp:Topic {id:$tid})
                           MERGE (e)-[:IN_COMMUNITY]->(tp)""",
                        name=name,
                        tid=t["id"],
                    )

        await asyncio.to_thread(_sync)

    async def upsert_co_occurs(self, cooccurrences: list[tuple]) -> None:
        """Write (entity_a, entity_b, weight) CO_OCCURS edges."""
        if not cooccurrences:
            return

        def _sync() -> None:
            for a, b, w in cooccurrences:
                self._run(
                    """MATCH (ea:Entity {name:$a})
                       MATCH (eb:Entity {name:$b})
                       MERGE (ea)-[r:CO_OCCURS]->(eb)
                       SET r.weight=$w""",
                    a=a,
                    b=b,
                    w=int(w),
                )

        await asyncio.to_thread(_sync)

    # ------------------------------------------------------------------
    # Session entity writes (agent investigative loop)
    # ------------------------------------------------------------------

    async def upsert_session_entities(
        self,
        session_id: str,
        entities: list[Any],
        relations: list[Any],
    ) -> None:
        """Write SessionEntity nodes and RELATES edges for a live agent session.

        Args:
            session_id: Active session namespace.
            entities: Iterable of objects with .id, .name, .label, .confidence, .doc_ids.
            relations: Iterable of objects with .source, .target, .type.
        """

        def _sync() -> None:
            with self._driver.session(database=self._db) as s:
                for e in entities:
                    s.run(
                        "MERGE (n:SessionEntity {entity_id:$eid, session_id:$sid}) SET n += $props",
                        eid=e.id,
                        sid=session_id,
                        props={
                            "name": e.name,
                            "label": e.label,
                            "confidence": e.confidence,
                            "doc_ids": e.doc_ids,
                        },
                    ).consume()
                for r in relations:
                    s.run(
                        """MATCH (a:SessionEntity {entity_id:$src, session_id:$sid})
                           MATCH (b:SessionEntity {entity_id:$tgt, session_id:$sid})
                           MERGE (a)-[rel:RELATES {type:$rtype, session_id:$sid}]->(b)""",
                        src=r.source,
                        tgt=r.target,
                        rtype=r.type,
                        sid=session_id,
                    ).consume()

        await asyncio.to_thread(_sync)

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    async def get_chunks_for_entity(self, entity_name: str, top_k: int = 5) -> list[dict]:
        """Return chunk records linked to *entity_name* via MENTIONS edges.

        Returns list of dicts: {chunk_id, doc_id, text}.
        """
        rows = await self._arun(
            """MATCH (e:Entity {name:$name})<-[:MENTIONS]-(c:Chunk)
               RETURN c.id AS id, c.doc_id AS doc_id, c.text AS text
               ORDER BY c.id LIMIT $top_k""",
            name=entity_name.upper(),
            top_k=top_k,
        )
        return [
            {
                "chunk_id": r["id"],
                "doc_id": r["doc_id"] or "",
                "text": (r["text"] or "")[:3000],
            }
            for r in rows
        ]

    async def get_entity_community(self, entity_name: str) -> dict | None:
        """Return the Topic this entity belongs to, or None."""
        rows = await self._arun(
            """MATCH (e:Entity {name:$name})-[:IN_COMMUNITY]->(t:Topic)
               RETURN t.id AS id, t.label AS label, t.summary AS summary LIMIT 1""",
            name=entity_name.upper(),
        )
        if rows:
            r = rows[0]
            return {"id": r["id"], "label": r["label"], "summary": r["summary"]}
        return None

    async def get_path(
        self,
        entity_a: str,
        entity_b: str,
    ) -> dict:
        """Return shortest path between two Entity nodes via LKIF_REL edges.

        Returns dict with keys: path (list of {name, label} dicts), found (bool).
        """
        rows = await self._arun(
            """
            MATCH (a:Entity {name:$na}), (b:Entity {name:$nb}),
                  p = shortestPath((a)-[:LKIF_REL*1..6]-(b))
            RETURN [n IN nodes(p) | {name: n.name, label: labels(n)[0]}] AS path_nodes
            LIMIT 1
            """,
            na=entity_a.upper(),
            nb=entity_b.upper(),
        )
        if rows:
            return {"path": rows[0]["path_nodes"], "found": True}
        return {"path": [], "found": False}

    async def get_all_entity_cooccurrences(self) -> list[tuple]:
        """Return (entity_a, entity_b, weight) pairs for Leiden community detection."""
        rows = await self._arun(
            """MATCH (a:Entity)<-[:MENTIONS]-(c:Chunk)-[:MENTIONS]->(b:Entity)
               WHERE a.name < b.name
               RETURN a.name AS a, b.name AS b, count(c) AS w"""
        )
        return [(r["a"], r["b"], r["w"]) for r in rows]

    async def stats(self) -> dict:
        """Return basic node/edge counts."""

        def _count(label: str) -> int:
            rows = self._run(f"MATCH (n:{label}) RETURN count(n) AS c")
            return rows[0]["c"] if rows else 0

        def _sync() -> dict:
            mentions = self._run("MATCH ()-[r:MENTIONS]->() RETURN count(r) AS c")
            return {
                "chunks": _count("Chunk"),
                "entities": _count("Entity"),
                "topics": _count("Topic"),
                "session_entities": _count("SessionEntity"),
                "mentions": mentions[0]["c"] if mentions else 0,
            }

        return await asyncio.to_thread(_sync)

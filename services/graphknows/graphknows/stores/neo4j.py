"""Neo4j store for graphknows — corpus reads + session writes + idempotent ingestion.

All sync Neo4j calls are wrapped in asyncio.to_thread for non-blocking operation.
Idempotent upserts use UUID5 deterministic IDs keyed on (doc_id, chunk_hash).
"""
from __future__ import annotations

import asyncio
import logging
import uuid
from typing import Any

logger = logging.getLogger(__name__)

# UUID5 namespace for graphknows nodes (fixed sentinel)
_GK_NS = uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")  # NAMESPACE_URL


def chunk_node_id(doc_id: str, chunk_hash: str) -> str:
    """Deterministic UUID5 for a Chunk node — idempotent across pipeline runs."""
    return str(uuid.uuid5(_GK_NS, f"chunk:{doc_id}:{chunk_hash}"))


def entity_node_id(name: str) -> str:
    """Deterministic UUID5 for an Entity node keyed on normalised name."""
    return str(uuid.uuid5(_GK_NS, f"entity:{name.strip().upper()}"))


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
                except Exception as exc:
                    logger.debug("Index creation note: %s", exc)

        await asyncio.to_thread(_sync)
        logger.info("Neo4j indexes ensured")

    # ------------------------------------------------------------------
    # Idempotent corpus writes (Phase 4 — MERGE + UUID5)
    # ------------------------------------------------------------------

    async def upsert_chunks(self, chunks: list[dict[str, Any]]) -> None:
        """Write Chunk nodes to Neo4j using MERGE (idempotent)."""
        if not chunks:
            return

        def _sync() -> None:
            for c in chunks:
                self._run(
                    "MERGE (ch:Chunk {id:$id}) "
                    "SET ch.doc_id=$doc_id, ch.text=$text",
                    id=c["id"],
                    doc_id=c.get("doc_id", ""),
                    text=(c.get("text") or "")[:3000],
                )

        await asyncio.to_thread(_sync)
        logger.debug("Upserted %d Chunk nodes", len(chunks))

    async def upsert_entities_and_mentions(self, chunks: list[dict[str, Any]]) -> None:
        """Write Entity nodes and MENTIONS edges derived from chunk entity lists (MERGE)."""
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
                    eid = entity_node_id(name)
                    self._run(
                        "MERGE (e:Entity {id:$eid}) "
                        "SET e.name=$name, e.label=$label",
                        eid=eid,
                        name=name,
                        label=label,
                    )
                    self._run(
                        """MATCH (ch:Chunk {id:$cid})
                           MATCH (e:Entity {id:$eid})
                           MERGE (ch)-[:MENTIONS]->(e)""",
                        cid=c["id"],
                        eid=eid,
                    )

        await asyncio.to_thread(_sync)

    async def upsert_topics(self, topics: list[dict[str, Any]]) -> None:
        """Write Topic nodes and IN_COMMUNITY edges (MERGE — idempotent)."""
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
                    eid = entity_node_id(name)
                    self._run(
                        """MATCH (e:Entity {id:$eid})
                           MATCH (tp:Topic {id:$tid})
                           MERGE (e)-[:IN_COMMUNITY]->(tp)""",
                        eid=eid,
                        tid=t["id"],
                    )

        await asyncio.to_thread(_sync)

    async def upsert_co_occurs(self, cooccurrences: list[tuple]) -> None:
        """Write (entity_a, entity_b, weight) CO_OCCURS edges (MERGE — idempotent)."""
        if not cooccurrences:
            return

        def _sync() -> None:
            for a, b, w in cooccurrences:
                ea_id = entity_node_id(a)
                eb_id = entity_node_id(b)
                self._run(
                    """MATCH (ea:Entity {id:$ea_id})
                       MATCH (eb:Entity {id:$eb_id})
                       MERGE (ea)-[r:CO_OCCURS]->(eb)
                       SET r.weight=$w""",
                    ea_id=ea_id,
                    eb_id=eb_id,
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
        """Write SessionEntity nodes and RELATES edges for a live agent session."""

        def _sync() -> None:
            for e in entities:
                self._run(
                    """MERGE (n:SessionEntity {entity_id:$eid, session_id:$sid})
                       SET n.name=$name, n.label=$label,
                           n.confidence=$conf, n.doc_ids=$doc_ids""",
                    eid=e.id,
                    sid=session_id,
                    name=e.name,
                    label=e.label,
                    conf=getattr(e, "confidence", 1.0),
                    doc_ids=getattr(e, "doc_ids", []),
                )
            for r in relations:
                self._run(
                    """MATCH (a:SessionEntity {entity_id:$src, session_id:$sid})
                       MATCH (b:SessionEntity {entity_id:$tgt, session_id:$sid})
                       MERGE (a)-[rel:RELATES {type:$rtype, session_id:$sid}]->(b)""",
                    src=r.source,
                    tgt=r.target,
                    rtype=r.type,
                    sid=session_id,
                )

        await asyncio.to_thread(_sync)

    async def seed_session_from_corpus(
        self, session_id: str, entity_names: list[str]
    ) -> int:
        """Copy named Entity nodes from corpus into session namespace."""

        def _sync() -> int:
            count = 0
            for name in entity_names:
                name_upper = name.strip().upper()
                rows = self._run(
                    "MATCH (e:Entity {name:$name}) RETURN e.label AS label",
                    name=name_upper,
                )
                if not rows:
                    continue
                label = rows[0]["label"] or "Entity"
                eid = entity_node_id(name_upper)
                self._run(
                    """MERGE (n:SessionEntity {entity_id:$eid, session_id:$sid})
                       SET n.name=$name, n.label=$label""",
                    eid=eid,
                    sid=session_id,
                    name=name_upper,
                    label=label,
                )
                count += 1
            return count

        return await asyncio.to_thread(_sync)

    async def merge_session_entities_to_corpus(
        self,
        session_id: str,
        nodes: list[dict[str, Any]],
        edges: list[dict[str, Any]],
    ) -> tuple[int, int]:
        """Promote session nodes and edges into the persistent corpus graph."""

        def _sync() -> tuple[int, int]:
            merged_nodes = 0
            merged_edges = 0
            for n in nodes:
                name = n.get("name", "").strip().upper()
                if not name:
                    continue
                eid = entity_node_id(name)
                self._run(
                    """MERGE (e:Entity {id:$id})
                       SET e.name=$name, e.label=$label""",
                    id=eid,
                    name=name,
                    label=n.get("type", "Entity"),
                )
                merged_nodes += 1
            for e in edges:
                left = e.get("from", "").strip().upper()
                right = e.get("to", "").strip().upper()
                rel = e.get("rel_type", "RELATES")
                if not left or not right:
                    continue
                ea_id = entity_node_id(left)
                eb_id = entity_node_id(right)
                self._run(
                    f"""MATCH (a:Entity {{id:$a_id}})
                        MATCH (b:Entity {{id:$b_id}})
                        MERGE (a)-[r:`{rel}`]->(b)""",
                    a_id=ea_id,
                    b_id=eb_id,
                )
                merged_edges += 1
            return merged_nodes, merged_edges

        return await asyncio.to_thread(_sync)

    # ------------------------------------------------------------------
    # Corpus reads (retrieval path)
    # ------------------------------------------------------------------

    async def get_chunks_for_entity(
        self, entity_name: str, top_k: int = 5
    ) -> list[dict[str, Any]]:
        """Return Chunk nodes linked to *entity_name* via MENTIONS edges."""
        name_upper = entity_name.strip().upper()
        rows = await self._arun(
            """MATCH (ch:Chunk)-[:MENTIONS]->(e:Entity {name:$name})
               RETURN ch.id AS chunk_id, ch.doc_id AS doc_id, ch.text AS text
               LIMIT $k""",
            name=name_upper,
            k=top_k,
        )
        return [dict(r) for r in rows]

    async def get_entity_community(
        self, entity_name: str
    ) -> dict[str, Any] | None:
        """Return the Topic node linked to *entity_name* via IN_COMMUNITY edge."""
        name_upper = entity_name.strip().upper()
        eid = entity_node_id(name_upper)
        rows = await self._arun(
            """MATCH (e:Entity {id:$eid})-[:IN_COMMUNITY]->(tp:Topic)
               RETURN tp.id AS id, tp.label AS label, tp.summary AS summary
               LIMIT 1""",
            eid=eid,
        )
        return dict(rows[0]) if rows else None

    async def count_nodes(self) -> dict[str, int]:
        """Return counts of key node types."""

        async def _count(label: str) -> int:
            rows = await self._arun(f"MATCH (n:{label}) RETURN count(n) AS c")
            return rows[0]["c"] if rows else 0

        chunk_count, entity_count, topic_count, session_count = await asyncio.gather(
            _count("Chunk"),
            _count("Entity"),
            _count("Topic"),
            _count("SessionEntity"),
        )
        return {
            "chunks": chunk_count,
            "entities": entity_count,
            "topics": topic_count,
            "session_entities": session_count,
        }

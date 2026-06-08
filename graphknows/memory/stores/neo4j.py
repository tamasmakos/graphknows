"""Neo4j store for graphknows — corpus reads + session writes + idempotent ingestion.

All sync Neo4j calls are wrapped in asyncio.to_thread for non-blocking operation.
Idempotent upserts use UUID5 deterministic IDs keyed on (doc_id, chunk_hash).
"""

from __future__ import annotations

import asyncio
import logging
import re
import uuid
from collections import defaultdict
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from graphknows.config.adapters.ports.relation_extractor import IRelationExtractor

logger = logging.getLogger(__name__)

# UUID5 namespace for graphknows nodes (fixed sentinel)
_GK_NS = uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")  # NAMESPACE_URL


def chunk_node_id(doc_id: str, chunk_hash: str) -> str:
    """Deterministic UUID5 for a Chunk node — idempotent across pipeline runs."""
    return str(uuid.uuid5(_GK_NS, f"chunk:{doc_id}:{chunk_hash}"))


def entity_node_id(name: str) -> str:
    """Deterministic UUID5 for an Entity node keyed on normalised name."""
    return str(uuid.uuid5(_GK_NS, f"entity:{name.strip().upper()}"))


def _safe_rel_label(raw: str) -> str:
    """Convert a free-text relation string to a valid Cypher relationship type.

    Uppercases, replaces spaces/hyphens with underscores, strips non-alphanumeric
    characters. Returns empty string if nothing valid remains.
    """
    label = raw.upper().replace("-", "_").replace(" ", "_")
    label = re.sub(r"[^\w]", "", label)
    return label


def _write_relation_edges(
    s: Any,
    triplets: list[tuple[str, str, str] | tuple[str, str, str, dict[str, Any]]],
    entity_by_name: dict[str, dict[str, Any]],
) -> int:
    """Write typed directed edges between ENTITY nodes using dynamic Cypher labels.

    Each triplet is (source_name, relation_label, target_name). The relation_label
    becomes an actual Cypher relationship type (e.g. REGULATES, FUNDS). Triplets
    whose endpoints cannot be matched to a known entity are silently skipped.

    Name matching is case-insensitive: the extractor may return uppercased names
    (e.g. "GRAPHRAG") while entity_by_name stores them as-extracted ("GraphRAG").

    Returns:
        Number of edges written.
    """
    # Build a case-insensitive index so we can resolve canonical names.
    canonical: dict[str, str] = {k.upper(): k for k in entity_by_name}

    edges = 0
    for item in triplets:
        src, rel, tgt = item[0], item[1], item[2]
        metadata = item[3] if len(item) > 3 and isinstance(item[3], dict) else {}
        src_canon = canonical.get(src.upper(), src)
        tgt_canon = canonical.get(tgt.upper(), tgt)
        safe = _safe_rel_label(rel)
        if not safe:
            continue
        src_type = metadata.get("source_type") or entity_by_name.get(src_canon, {}).get("type", "ENTITY")
        tgt_type = metadata.get("target_type") or entity_by_name.get(tgt_canon, {}).get("type", "ENTITY")
        s.run(
            "MERGE (e1:ENTITY {name: $src}) "
            "SET e1.type = coalesce(e1.type, $src_type) "
            "MERGE (e2:ENTITY {name: $tgt}) "
            "SET e2.type = coalesce(e2.type, $tgt_type) "
            f"MERGE (e1)-[r:{safe}]->(e2) "
            "SET r.weight = coalesce(r.weight, 0) + 1, "
            "r.confidence = coalesce($confidence, r.confidence), "
            "r.evidence = coalesce($evidence, r.evidence), "
            "r.reasoning = coalesce($reasoning, r.reasoning)",
            src=src_canon,
            tgt=tgt_canon,
            src_type=src_type,
            tgt_type=tgt_type,
            confidence=metadata.get("confidence"),
            evidence=metadata.get("evidence"),
            reasoning=metadata.get("reasoning"),
        )
        edges += 1
    return edges


def _compute_silhouette(
    names: list[str],
    name_idx: dict[str, int],
    communities: dict[int, list[str]],
    rows: list[dict[str, Any]],
    weights: list[float],
) -> float | None:
    """Compute silhouette score from weighted edge co-occurrence data.

    Uses a distance matrix: d(i,j) = 1 - weight/max_weight. Returns None on
    any failure (missing optional deps, single community, etc.).
    """
    try:
        import numpy as np
        from sklearn.metrics import silhouette_score

        labels = [0] * len(names)
        for comm_id, members in communities.items():
            for m in members:
                labels[name_idx[m]] = comm_id
        if len(set(labels)) < 2:
            return None
        max_w = max(weights) if weights else 1.0
        adj = [[1.0] * len(names) for _ in range(len(names))]
        for edge_data, w in zip(rows, weights, strict=False):
            i, j = name_idx[edge_data["a"]], name_idx[edge_data["b"]]
            d = 1.0 - w / max_w
            adj[i][j] = d
            adj[j][i] = d
            adj[i][i] = 0.0
            adj[j][j] = 0.0
        return float(silhouette_score(np.array(adj), labels, metric="precomputed"))
    except Exception as exc:
        logger.warning("Silhouette computation skipped: %s", exc)
        return None


async def _write_pgvector_topic(
    pg_store: Any,
    topic: dict[str, Any],
    *,
    title: str,
    summary: str,
    comm_id: int,
) -> None:
    """Write a topic row to pgvector, embedding summary/title text when available."""
    embedding_text = (summary or "").strip() or (title or "").strip()
    if embedding_text:
        try:
            from graphknows.memory.stores.embedder import embed_one

            topic["embedding"] = await asyncio.to_thread(embed_one, embedding_text)
        except Exception as exc:
            logger.warning("Topic embedding failed for community %d: %s", comm_id, exc)

    await pg_store.write_ltm_topic(topic)


def _write_evidence_graph(
    s: Any,
    session_id: str,
    stm_data: dict[str, Any],
    entity_by_name: dict[str, dict[str, Any]],
    now: str,
) -> tuple[int, int, dict[str, list[str]]]:
    """MERGE SESSION→FILE→CHUNK→ENTITY into *s*. Returns (nodes, edges, chunk_to_ents).

    CO_OCCURS edges are NOT written here. Relation extraction runs separately in
    flush.py using the returned chunk_to_ents map.
    """
    nodes, edges = 1, 0
    s.run("MERGE (sess:SESSION {id: $sid}) SET sess.flushed_at = $now", sid=session_id, now=now)
    for f in stm_data.get("files", []):
        s.run(
            "MERGE (f:FILE {id: $fid}) "
            "SET f.title=$title, f.parser_name=$parser_name, f.source_hint=$source_hint "
            "WITH f MATCH (sess:SESSION {id: $sid}) MERGE (sess)-[:IN_SESSION]->(f)",
            fid=f["id"],
            title=f.get("title", ""),
            parser_name=f.get("parser_name", ""),
            source_hint=f.get("source_hint", ""),
            sid=session_id,
        )
        nodes += 1
        edges += 1
    for c in stm_data.get("chunks", []):
        s.run(
            "MERGE (c:CHUNK {id: $cid}) "
            "SET c.text=$text, c.heading_path=$hp, c.chunk_index=$ci "
            "WITH c MATCH (f:FILE {id: $fid}) MERGE (f)-[:HAS_CHUNK]->(c)",
            cid=c["id"],
            text=c.get("text", ""),
            hp=c.get("heading_path", ""),
            ci=c.get("chunk_index", 0),
            fid=c["file_id"],
        )
        nodes += 1
        edges += 1
    written: set[str] = set()
    chunk_to_ents: dict[str, list[str]] = defaultdict(list)
    for ce in stm_data.get("chunk_entity_edges", []):
        ename = ce["entity_name"]
        chunk_to_ents[ce["chunk_id"]].append(ename)
        em = entity_by_name.get(ename, {"type": "UNKNOWN", "confidence": 1.0})
        if ename not in written:
            s.run(
                "MERGE (e:ENTITY {name: $n}) SET e.type=$t, e.confidence=$c",
                n=ename,
                t=em.get("type", "UNKNOWN"),
                c=em.get("confidence", 1.0),
            )
            nodes += 1
            written.add(ename)
        s.run(
            "MATCH (c:CHUNK {id: $cid}), (e:ENTITY {name: $n}) MERGE (c)-[:HAS_ENTITY]->(e)",
            cid=ce["chunk_id"],
            n=ename,
        )
        edges += 1
    return nodes, edges, dict(chunk_to_ents)


def _write_interaction_graph(
    s: Any,
    session_id: str,
    stm_data: dict[str, Any],
    entity_by_name: dict[str, dict[str, Any]],
    base_written: set[str],
) -> tuple[int, int]:
    """MERGE SESSION→DIALOG→TURN + NEXT + TURN→ENTITY into *s*. Returns (nodes, edges)."""
    nodes, edges = 0, 0
    for d in stm_data.get("dialogs", []):
        s.run(
            "MERGE (d:DIALOG {id: $did}) SET d.session_id=$sid, d.started_at=$sa "
            "WITH d MATCH (sess:SESSION {id: $sid}) MERGE (sess)-[:IN_SESSION]->(d)",
            did=d["id"],
            sid=session_id,
            sa=d.get("started_at", ""),
        )
        nodes += 1
        edges += 1
    for t in stm_data.get("turns", []):
        s.run(
            "MERGE (t:TURN {id: $tid}) "
            "SET t.session_id=$sid, t.turn_index=$ti, t.role=$r, t.text=$txt "
            "WITH t MATCH (d:DIALOG {id: $did}) MERGE (d)-[:HAS_TURN]->(t)",
            tid=t["id"],
            sid=session_id,
            ti=t.get("turn_index", 0),
            r=t.get("role", "unknown"),
            txt=t.get("text", ""),
            did=t["dialog_id"],
        )
        nodes += 1
        edges += 1
    for ne in stm_data.get("next_edges", []):
        s.run(
            "MATCH (t1:TURN {id: $a}), (t2:TURN {id: $b}) MERGE (t1)-[:NEXT]->(t2)",
            a=ne["from_turn_id"],
            b=ne["to_turn_id"],
        )
        edges += 1
    written = set(base_written)
    for te in stm_data.get("turn_entity_edges", []):
        ename = te["entity_name"]
        em = entity_by_name.get(ename, {"type": "UNKNOWN", "confidence": 1.0})
        if ename not in written:
            s.run(
                "MERGE (e:ENTITY {name: $n}) SET e.type=$t, e.confidence=$c",
                n=ename,
                t=em.get("type", "UNKNOWN"),
                c=em.get("confidence", 1.0),
            )
            nodes += 1
            written.add(ename)
        s.run(
            "MATCH (t:TURN {id: $tid}), (e:ENTITY {name: $n}) MERGE (t)-[:HAS_ENTITY]->(e)",
            tid=te["turn_id"],
            n=ename,
        )
        edges += 1
    return nodes, edges


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
        relation_extractor: IRelationExtractor | None = None,
    ) -> None:
        from neo4j import GraphDatabase

        self._driver = GraphDatabase.driver(uri, auth=(user, password))
        self._db = database
        self._relation_extractor = relation_extractor

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run(self, cypher: str, **params: Any) -> list[Any]:
        with self._driver.session(database=self._db) as s:
            return list(s.run(cypher, **params))

    async def _arun(self, cypher: str, **params: Any) -> list[Any]:
        return await asyncio.to_thread(self._run, cypher, **params)

    def close(self) -> None:
        """Close the Neo4j Bolt driver connection."""
        self._driver.close()

    # ------------------------------------------------------------------
    # Indexes
    # ------------------------------------------------------------------

    async def ensure_indexes(self) -> None:
        """Create corpus + session indexes idempotently."""

        def _sync() -> None:
            for stmt in [
                "CREATE INDEX chunk_id IF NOT EXISTS FOR (c:CHUNK) ON (c.id)",
                "CREATE INDEX entity_name IF NOT EXISTS FOR (e:ENTITY) ON (e.name)",
                "CREATE INDEX topic_id IF NOT EXISTS FOR (t:TOPIC) ON (t.id)",
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
                    "MERGE (ch:CHUNK {id:$id}) SET ch.doc_id=$doc_id, ch.text=$text",
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
                        "MERGE (e:ENTITY {id:$eid}) SET e.name=$name, e.label=$label",
                        eid=eid,
                        name=name,
                        label=label,
                    )
                    self._run(
                        """MATCH (ch:CHUNK {id:$cid})
                           MATCH (e:ENTITY {id:$eid})
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
                    """MERGE (tp:TOPIC {id:$id})
                       SET tp.label=$label, tp.summary=$summary,
                           tp.community_id=$community_id,
                           tp.community_hash=$community_hash,
                           tp.modularity=$modularity,
                           tp.gds_modularity=$gds_modularity,
                           tp.community_algorithm=$community_algorithm,
                           tp.silhouette=$silhouette""",
                    id=t["id"],
                    label=t.get("label", ""),
                    summary=t.get("summary", ""),
                    community_id=t.get("community_id"),
                    community_hash=t.get("community_hash"),
                    modularity=t.get("modularity"),
                    gds_modularity=t.get("gds_modularity", t.get("modularity")),
                    community_algorithm=t.get("community_algorithm", ""),
                    silhouette=t.get("silhouette"),
                )
                for name in t.get("members", []):
                    eid = entity_node_id(name)
                    self._run(
                        """MATCH (e:ENTITY {id:$eid})
                           MATCH (tp:TOPIC {id:$tid})
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
                    """MATCH (ea:ENTITY {id:$ea_id})
                       MATCH (eb:ENTITY {id:$eb_id})
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

    async def seed_session_from_corpus(self, session_id: str, entity_names: list[str]) -> int:
        """Copy named Entity nodes from corpus into session namespace."""

        def _sync() -> int:
            count = 0
            for name in entity_names:
                name_upper = name.strip().upper()
                rows = self._run(
                    "MATCH (e:ENTITY {name:$name}) RETURN e.label AS label",
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
        """Promote session nodes and edges into the persistent corpus graph.

        Records temporal provenance on each merged Entity node:
        ``ingested_at``, ``contributing_session_id``, ``merged_session_count``.
        """
        ingested_at = datetime.now(UTC).isoformat()

        def _sync() -> tuple[int, int]:
            merged_nodes = 0
            merged_edges = 0
            for n in nodes:
                name = n.get("name", "").strip().upper()
                if not name:
                    continue
                eid = entity_node_id(name)
                self._run(
                    """MERGE (e:ENTITY {id:$id})
                       SET e.name=$name, e.label=$label,
                           e.ingested_at=$ingested_at,
                           e.contributing_session_id=$session_id,
                           e.merged_session_count =
                               coalesce(e.merged_session_count, 0) + 1""",
                    id=eid,
                    name=name,
                    label=n.get("type", "Entity"),
                    ingested_at=ingested_at,
                    session_id=session_id,
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
                    f"""MATCH (a:ENTITY {{id:$a_id}})
                        MATCH (b:ENTITY {{id:$b_id}})
                        MERGE (a)-[r:`{rel}`]->(b)""",
                    a_id=ea_id,
                    b_id=eb_id,
                )
                merged_edges += 1
            return merged_nodes, merged_edges

        return await asyncio.to_thread(_sync)

    async def preview_session_merge(
        self,
        session_id: str,
        nodes: list[dict[str, Any]],
        edges: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Return a preview of what ``merge_session_entities_to_corpus`` would do.

        Does NOT write anything.  Checks how many session nodes already exist
        in the corpus so callers can gauge the merge impact.

        Args:
            session_id: Session namespace identifier.
            nodes: Memgraph entity nodes for this session.
            edges: Memgraph relation edges for this session.

        Returns:
            Dict with node_count, edge_count, existing_overlap_count, sample_nodes.
        """
        names = [n.get("name", "").strip().upper() for n in nodes if n.get("name")]

        def _count_existing(name_list: list[str]) -> int:
            if not name_list:
                return 0
            rows = self._run(
                "UNWIND $names AS n MATCH (e:ENTITY {name:n}) RETURN count(e) AS c",
                names=name_list,
            )
            return int(rows[0]["c"]) if rows else 0

        existing = await asyncio.to_thread(_count_existing, names)
        return {
            "session_id": session_id,
            "node_count": len(nodes),
            "edge_count": len(edges),
            "existing_overlap_count": existing,
            "sample_nodes": [n.get("name") for n in nodes[:10]],
        }

    # ------------------------------------------------------------------
    # Corpus reads (retrieval path)
    # ------------------------------------------------------------------

    async def get_chunk_entities(self, chunk_id: str) -> list[dict[str, Any]]:
        """Return Entity nodes linked to a Chunk via MENTIONS edges.

        Args:
            chunk_id: The Chunk node id to look up.

        Returns:
            List of dicts with name and label.
        """
        rows = await self._arun(
            """MATCH (ch:CHUNK {id: $cid})-[:MENTIONS]->(e:ENTITY)
               RETURN e.name AS name, e.label AS label
               LIMIT 20""",
            cid=chunk_id,
        )
        return [{"name": r["name"], "label": r["label"] or "Entity"} for r in rows]

    async def get_entity_neighbors(
        self,
        entity_name: str,
        depth: int = 1,
        relationship_types: list[str] | None = None,
    ) -> dict[str, Any]:
        """Return neighboring Entity/Topic nodes reachable from *entity_name*.

        Args:
            entity_name: Starting entity (case-insensitive — normalised to UPPER).
            depth: Number of hops to traverse (capped 1-3).
            relationship_types: Optional allowlist of relationship type names.
                Only alphanumeric + underscore names accepted (injection prevention).

        Returns:
            Dict with entity, depth, neighbor_count, and neighbors list.
        """
        name_upper = entity_name.strip().upper()
        depth = max(1, min(depth, 3))  # cap at 3 hops to prevent runaway queries

        # Validate relationship type names — alphanumeric + underscore only
        safe_types: list[str] = []
        if relationship_types:
            safe_types = [
                t for t in relationship_types if t and all(c.isalnum() or c == "_" for c in t)
            ]

        # Build the relationship pattern (safe: types validated above)
        if safe_types:
            type_list = "|".join(safe_types)
            rel_pattern = f"[r:{type_list}*1..{depth}]"
        else:
            rel_pattern = f"[r*1..{depth}]"

        rows = await self._arun(
            f"""MATCH (start:ENTITY {{name: $name}})-{rel_pattern}-(neighbor)
               WHERE neighbor:ENTITY OR neighbor:TOPIC
               RETURN
                   neighbor.name AS name,
                   labels(neighbor)[0] AS node_label,
                   [rel IN r | type(rel)] AS rel_path
               LIMIT 100""",
            name=name_upper,
        )

        nodes: list[dict] = []
        seen: set[str] = set()
        for row in rows:
            n_name = row["name"] or ""
            if n_name and n_name not in seen:
                seen.add(n_name)
                nodes.append({"name": n_name, "label": row["node_label"] or "Entity"})

        return {
            "entity": entity_name,
            "depth": depth,
            "neighbor_count": len(nodes),
            "neighbors": nodes,
        }

    async def get_chunks_for_entity(self, entity_name: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Return Chunk nodes linked to *entity_name* via MENTIONS edges."""
        name_upper = entity_name.strip().upper()
        rows = await self._arun(
            """MATCH (ch:CHUNK)-[:MENTIONS]->(e:ENTITY {name:$name})
               RETURN ch.id AS chunk_id, ch.doc_id AS doc_id, ch.text AS text
               LIMIT $k""",
            name=name_upper,
            k=top_k,
        )
        return [dict(r) for r in rows]

    async def get_entity_community(self, entity_name: str) -> dict[str, Any] | None:
        """Return the Topic node linked to *entity_name* via IN_COMMUNITY edge."""
        name_upper = entity_name.strip().upper()
        eid = entity_node_id(name_upper)
        rows = await self._arun(
            """MATCH (e:ENTITY)
               WHERE e.id = $eid OR toLower(e.name) = toLower($name)
               MATCH (e)-[:IN_TOPIC|IN_COMMUNITY]->(tp:TOPIC)
               RETURN tp.id AS id, tp.label AS label, tp.summary AS summary,
                      tp.community_id AS community_id,
                      tp.modularity AS modularity,
                      tp.gds_modularity AS gds_modularity,
                      tp.community_algorithm AS community_algorithm,
                      tp.entity_names AS entity_names,
                      tp.entity_count AS entity_count
               LIMIT 1""",
            eid=eid,
            name=entity_name.strip(),
        )
        return dict(rows[0]) if rows else None

    async def write_session_subgraph(
        self, session_id: str, stm_data: dict[str, Any]
    ) -> dict[str, int]:
        """MERGE a full STM subgraph into Neo4j LTM.

        Idempotent — all writes use MERGE.  Returns node/edge counts written.

        Args:
            session_id: Session identifier.
            stm_data: Dict produced by MemgraphStore.read_session_subgraph().

        Returns:
            {"nodes": int, "edges": int}
        """
        now = datetime.now(UTC).isoformat()
        entity_by_name: dict[str, dict[str, Any]] = {
            e["name"]: e for e in stm_data.get("entities", [])
        }

        def _sync() -> dict[str, int]:
            with self._driver.session() as s:
                n1, e1, _chunk_to_ents = _write_evidence_graph(
                    s, session_id, stm_data, entity_by_name, now
                )
                written: set[str] = {e["name"] for e in stm_data.get("entities", [])}
                n2, e2 = _write_interaction_graph(s, session_id, stm_data, entity_by_name, written)
                relation_edges = stm_data.get("relation_edges", [])
                e3 = 0
                if relation_edges:
                    triplets = [
                        (r["source"], r["relation"], r["target"])
                        for r in relation_edges
                        if r.get("source") and r.get("relation") and r.get("target")
                    ]
                    if triplets:
                        e3 = _write_relation_edges(s, triplets, entity_by_name)
            return {"nodes": n1 + n2, "edges": e1 + e2 + e3}

        return await asyncio.to_thread(_sync)

    async def extract_and_write_relations(
        self,
        stm_data: dict[str, Any],
        openrouter_key: str,
        llm_model: str = "openrouter/meta-llama/llama-3.1-8b-instruct",
    ) -> int:
        """Run DSPyExtractor per chunk; write dynamic typed ENTITY→ENTITY edges.

        Each chunk's text is passed to the DSPy module. The extractor loads its RDF
        ontology file and returns ontology-constrained (source, relation_label, target)
        triplets. These become actual Cypher relationship types via _write_relation_edges.

        Raises:
            Any exception from DSPyExtractor.extract_relations propagates immediately.
            This is intentional — partial relation writes are worse than no writes.

        Returns:
            Total number of relation edges written.
        """
        entity_by_name: dict[str, dict[str, Any]] = {
            e["name"]: e for e in stm_data.get("entities", [])
        }
        extractor = self._relation_extractor
        if extractor is None:
            raise RuntimeError(
                "Neo4jStore requires an injected relation_extractor for relation extraction "
                f"(llm_model={llm_model}, openrouter_key_configured={bool(openrouter_key)})"
            )

        all_triplets: list[tuple[str, str, str, dict[str, Any]]] = []

        for chunk in stm_data.get("chunks", []):
            # extract_relations returns (relations_list, nodes_data_list, audit_dict)
            relations, nodes_data, _audit = await extractor.extract_relations(
                text=chunk["text"],
            )
            for node in nodes_data:
                node_id = str(node.get("id") or "").strip()
                if node_id and node_id not in entity_by_name:
                    entity_by_name[node_id] = {
                        "name": node_id,
                        "type": node.get("type") or "ENTITY",
                        "confidence": 1.0,
                    }
            for item in relations:
                # item is a tuple: (source, relation, target, metadata_dict)
                src, rel, tgt = item[0], item[1], item[2]
                props = item[3] if len(item) > 3 and isinstance(item[3], dict) else {}
                if src and rel and tgt:
                    all_triplets.append((src, rel, tgt, props))

        if not all_triplets:
            return 0

        def _sync_write() -> int:
            with self._driver.session() as s:
                return _write_relation_edges(s, all_triplets, entity_by_name)

        return await asyncio.to_thread(_sync_write)

    async def summarize_topics(
        self,
        session_id: str,
        openrouter_key: str,
        llm_model: str = "meta-llama/llama-3.1-8b-instruct",
        pg_store: Any = None,
        gds_modularity: float | None = None,
        community_algorithm: str | None = None,
    ) -> int:
        """Summarize graph communities and write TOPIC nodes.

        GDS-owned communities are preferred: if ``ENTITY.community_id`` exists,
        those yielded communities define topics. If GDS has not run, a local
        Leiden fallback is used and the fallback community IDs are written back
        to entities for notebook and UI inspection.

        Only communities whose member set changed since the previous flush are
        re-summarized. Topic nodes always receive the available modularity and
        community algorithm metadata.
        """
        import hashlib
        import uuid as _uuid

        communities, local_modularity, silhouette, algorithm = await self._topic_communities()
        if not communities:
            logger.info("summarize_topics %s: no communities found; skipping", session_id)
            return 0

        modularity = gds_modularity if gds_modularity is not None else local_modularity
        algorithm = (community_algorithm or "").strip() or algorithm

        existing_hashes: dict[int, str] = {}
        try:
            hash_rows = await self._arun(
                "MATCH (t:TOPIC) "
                "WHERE t.community_id IS NOT NULL AND t.community_hash IS NOT NULL "
                "RETURN t.community_id AS cid, t.community_hash AS h"
            )
            existing_hashes = {int(r["cid"]): r["h"] for r in hash_rows}
        except Exception as exc:
            logger.warning("summarize_topics: could not load existing hashes: %s", exc)

        summarizer = None
        if openrouter_key:
            from graphknows.memory.ltm.topic_summarizer import TopicSummarizer

            summarizer = TopicSummarizer(openrouter_key, llm_model)
        topics_created = 0
        for comm_id, members in communities.items():
            if len(members) < 2:
                continue

            comm_hash = hashlib.sha1(
                "|".join(sorted(members)).encode(), usedforsecurity=False
            ).hexdigest()
            if existing_hashes.get(comm_id) == comm_hash:
                logger.debug("summarize_topics: community %d unchanged; refreshing metadata", comm_id)
                await self._refresh_topic_metadata(
                    comm_id=comm_id,
                    modularity=modularity,
                    gds_modularity=gds_modularity,
                    community_algorithm=algorithm,
                )
                continue

            evidence_rows = await self._arun(
                "MATCH (c:CHUNK)-[:HAS_ENTITY]->(e:ENTITY) "
                "WHERE e.name IN $members "
                "WITH DISTINCT c "
                "RETURN c.text AS text, c.heading_path AS hp "
                "ORDER BY rand() LIMIT 100",
                members=members,
            )
            passages = "\n\n".join(
                (f"[{r['hp']}] " if r.get("hp") else "") + (r.get("text") or "")
                for r in evidence_rows
            )

            fallback_title = ", ".join(members[:5])
            title, summary = fallback_title, ""
            if summarizer is not None:
                title, summary = await summarizer.summarize(
                    members=members,
                    passages=passages,
                    fallback_title=fallback_title,
                )

            topic_id = str(_uuid.uuid5(_GK_NS, f"topic:{session_id}:{comm_id}"))
            topic = {
                "id": topic_id,
                "topic_id": topic_id,
                "label": title,
                "summary": summary,
                "community_id": comm_id,
                "community_hash": comm_hash,
                "entity_count": len(members),
                "entity_names": members,
                "session_ids": [session_id],
                "members": members,
                "modularity": modularity,
                "gds_modularity": gds_modularity,
                "community_algorithm": algorithm,
                "silhouette": silhouette,
            }
            await self.upsert_topics([topic])

            def _link_topic(tid: str, mbs: list[str]) -> None:
                with self._driver.session() as s:
                    for name in mbs:
                        s.run(
                            "MATCH (tp:TOPIC {id: $tid}), (e:ENTITY {name: $name}) "
                            "MERGE (e)-[:IN_TOPIC]->(tp)",
                            tid=tid,
                            name=name,
                        )

            await asyncio.to_thread(_link_topic, topic_id, members)

            if pg_store is not None:
                await _write_pgvector_topic(
                    pg_store,
                    topic,
                    title=title,
                    summary=summary,
                    comm_id=comm_id,
                )

            topics_created += 1

        return topics_created

    async def _topic_communities(self) -> tuple[dict[int, list[str]], float | None, float | None, str]:
        """Return topic communities from GDS, or local Leiden fallback."""
        gds_rows = await self._arun(
            "MATCH (e:ENTITY) "
            "WHERE e.community_id IS NOT NULL "
            "RETURN e.community_id AS cid, collect(DISTINCT e.name) AS members"
        )
        gds_communities: dict[int, list[str]] = {}
        for r in gds_rows:
            cid = r.get("cid") if hasattr(r, "get") else r["cid"]
            members = r.get("members", []) if hasattr(r, "get") else r["members"]
            if cid is not None:
                gds_communities[int(cid)] = sorted(n for n in members if n)
        if any(len(members) >= 2 for members in gds_communities.values()):
            return gds_communities, None, None, "gds_leiden"

        import igraph as ig
        import leidenalg

        rel_rows = await self._arun(
            "MATCH (a:ENTITY)-[r]->(b:ENTITY) "
            "WHERE type(r) <> 'CO_OCCURS' "
            "RETURN a.name AS a, b.name AS b, coalesce(r.weight, 1.0) AS weight",
        )
        if not rel_rows:
            return {}, None, None, "leidenalg"
        algorithm = "leidenalg"

        names: list[str] = sorted({r["a"] for r in rel_rows} | {r["b"] for r in rel_rows})
        name_idx = {n: i for i, n in enumerate(names)}
        g = ig.Graph(n=len(names), directed=False)
        g.vs["name"] = names
        edges: list[tuple[int, int]] = []
        weights: list[float] = []
        for r in rel_rows:
            ai, bi = name_idx[r["a"]], name_idx[r["b"]]
            if ai != bi:
                edges.append((ai, bi))
                weights.append(max(0.0, float(r["weight"])))
        g.add_edges(edges)

        partition = leidenalg.find_partition(
            g, leidenalg.ModularityVertexPartition, weights=weights
        )
        communities: dict[int, list[str]] = {
            i: sorted(g.vs[vi]["name"] for vi in partition[i]) for i in range(len(partition))
        }
        modularity = float(partition.quality())
        silhouette = _compute_silhouette(names, name_idx, communities, rel_rows, weights)

        def _write_entity_communities() -> None:
            with self._driver.session() as s:
                for cid, members in communities.items():
                    for name in members:
                        s.run(
                            "MATCH (e:ENTITY {name: $name}) SET e.community_id = $cid",
                            name=name,
                            cid=cid,
                        )

        await asyncio.to_thread(_write_entity_communities)
        return communities, modularity, silhouette, algorithm

    async def _refresh_topic_metadata(
        self,
        *,
        comm_id: int,
        modularity: float | None,
        gds_modularity: float | None,
        community_algorithm: str,
    ) -> None:
        """Keep unchanged topic nodes aligned with the latest GDS metrics."""
        await self._arun(
            "MATCH (t:TOPIC {community_id: $comm_id}) "
            "SET t.modularity = $modularity, "
            "    t.gds_modularity = $gds_modularity, "
            "    t.community_algorithm = $community_algorithm",
            comm_id=comm_id,
            modularity=modularity,
            gds_modularity=gds_modularity,
            community_algorithm=community_algorithm,
        )

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

    # ------------------------------------------------------------------
    # Temporal SEGMENT reads (E6)
    # ------------------------------------------------------------------

    async def get_segment_info(self, segment_id: str) -> dict[str, Any] | None:
        """Return provenance + statistics for a SEGMENT node by *segment_id*."""
        rows = await self._arun(
            """MATCH (s:SEGMENT {segment_id:$sid})
               OPTIONAL MATCH (s)-[:CONTAINS_DOC]->(d:DOC)
               RETURN s.segment_id AS segment_id,
                      s.ingested_at AS ingested_at,
                      s.run_id AS run_id,
                      s.git_revision AS git_revision,
                      s.config_hash AS config_hash,
                      s.pipeline_version AS pipeline_version,
                      s.source_count AS source_count,
                      s.input_dir AS input_dir,
                      s.taxonomy_name AS taxonomy_name,
                      count(d) AS doc_count""",
            sid=segment_id,
        )
        if not rows:
            return None
        row = dict(rows[0])
        row["doc_count"] = int(row.get("doc_count") or 0)
        return row

    async def list_segments(self, limit: int = 20) -> list[dict[str, Any]]:
        """List SEGMENT nodes ordered by ingestion time descending."""
        rows = await self._arun(
            """MATCH (s:SEGMENT)
               RETURN s.segment_id AS segment_id,
                      s.ingested_at AS ingested_at,
                      s.run_id AS run_id,
                      s.taxonomy_name AS taxonomy_name,
                      s.source_count AS source_count
               ORDER BY s.ingested_at DESC
               LIMIT $limit""",
            limit=limit,
        )
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # LTM entity / topic reads (used by MCP tools)
    # ------------------------------------------------------------------

    def get_entity_by_name(self, name: str) -> dict[str, Any] | None:
        """Return properties of an ENTITY node by case-insensitive name match.

        Also appends ``segment_count`` — the number of distinct Segment nodes
        the entity has appeared in via APPEARED_IN edges.

        Args:
            name: Entity name (case-insensitive).

        Returns:
            Dict of entity properties with ``segment_count`` appended, or None
            if no matching entity is found.
        """
        rows = self._run(
            "MATCH (e:ENTITY) "
            "WHERE toLower(e.name) = toLower($name) "
            "OPTIONAL MATCH (e)-[:APPEARED_IN]->(seg:Segment) "
            "RETURN properties(e) AS props, count(DISTINCT seg) AS segment_count",
            name=name,
        )
        if not rows:
            return None
        row = rows[0]
        props = dict(row["props"])
        props["segment_count"] = row["segment_count"]
        return props

    def list_topic_nodes(self, min_entities: int = 3) -> list[dict[str, Any]]:
        """List TOPIC nodes whose entity_count meets the minimum threshold.

        Results are ordered by entity_count descending.

        Args:
            min_entities: Minimum entity count threshold (inclusive).

        Returns:
            List of dicts with ``community_id``, ``summary``, and
            ``entity_count`` keys.
        """
        rows = self._run(
            "MATCH (t:TOPIC) "
            "OPTIONAL MATCH (e)-[:IN_TOPIC]->(t) "
            "WITH t, count(DISTINCT e) AS computed_entity_count "
            "RETURN properties(t) AS props, computed_entity_count"
        )
        topics = []
        for row in rows:
            props = dict(row.get("props") or {})
            entity_count = int(props.get("entity_count") or row.get("computed_entity_count") or 0)
            if entity_count < min_entities:
                continue
            topics.append(
                {
                    "topic_id": props.get("id"),
                    "label": props.get("label"),
                    "community_id": props.get("community_id"),
                    "summary": props.get("summary"),
                    "entity_count": entity_count,
                    "entity_names": props.get("entity_names") or [],
                    "community_hash": props.get("community_hash"),
                    "modularity": props.get("modularity"),
                    "gds_modularity": props.get("gds_modularity"),
                    "community_algorithm": props.get("community_algorithm"),
                    "silhouette": props.get("silhouette"),
                }
            )
        return sorted(topics, key=lambda t: t["entity_count"], reverse=True)

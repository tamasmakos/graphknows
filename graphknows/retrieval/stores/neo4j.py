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
    triplets: list[tuple[str, str, str]],
    entity_by_name: dict[str, dict[str, Any]],
) -> int:
    """Write typed directed edges between ENTITY nodes using dynamic Cypher labels.

    Each triplet is (source_name, relation_label, target_name). The relation_label
    becomes an actual Cypher relationship type (e.g. REGULATES, FUNDS). Triplets
    whose endpoints are not in entity_by_name are silently skipped.

    Returns:
        Number of edges written.
    """
    edges = 0
    for src, rel, tgt in triplets:
        if src not in entity_by_name or tgt not in entity_by_name:
            continue
        safe = _safe_rel_label(rel)
        if not safe:
            continue
        s.run(
            f"MATCH (e1:ENTITY {{name: $src}}), (e2:ENTITY {{name: $tgt}}) "
            f"MERGE (e1)-[r:{safe}]->(e2) "
            f"SET r.weight = coalesce(r.weight, 0) + 1",
            src=src,
            tgt=tgt,
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
                           tp.modularity=$modularity,
                           tp.silhouette=$silhouette""",
                    id=t["id"],
                    label=t.get("label", ""),
                    summary=t.get("summary", ""),
                    community_id=t.get("community_id"),
                    modularity=t.get("modularity"),
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
            """MATCH (e:ENTITY {id:$eid})-[:IN_COMMUNITY]->(tp:TOPIC)
               RETURN tp.id AS id, tp.label AS label, tp.summary AS summary
               LIMIT 1""",
            eid=eid,
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
            return {"nodes": n1 + n2, "edges": e1 + e2}

        return await asyncio.to_thread(_sync)

    async def extract_and_write_relations(
        self,
        stm_data: dict[str, Any],
        openrouter_key: str,
        llm_model: str = "openrouter/meta-llama/llama-3.1-8b-instruct",
    ) -> int:
        """Run DSPyExtractor per chunk; write dynamic typed ENTITY→ENTITY edges.

        Each chunk's entity list is used as hints for the DSPy module. The extractor
        returns (source, relation_label, target) triplets where relation_label is a
        taxonomy-driven semantic type (e.g. REGULATES, FUNDS, APPOINTS). These become
        actual Cypher relationship types via _write_relation_edges.

        Raises:
            Any exception from DSPyExtractor.extract_relations propagates immediately.
            This is intentional — partial relation writes are worse than no writes.

        Returns:
            Total number of relation edges written.
        """
        from graphknows.graphgen.pipeline.entity_relation.extractors import (
            DSPyExtractor,
        )

        entity_by_name: dict[str, dict[str, Any]] = {
            e["name"]: e for e in stm_data.get("entities", [])
        }
        chunk_to_ents: dict[str, list[str]] = defaultdict(list)
        for ce in stm_data.get("chunk_entity_edges", []):
            chunk_to_ents[ce["chunk_id"]].append(ce["entity_name"])

        extractor = DSPyExtractor(
            config={
                "llm": {"model": llm_model},
                "openrouter_api_key": openrouter_key,
            }
        )

        all_triplets: list[tuple[str, str, str]] = []
        for chunk in stm_data.get("chunks", []):
            ents = chunk_to_ents.get(chunk["id"], [])
            if len(ents) < 2:
                continue  # need at least 2 entities to form a relation
            # extract_relations returns (relations_list, nodes_data_list, audit_dict)
            relations, _, _audit = await extractor.extract_relations(
                text=chunk["text"],
                keywords=ents,
                entities=ents,
            )
            for item in relations:
                # item is a tuple: (source, relation, target, metadata_dict)
                src, rel, tgt = item[0], item[1], item[2]
                if src and rel and tgt:
                    all_triplets.append((src, rel, tgt))

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
    ) -> int:
        """Detect entity communities via Leiden, optionally summarize with LLM, write TOPICs.

        Args:
            session_id: The session to summarize.
            openrouter_key: OpenRouter API key. If empty, LLM step is skipped.
            llm_model: Model identifier passed to OpenRouter.
            pg_store: Optional PGVectorStore — if provided, writes ltm_topics rows.

        Returns:
            Number of topics created.
        """
        import json
        import uuid as _uuid

        import igraph as ig
        import leidenalg

        # Step 1: fetch co-occurrence pairs from Neo4j
        rows = await self._arun(
            "MATCH (sess:SESSION {id: $sid})-[:IN_SESSION]->(f:FILE)"
            "-[:HAS_CHUNK]->(c:CHUNK)-[:HAS_ENTITY]->(e1:ENTITY) "
            "MATCH (c)-[:HAS_ENTITY]->(e2:ENTITY) "
            "WHERE e1.name < e2.name "
            "RETURN e1.name AS a, e2.name AS b, count(c) AS weight",
            sid=session_id,
        )
        if not rows:
            return 0

        # Step 2: build igraph + run Leiden
        names: list[str] = sorted({r["a"] for r in rows} | {r["b"] for r in rows})
        name_idx = {n: i for i, n in enumerate(names)}
        g = ig.Graph(n=len(names))
        g.vs["name"] = names
        g.add_edges([(name_idx[r["a"]], name_idx[r["b"]]) for r in rows])
        weights = [float(r["weight"]) for r in rows]
        partition = leidenalg.find_partition(
            g, leidenalg.ModularityVertexPartition, weights=weights
        )
        communities = {i: [g.vs[vi]["name"] for vi in partition[i]] for i in range(len(partition))}

        # Quality metrics — computed once for the full partition
        modularity: float = partition.quality()
        # Silhouette score requires ≥2 communities (optional deps: sklearn, numpy)
        silhouette: float | None = _compute_silhouette(names, name_idx, communities, rows, weights)

        topics_created = 0
        for comm_id, members in communities.items():
            if len(members) < 2:
                continue  # skip singletons

            # Step 3: gather evidence chunks
            evidence_rows = await self._arun(
                "MATCH (sess:SESSION {id: $sid})-[:IN_SESSION]->(f:FILE)"
                "-[:HAS_CHUNK]->(c:CHUNK)-[:HAS_ENTITY]->(e:ENTITY) "
                "WHERE e.name IN $members "
                "RETURN DISTINCT c.text AS text, c.heading_path AS hp "
                "LIMIT 5",
                sid=session_id,
                members=members,
            )
            passages = "\n\n".join(
                (f"[{r['hp']}] " if r["hp"] else "") + (r["text"] or "") for r in evidence_rows
            )

            # Step 4: LLM summarization
            title = ", ".join(members[:5])
            summary = ""
            if openrouter_key:
                try:
                    import httpx

                    payload = {
                        "model": llm_model,
                        "messages": [
                            {
                                "role": "system",
                                "content": (
                                    "You are a knowledge graph analyst. Given a set of entities "
                                    "and text passages, produce a short topic title and a 2-3 "
                                    "sentence summary."
                                ),
                            },
                            {
                                "role": "user",
                                "content": (
                                    f"Entities: {', '.join(members)}\n\n"
                                    f"Passages:\n{passages}\n\n"
                                    'Output JSON: {"title": "...", "summary": "..."}'
                                ),
                            },
                        ],
                        "response_format": {"type": "json_object"},
                    }
                    async with httpx.AsyncClient(timeout=30) as client:
                        resp = await client.post(
                            "https://openrouter.ai/api/v1/chat/completions",
                            headers={"Authorization": f"Bearer {openrouter_key}"},
                            json=payload,
                        )
                    resp.raise_for_status()
                    content = resp.json()["choices"][0]["message"]["content"]
                    parsed = json.loads(content)
                    title = parsed.get("title", title)
                    summary = parsed.get("summary", "")
                except Exception as exc:
                    logger.warning(
                        "LLM topic summarization failed for community %d: %s", comm_id, exc
                    )

            # Step 5: upsert TOPIC to Neo4j
            topic_id = str(_uuid.uuid5(_GK_NS, f"topic:{session_id}:{comm_id}"))
            topic = {
                "id": topic_id,
                "topic_id": topic_id,
                "label": title,
                "summary": summary,
                "community_id": comm_id,
                "members": members,
                "modularity": modularity,
                "silhouette": silhouette,
            }
            await self.upsert_topics([topic])

            # IN_TOPIC edges (new schema label)
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

            # Write pgvector ltm_topics if store provided
            if pg_store is not None:
                embedding_text = (summary or "").strip() or (title or "").strip()
                if embedding_text:
                    try:
                        from graphknows.retrieval.stores.embedder import embed_one

                        topic["embedding"] = await asyncio.to_thread(embed_one, embedding_text)
                    except Exception as exc:
                        logger.warning(
                            "Topic embedding failed for community %d: %s", comm_id, exc
                        )
                await pg_store.write_ltm_topic(topic)

            topics_created += 1

        return topics_created

    async def run_gds_enrichment(self, session_id: str) -> dict[str, Any]:
        """Run full GDS feature enrichment via LTMEnrichmentService.

        Args:
            session_id: Used to scope projection (session entities only).

        Returns:
            Dict with entity_count, edge_count, community_count, modularity, etc.

        Raises:
            Exception: Propagates any GDS or Neo4j failure.
        """
        from graphknows.memory.ltm.enrichment import LTMEnrichmentService

        svc = LTMEnrichmentService()
        result = await asyncio.to_thread(svc.run_gds, self._driver)
        return result.model_dump()

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

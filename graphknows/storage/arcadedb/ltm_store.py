"""ArcadeDB LTM graph store.

All graph queries use ArcadeDB openCypher. Dynamic relation edges require
declaring the edge type before MERGE (ArcadeDB constraint).

Community detection runs as a Python-native Leiden fallback inside
write_topic_summaries.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import uuid
from datetime import UTC, datetime
from typing import Any

from graphknows.storage.models.gds import CommunityResult
from graphknows.storage.arcadedb._labels import _safe_rel_label
from graphknows.storage.arcadedb.client import ArcadeDBClient

logger = logging.getLogger(__name__)

_LTM_DB = "ltm"
_GK_NS = uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")
UTC = UTC

# Community detection via ArcadeDB node2vec embeddings.
_N2V_DIM = 64
_N2V_KNN = 4
_N2V_MIN_SIM = 0.5
# Graph Analytical View (OLAP) that backs CALL algo.* — without a covering view
# ArcadeDB runs graph algorithms through the OLTP path, orders of magnitude
# slower (node2vec blows past the client read timeout and looks hung).
# Requires ArcadeDB >=26.3.2. UPDATE MODE OFF keeps ingest commits cheap; we
# REBUILD it on demand right before node2vec runs at promotion time.
_OLAP_VIEW = "ltm_olap"
# Structural/provenance edge types excluded when reading the entity-relation graph
# (we want semantic relations, not graph plumbing).
_NON_COMMUNITY_EDGE_TYPES = (
    "['MENTIONS','HAS_CHUNK','IN_SESSION','IN_TOPIC','IN_COMMUNITY','HAS_TURN',"
    "'HAS_ENTITY','CO_OCCURS','APPEARED_IN','MENTIONS_DATE','NEXT','CONTAINS_DOC']"
)

# ---------------------------------------------------------------------------
# DDL
# ---------------------------------------------------------------------------

_SCHEMA_DDL: list[tuple[str, str]] = [
    # Vertex types
    ("sql", "CREATE VERTEX TYPE ENTITY"),
    ("sql", "CREATE PROPERTY ENTITY.id IF NOT EXISTS STRING"),
    ("sql", "CREATE PROPERTY ENTITY.name IF NOT EXISTS STRING"),
    ("sql", "CREATE PROPERTY ENTITY.label IF NOT EXISTS STRING"),
    ("sql", "CREATE PROPERTY ENTITY.type IF NOT EXISTS STRING"),
    ("sql", "CREATE PROPERTY ENTITY.confidence IF NOT EXISTS DOUBLE"),
    ("sql", "CREATE PROPERTY ENTITY.community_id IF NOT EXISTS INTEGER"),
    ("sql", "CREATE PROPERTY ENTITY.pagerank IF NOT EXISTS DOUBLE"),
    ("sql", "CREATE PROPERTY ENTITY.betweenness IF NOT EXISTS DOUBLE"),
    ("sql", "CREATE PROPERTY ENTITY.degree_centrality IF NOT EXISTS DOUBLE"),
    ("sql", "CREATE PROPERTY ENTITY.ingested_at IF NOT EXISTS STRING"),
    ("sql", "CREATE PROPERTY ENTITY.contributing_session_id IF NOT EXISTS STRING"),
    ("sql", "CREATE PROPERTY ENTITY.merged_session_count IF NOT EXISTS INTEGER"),
    ("sql", "CREATE INDEX ON ENTITY(name) UNIQUE"),
    ("sql", "CREATE INDEX ON ENTITY(id) NOTUNIQUE"),
    ("sql", "CREATE VERTEX TYPE CHUNK"),
    ("sql", "CREATE PROPERTY CHUNK.id IF NOT EXISTS STRING"),
    ("sql", "CREATE PROPERTY CHUNK.session_id IF NOT EXISTS STRING"),
    ("sql", "CREATE PROPERTY CHUNK.text IF NOT EXISTS STRING"),
    ("sql", "CREATE PROPERTY CHUNK.heading_path IF NOT EXISTS STRING"),
    ("sql", "CREATE PROPERTY CHUNK.chunk_index IF NOT EXISTS INTEGER"),
    ("sql", "CREATE PROPERTY CHUNK.file_id IF NOT EXISTS STRING"),
    ("sql", "CREATE INDEX ON CHUNK(id) UNIQUE"),
    ("sql", "CREATE INDEX ON CHUNK(session_id) NOTUNIQUE"),
    ("sql", "CREATE VERTEX TYPE TOPIC"),
    ("sql", "CREATE PROPERTY TOPIC.id IF NOT EXISTS STRING"),
    ("sql", "CREATE PROPERTY TOPIC.topic_id IF NOT EXISTS STRING"),
    ("sql", "CREATE PROPERTY TOPIC.label IF NOT EXISTS STRING"),
    ("sql", "CREATE PROPERTY TOPIC.summary IF NOT EXISTS STRING"),
    ("sql", "CREATE PROPERTY TOPIC.community_id IF NOT EXISTS INTEGER"),
    ("sql", "CREATE PROPERTY TOPIC.community_hash IF NOT EXISTS STRING"),
    ("sql", "CREATE PROPERTY TOPIC.entity_count IF NOT EXISTS INTEGER"),
    ("sql", "CREATE PROPERTY TOPIC.entity_names IF NOT EXISTS LIST"),
    ("sql", "CREATE PROPERTY TOPIC.session_ids IF NOT EXISTS LIST"),
    ("sql", "CREATE PROPERTY TOPIC.modularity IF NOT EXISTS DOUBLE"),
    ("sql", "CREATE PROPERTY TOPIC.gds_modularity IF NOT EXISTS DOUBLE"),
    ("sql", "CREATE PROPERTY TOPIC.community_algorithm IF NOT EXISTS STRING"),
    ("sql", "CREATE PROPERTY TOPIC.silhouette IF NOT EXISTS DOUBLE"),
    ("sql", "CREATE INDEX ON TOPIC(id) UNIQUE"),
    ("sql", "CREATE INDEX ON TOPIC(community_id) NOTUNIQUE"),
    ("sql", "CREATE VERTEX TYPE FILE"),
    ("sql", "CREATE PROPERTY FILE.id IF NOT EXISTS STRING"),
    ("sql", "CREATE PROPERTY FILE.title IF NOT EXISTS STRING"),
    ("sql", "CREATE PROPERTY FILE.parser_name IF NOT EXISTS STRING"),
    ("sql", "CREATE PROPERTY FILE.source_hint IF NOT EXISTS STRING"),
    ("sql", "CREATE INDEX ON FILE(id) UNIQUE"),
    ("sql", "CREATE VERTEX TYPE DIALOG"),
    ("sql", "CREATE PROPERTY DIALOG.id IF NOT EXISTS STRING"),
    ("sql", "CREATE PROPERTY DIALOG.session_id IF NOT EXISTS STRING"),
    ("sql", "CREATE PROPERTY DIALOG.started_at IF NOT EXISTS STRING"),
    ("sql", "CREATE INDEX ON DIALOG(id) UNIQUE"),
    ("sql", "CREATE VERTEX TYPE SESSION"),
    ("sql", "CREATE PROPERTY SESSION.id IF NOT EXISTS STRING"),
    ("sql", "CREATE PROPERTY SESSION.flushed_at IF NOT EXISTS STRING"),
    ("sql", "CREATE INDEX ON SESSION(id) UNIQUE"),
    ("sql", "CREATE VERTEX TYPE SEGMENT"),
    ("sql", "CREATE PROPERTY SEGMENT.segment_id IF NOT EXISTS STRING"),
    ("sql", "CREATE PROPERTY SEGMENT.ingested_at IF NOT EXISTS STRING"),
    ("sql", "CREATE PROPERTY SEGMENT.run_id IF NOT EXISTS STRING"),
    ("sql", "CREATE PROPERTY SEGMENT.git_revision IF NOT EXISTS STRING"),
    ("sql", "CREATE PROPERTY SEGMENT.config_hash IF NOT EXISTS STRING"),
    ("sql", "CREATE PROPERTY SEGMENT.pipeline_version IF NOT EXISTS STRING"),
    ("sql", "CREATE PROPERTY SEGMENT.source_count IF NOT EXISTS INTEGER"),
    ("sql", "CREATE PROPERTY SEGMENT.input_dir IF NOT EXISTS STRING"),
    ("sql", "CREATE PROPERTY SEGMENT.taxonomy_name IF NOT EXISTS STRING"),
    ("sql", "CREATE INDEX ON SEGMENT(segment_id) UNIQUE"),
    ("sql", "CREATE VERTEX TYPE DOC"),
    ("sql", "CREATE PROPERTY DOC.id IF NOT EXISTS STRING"),
    ("sql", "CREATE INDEX ON DOC(id) UNIQUE"),
    ("sql", "CREATE VERTEX TYPE SESSIONENTITY"),
    ("sql", "CREATE PROPERTY SESSIONENTITY.id IF NOT EXISTS STRING"),
    ("sql", "CREATE PROPERTY SESSIONENTITY.session_id IF NOT EXISTS STRING"),
    ("sql", "CREATE PROPERTY SESSIONENTITY.name IF NOT EXISTS STRING"),
    ("sql", "CREATE INDEX ON SESSIONENTITY(session_id) NOTUNIQUE"),
    # Canonical temporal nodes — one global node per resolved ISO date.
    ("sql", "CREATE VERTEX TYPE TEMPORAL"),
    ("sql", "CREATE PROPERTY TEMPORAL.iso IF NOT EXISTS STRING"),
    ("sql", "CREATE PROPERTY TEMPORAL.granularity IF NOT EXISTS STRING"),
    ("sql", "CREATE INDEX ON TEMPORAL(iso) UNIQUE"),
    # Semantic-frame instances (deterministic FrameNet layer). One node per
    # frame evoked in a chunk; fillers/speaker carry the frame-element roles.
    ("sql", "CREATE VERTEX TYPE FRAME"),
    ("sql", "CREATE PROPERTY FRAME.id IF NOT EXISTS STRING"),
    ("sql", "CREATE PROPERTY FRAME.session_id IF NOT EXISTS STRING"),
    ("sql", "CREATE PROPERTY FRAME.frame_type IF NOT EXISTS STRING"),
    ("sql", "CREATE PROPERTY FRAME.chunk_id IF NOT EXISTS STRING"),
    ("sql", "CREATE PROPERTY FRAME.lu IF NOT EXISTS STRING"),
    ("sql", "CREATE PROPERTY FRAME.speaker IF NOT EXISTS STRING"),
    ("sql", "CREATE PROPERTY FRAME.fes IF NOT EXISTS STRING"),
    ("sql", "CREATE PROPERTY FRAME.fillers IF NOT EXISTS LIST"),
    ("sql", "CREATE INDEX ON FRAME(id) UNIQUE"),
    ("sql", "CREATE INDEX ON FRAME(session_id) NOTUNIQUE"),
    ("sql", "CREATE INDEX ON FRAME(frame_type) NOTUNIQUE"),
    # Edge types
    ("sql", "CREATE EDGE TYPE MENTIONS"),
    ("sql", "CREATE EDGE TYPE MENTIONS_DATE"),
    ("sql", "CREATE EDGE TYPE HAS_CHUNK"),
    ("sql", "CREATE EDGE TYPE IN_SESSION"),
    ("sql", "CREATE EDGE TYPE HAS_TURN"),
    ("sql", "CREATE EDGE TYPE NEXT"),
    ("sql", "CREATE EDGE TYPE IN_TOPIC"),
    ("sql", "CREATE EDGE TYPE IN_COMMUNITY"),
    ("sql", "CREATE EDGE TYPE CO_OCCURS"),
    ("sql", "CREATE EDGE TYPE APPEARED_IN"),
    ("sql", "CREATE EDGE TYPE CONTAINS_DOC"),
    ("sql", "CREATE EDGE TYPE HAS_ENTITY"),
]


# Single-valued predicates: when a newer edge for the same (source, predicate)
# arrives pointing to a different object, the older edge is tombstoned by
# setting invalid_at instead of deleted.  Only include predicates that are
# genuinely single-valued in the world (a person has one current employer, one
# current city, etc.).
_LTM_SINGLE_VALUED_RELATIONS: frozenset[str] = frozenset(
    {
        "WORKS_AS",
        "WORKS_AT",
        "EMPLOYED_BY",
        "LIVES_IN",
        "LOCATED_IN",
        "BASED_IN",
        "MARRIED_TO",
        "AFFILIATED_WITH",
        "BELONGS_TO",
        "LED_BY",
    }
)


# Pruning: tombstoned (is_current=false) relation edges older than this many
# days are eligible for removal.  Conservative default — only stale superseded
# edges are touched; still-current edges are never pruned.
_TOMBSTONE_RETENTION_DAYS: int = 90


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------


class ArcadeDBLTMStore:
    """ArcadeDB-backed LTM graph store.

    Attributes:
        _c: ArcadeDBClient.
        _relation_extractor: Optional relation extractor for LLM-assisted mode.
    """

    def __init__(
        self,
        client: ArcadeDBClient,
        relation_extractor: Any | None = None,
        db: str = _LTM_DB,
    ) -> None:
        self._c = client
        self._db = db
        self._relation_extractor = relation_extractor
        self._declared_edge_types: set[str] = set()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @property
    def client(self) -> ArcadeDBClient:
        """The underlying ArcadeDB client (shared across stores)."""
        return self._c

    @property
    def database(self) -> str:
        """Name of the ArcadeDB database this store reads/writes."""
        return self._db

    async def connect(self) -> None:
        """Open the shared client connection. Idempotent.

        Schema creation is deliberately not run here — call ensure_schema()
        explicitly on write paths so read paths avoid idempotent DDL round-trips.
        """
        await self._c.connect()

    async def close(self) -> None:
        """Close the shared client connection. Idempotent."""
        await self._c.close()

    async def ensure_indexes(self) -> None:
        """Create LTM database and schema. Idempotent."""
        await self._c.create_database(self._db)
        for lang, stmt in _SCHEMA_DDL:
            try:
                await self._c.command(self._db, stmt, language=lang)
            except Exception as exc:
                exc_s = str(exc).lower()
                if "already exists" in exc_s or "duplicate" in exc_s:
                    pass  # idempotent re-creation — expected
                else:
                    logger.warning("LTM DDL failed (non-idempotent): %s | %s", stmt[:60], exc)
                    raise
        logger.info("ArcadeDBLTMStore schema ensured")

    # keep legacy name used by container/builder
    ensure_schema = ensure_indexes

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _cmd(self, cypher: str, **params: Any) -> list[dict]:
        return await self._c.command(self._db, cypher, params=params or None)

    async def _qry(self, cypher: str, **params: Any) -> list[dict]:
        return await self._c.query(self._db, cypher, params=params or None)

    async def _ensure_edge_type(self, edge_type: str) -> None:
        """Declare edge type before first MERGE (DDL idempotent)."""
        if edge_type in self._declared_edge_types:
            return
        safe = _safe_rel_label(edge_type)
        if not safe:
            return
        try:
            await self._c.command(self._db, f"CREATE EDGE TYPE {safe} IF NOT EXISTS", language="sql")
            self._declared_edge_types.add(edge_type)
        except Exception as exc:
            exc_s = str(exc).lower()
            if "already exists" in exc_s or "duplicate" in exc_s:
                self._declared_edge_types.add(edge_type)
            else:
                logger.warning("ensure_edge_type %s failed: %s", safe, exc)
                raise

    async def _supersede_ltm_edge(self, safe_rel: str, src: str, tgt: str, now: str) -> None:
        """Tombstone any current LTM relation edge for (src, predicate) that points
        to a different target.

        Called only for single-valued predicates (defined in _LTM_SINGLE_VALUED_RELATIONS).
        Sets ``invalid_at`` to *now* and ``is_current`` to false on the superseded edge
        instead of deleting it, preserving the historical record.
        """
        try:
            await self._cmd(
                f"MATCH (a:ENTITY {{name: $src}})"
                f"-[r:{safe_rel} {{is_current: true}}]"
                f"->(b:ENTITY) "
                f"WHERE toLower(b.name) <> toLower($tgt) "
                f"SET r.invalid_at = $now, r.is_current = false",
                src=src,
                tgt=tgt,
                now=now,
            )
        except Exception as exc:
            logger.debug("_supersede_ltm_edge %s (%s → ≠%s) skipped: %s", safe_rel, src, tgt, exc)

    # ------------------------------------------------------------------
    # Session subgraph write (main flush path)
    # ------------------------------------------------------------------

    async def write_session_subgraph(
        self, session_id: str, stm_data: dict[str, Any]
    ) -> dict[str, int]:
        """MERGE full STM subgraph into LTM. Returns {nodes, edges}.

        Writes are auto-committed per element. A single enclosing transaction
        is NOT used: the flush runs in a worker-thread event loop
        (``runtime.flush_memory`` → ``run_in_executor``) while the ArcadeDB
        client is bound to the server's main loop, so a begin/commit pair would
        land on different connections and ArcadeDB would discard the
        transaction's writes silently. Relation edge types are pre-declared via
        ``_ensure_edge_type`` before first use.
        """
        now = datetime.now(UTC).isoformat()
        nodes = edges = 0
        entity_by_name: dict[str, dict[str, Any]] = {
            e["name"]: e for e in stm_data.get("entities", [])
        }

        # Pre-declare relation edge types (DDL auto-commits).
        for r in stm_data.get("relation_edges", []):
            rel = r.get("relation")
            if rel:
                safe = _safe_rel_label(rel)
                if safe:
                    await self._ensure_edge_type(safe)

        # SESSION node
        await self._cmd(
            "MERGE (s:SESSION {id: $sid}) SET s.flushed_at = $now", sid=session_id, now=now
        )
        nodes += 1

        # FILE nodes
        for f in stm_data.get("files", []):
            await self._cmd(
                "MERGE (f:FILE {id: $fid}) SET f.title = $title, f.parser_name = $pn, f.source_hint = $sh "
                "WITH f MATCH (sess:SESSION {id: $sid}) MERGE (sess)-[:IN_SESSION]->(f)",
                fid=f["id"],
                title=f.get("title", ""),
                pn=f.get("parser_name", ""),
                sh=f.get("source_hint", ""),
                sid=session_id,
            )
            nodes += 1
            edges += 1

        # CHUNK nodes — session_id stored directly so text search can filter by session
        for c in stm_data.get("chunks", []):
            await self._cmd(
                "MERGE (c:CHUNK {id: $cid}) SET c.text = $text, c.heading_path = $hp, "
                "c.chunk_index = $ci, c.session_id = $sid, c.file_id = $fid "
                "WITH c MATCH (f:FILE {id: $fid}) MERGE (f)-[:HAS_CHUNK]->(c)",
                cid=c["id"],
                text=c.get("text", ""),
                hp=c.get("heading_path", ""),
                ci=c.get("chunk_index", 0),
                fid=c.get("file_id", ""),
                sid=session_id,
            )
            nodes += 1
            edges += 1

            # Canonical TEMPORAL nodes — global per ISO date, linked to the chunk.
            for tref in (c.get("metadata") or {}).get("temporal") or []:
                iso = str(tref.get("iso", "")).strip()
                if not iso:
                    continue
                await self._cmd(
                    "MERGE (d:TEMPORAL {iso: $iso}) SET d.granularity = $g "
                    "WITH d MATCH (c:CHUNK {id: $cid}) MERGE (c)-[:MENTIONS_DATE]->(d)",
                    iso=iso,
                    g=str(tref.get("granularity", "")),
                    cid=c["id"],
                )
                nodes += 1
                edges += 1

        # ENTITY nodes + MENTIONS edges
        written: set[str] = set()
        for ce in stm_data.get("chunk_entity_edges", []):
            ename = ce["entity_name"]
            em = entity_by_name.get(ename, {"type": "UNKNOWN", "confidence": 1.0})
            if ename not in written:
                await self._cmd(
                    "MERGE (e:ENTITY {name: $n}) SET e.type = $t, e.confidence = $c",
                    n=ename,
                    t=em.get("type", "UNKNOWN"),
                    c=em.get("confidence", 1.0),
                )
                nodes += 1
                written.add(ename)
            await self._cmd(
                "MATCH (c:CHUNK {id: $cid}), (e:ENTITY {name: $n}) MERGE (c)-[:MENTIONS]->(e)",
                cid=ce["chunk_id"],
                n=ename,
            )
            edges += 1

        # DIALOG nodes
        for d in stm_data.get("dialogs", []):
            await self._cmd(
                "MERGE (d:DIALOG {id: $did}) SET d.session_id = $sid, d.started_at = $sa "
                "WITH d MATCH (sess:SESSION {id: $sid}) MERGE (sess)-[:IN_SESSION]->(d)",
                did=d["id"],
                sid=session_id,
                sa=d.get("started_at", ""),
            )
            nodes += 1
            edges += 1

        # TURN nodes + HAS_TURN edges
        for t in stm_data.get("turns", []):
            await self._cmd(
                "MERGE (t:TURN {id: $tid}) SET t.session_id = $sid, t.turn_index = $ti, "
                "t.role = $r, t.text = $txt "
                "WITH t MATCH (d:DIALOG {id: $did}) MERGE (d)-[:HAS_TURN]->(t)",
                tid=t["id"],
                sid=session_id,
                ti=t.get("turn_index", 0),
                r=t.get("role", "unknown"),
                txt=t.get("text", ""),
                did=t.get("dialog_id", ""),
            )
            nodes += 1
            edges += 1

        # NEXT edges (turn sequence)
        for ne in stm_data.get("next_edges", []):
            await self._cmd(
                "MATCH (t1:TURN {id: $a}), (t2:TURN {id: $b}) MERGE (t1)-[:NEXT]->(t2)",
                a=ne["from_turn_id"],
                b=ne["to_turn_id"],
            )
            edges += 1

        # Turn-entity edges
        for te in stm_data.get("turn_entity_edges", []):
            ename = te["entity_name"]
            em = entity_by_name.get(ename, {"type": "UNKNOWN", "confidence": 1.0})
            if ename not in written:
                await self._cmd(
                    "MERGE (e:ENTITY {name: $n}) SET e.type = $t, e.confidence = $c",
                    n=ename,
                    t=em.get("type", "UNKNOWN"),
                    c=em.get("confidence", 1.0),
                )
                nodes += 1
                written.add(ename)
            await self._cmd(
                "MATCH (t:TURN {id: $tid}), (e:ENTITY {name: $n}) MERGE (t)-[:HAS_ENTITY]->(e)",
                tid=te["turn_id"],
                n=ename,
            )
            edges += 1

        # Relation edges from STM (bi-temporal)
        # Each edge carries valid_at (world time), invalid_at (tombstone), and
        # created_at (transaction/ingestion time).  valid_at comes from the STM
        # edge (which in turn is populated from turn timestamps during STM ingest).
        # For single-valued predicates a supersedence pass tombstones any existing
        # current edge pointing to a different object before upserting the new one.
        for r in stm_data.get("relation_edges", []):
            src, rel, tgt = r.get("source"), r.get("relation"), r.get("target")
            if not (src and rel and tgt):
                continue
            safe = _safe_rel_label(rel)
            if not safe:
                continue

            # Temporal metadata forwarded from STM edge (may be absent for legacy data).
            edge_valid_at: str = str(r.get("valid_at") or r.get("valid_from") or "").strip()
            edge_created_at: str = now

            # Supersedence: tombstone older current edges for single-valued predicates.
            if rel.upper() in _LTM_SINGLE_VALUED_RELATIONS:
                await self._supersede_ltm_edge(safe, src, tgt, now)

            await self._cmd(
                f"MERGE (a:ENTITY {{name: $src}}) "
                f"MERGE (b:ENTITY {{name: $tgt}}) "
                f"MERGE (a)-[r:{safe}]->(b) "
                f"SET r.weight = coalesce(r.weight, 0) + 1, "
                f"    r.session_id = $sid, "
                f"    r.valid_at = coalesce(r.valid_at, $valid_at), "
                f"    r.invalid_at = null, "
                f"    r.is_current = true, "
                f"    r.created_at = coalesce(r.created_at, $created_at)",
                src=src,
                tgt=tgt,
                sid=session_id,
                valid_at=edge_valid_at or None,
                created_at=edge_created_at,
            )
            edges += 1

        return {"nodes": nodes, "edges": edges}

    # ------------------------------------------------------------------
    # PageRank / centrality persistence
    # ------------------------------------------------------------------

    async def persist_centrality(
        self,
        centrality: dict[str, dict[str, float]],
    ) -> int:
        """Write PageRank and optional degree/betweenness onto ENTITY nodes.

        Args:
            centrality: Mapping of entity_name → {pagerank, degree_centrality,
                        betweenness}.  Unknown keys are ignored.  All values
                        are stored as DOUBLE on the ENTITY node so that WS-2
                        retrieval can read ``e.pagerank`` without special-casing
                        absent nodes (the property simply won't exist until
                        this method is called at least once).

        Returns:
            Number of ENTITY nodes updated.
        """
        if not centrality:
            return 0
        updated = 0
        for name, scores in centrality.items():
            if not name:
                continue
            pagerank = scores.get("pagerank")
            degree = scores.get("degree_centrality")
            betweenness = scores.get("betweenness")
            # Build a SET clause only for present scores to avoid overwriting
            # previously computed values with None.
            parts: list[str] = []
            params: dict[str, Any] = {"name": name}
            if pagerank is not None:
                parts.append("e.pagerank = $pr")
                params["pr"] = float(pagerank)
            if degree is not None:
                parts.append("e.degree_centrality = $deg")
                params["deg"] = float(degree)
            if betweenness is not None:
                parts.append("e.betweenness = $bet")
                params["bet"] = float(betweenness)
            if not parts:
                continue
            set_clause = ", ".join(parts)
            try:
                await self._cmd(
                    f"MATCH (e:ENTITY {{name: $name}}) SET {set_clause}",
                    **params,
                )
                updated += 1
            except Exception as exc:
                logger.warning("persist_centrality: entity %r failed: %s", name, exc)

        logger.info("persist_centrality: updated %d ENTITY nodes", updated)
        return updated

    async def compute_and_persist_pagerank(self) -> int:
        """Compute a lightweight in-process PageRank from the entity-relation graph
        and persist it on ENTITY nodes.

        This runs after the entity graph is fully written but before GDS/topics.
        It uses the same entity graph as community detection (relation edges +
        co-occurrence).  Falls back gracefully when numpy is unavailable.

        Returns:
            Number of ENTITY nodes updated with pagerank scores.
        """
        try:
            import numpy as np
        except ImportError:
            logger.info("compute_and_persist_pagerank: numpy unavailable, skipping")
            return 0

        # Fetch all entity names and directed relation edges.
        try:
            entity_rows = await self._qry(
                "MATCH (e:ENTITY) WHERE e.name IS NOT NULL RETURN e.name AS name"
            )
            edge_rows = await self._qry(
                "MATCH (a:ENTITY)-[r]->(b:ENTITY) WHERE r.is_current = true OR r.is_current IS NULL "
                f"AND NOT type(r) IN {_NON_COMMUNITY_EDGE_TYPES} "
                "RETURN DISTINCT a.name AS src, b.name AS tgt, "
                "coalesce(r.weight, 1.0) AS w"
            )
        except Exception as exc:
            logger.warning("compute_and_persist_pagerank: graph query failed: %s", exc)
            return 0

        names = [r["name"] for r in entity_rows if r.get("name")]
        if not names:
            return 0
        name_idx = {n: i for i, n in enumerate(names)}
        n = len(names)

        # Build adjacency matrix.
        adj = np.zeros((n, n), dtype=float)
        for row in edge_rows:
            src, tgt = row.get("src"), row.get("tgt")
            w = float(row.get("w") or 1.0)
            si, ti = name_idx.get(src), name_idx.get(tgt)  # type: ignore[arg-type]
            if si is not None and ti is not None and si != ti:
                adj[si][ti] += w

        # Column-normalise (dangling nodes get uniform 1/n).
        col_sum = adj.sum(axis=0)
        dangling = col_sum == 0
        col_sum[dangling] = 1.0
        adj = adj / col_sum

        # Power iteration (damping=0.85, max 50 iters).
        damping = 0.85
        rank = np.full(n, 1.0 / n)
        dangling_nodes = np.where(adj.sum(axis=0) == 0)[0]
        for _ in range(50):
            dangling_sum = rank[dangling_nodes].sum() / n if len(dangling_nodes) else 0.0
            new_rank = (1 - damping) / n + damping * (adj @ rank + dangling_sum)
            if np.linalg.norm(new_rank - rank, 1) < 1e-6:
                break
            rank = new_rank
        rank = new_rank  # type: ignore[possibly-undefined]

        # Degree centrality (out-degree / (n-1)).
        out_degree = (adj > 0).sum(axis=1).astype(float)
        degree_centrality = out_degree / max(n - 1, 1)

        centrality = {
            names[i]: {
                "pagerank": float(rank[i]),
                "degree_centrality": float(degree_centrality[i]),
            }
            for i in range(n)
        }
        return await self.persist_centrality(centrality)

    # ------------------------------------------------------------------
    # Graph pruning / cleaning
    # ------------------------------------------------------------------

    async def prune_tombstoned_edges(
        self,
        retention_days: int = _TOMBSTONE_RETENTION_DAYS,
    ) -> int:
        """Remove relation edges that have been tombstoned beyond the retention window.

        Only edges where ALL of the following are true are deleted:
        - ``is_current = false``  (tombstoned by WS-5 supersedence)
        - ``invalid_at`` is set and older than ``now - retention_days``

        Still-current edges (``is_current = true`` or ``is_current IS NULL``) are
        NEVER touched.  This is intentionally conservative: the window is long
        (default 90 days) and the filter is strict.

        Returns:
            Number of edges deleted.
        """
        from datetime import timedelta

        cutoff = (datetime.now(UTC) - timedelta(days=retention_days)).isoformat()
        deleted = 0
        try:
            rows = await self._cmd(
                "MATCH (a:ENTITY)-[r]->(b:ENTITY) "
                "WHERE r.is_current = false "
                "AND r.invalid_at IS NOT NULL "
                "AND r.invalid_at < $cutoff "
                "DELETE r "
                "RETURN count(r) AS deleted",
                cutoff=cutoff,
            )
            deleted = int((rows[0].get("deleted") or 0) if rows else 0)
        except Exception as exc:
            logger.warning("prune_tombstoned_edges failed (non-fatal): %s", exc)
            return 0

        logger.info(
            "prune_tombstoned_edges: deleted %d tombstoned edges older than %d days (cutoff=%s)",
            deleted,
            retention_days,
            cutoff,
        )
        return deleted

    async def prune_duplicate_edges(self) -> int:
        """Remove low-confidence duplicate non-current relation edges.

        Targets structural duplicates: parallel edges of the same type between
        the same pair of ENTITY nodes where ``is_current = false`` and
        ``confidence < 0.3`` (or weight < 0.5).  These arise when the same
        fact was extracted multiple times before supersedence — once tombstoned
        they add no value.

        Returns:
            Number of edges deleted.
        """
        deleted = 0
        try:
            rows = await self._cmd(
                "MATCH (a:ENTITY)-[r]->(b:ENTITY) "
                "WHERE r.is_current = false "
                "AND (r.weight IS NOT NULL AND r.weight < 0.5 "
                "     OR r.confidence IS NOT NULL AND r.confidence < 0.3) "
                "DELETE r "
                "RETURN count(r) AS deleted",
            )
            deleted = int((rows[0].get("deleted") or 0) if rows else 0)
        except Exception as exc:
            logger.warning("prune_duplicate_edges failed (non-fatal): %s", exc)
            return 0

        logger.info("prune_duplicate_edges: deleted %d low-confidence tombstoned edges", deleted)
        return deleted

    # ------------------------------------------------------------------
    # Relation extraction (LLM-assisted mode)
    # ------------------------------------------------------------------

    async def extract_and_write_relations(
        self,
        stm_data: dict[str, Any],
        llm_api_key: str = "",
        llm_model: str = "",
        session_id: str = "",
    ) -> int:
        if self._relation_extractor is None:
            raise RuntimeError("ArcadeDBLTMStore requires a relation_extractor for LLM mode")
        entity_by_name: dict[str, dict[str, Any]] = {
            e["name"]: e for e in stm_data.get("entities", [])
        }
        canonical: dict[str, str] = {k.upper().replace("_", " "): k for k in entity_by_name}
        all_triplets: list[tuple[str, str, str, dict]] = []
        chunks = stm_data.get("chunks", [])
        # Cap LLM calls to avoid MCP timeout (120 s). Pick the longest chunks
        # (most content) so we maximise coverage within the budget.
        _MAX_CHUNKS = 10
        if len(chunks) > _MAX_CHUNKS:
            chunks = sorted(chunks, key=lambda c: len(c.get("text", "")), reverse=True)[
                :_MAX_CHUNKS
            ]
        for chunk in chunks:
            relations, nodes_data, _ = await self._relation_extractor.extract_relations(
                text=chunk["text"]
            )
            for node in nodes_data:
                nid = str(node.get("id") or "").strip()
                if nid and nid.upper().replace("_", " ") not in {
                    k.upper().replace("_", " ") for k in entity_by_name
                }:
                    entity_by_name[nid] = {"name": nid, "type": node.get("type", "ENTITY")}
            for item in relations:
                if len(item) >= 3:
                    all_triplets.append(
                        (item[0], item[1], item[2], item[3] if len(item) > 3 else {})
                    )

        now_ts = datetime.now(UTC).isoformat()
        edges = 0
        for src, rel, tgt, _meta in all_triplets:
            safe = _safe_rel_label(rel)
            if not safe:
                continue
            src_canon = canonical.get(src.upper().replace("_", " "), src)
            tgt_canon = canonical.get(tgt.upper().replace("_", " "), tgt)
            await self._ensure_edge_type(safe)
            # Supersede single-valued predicates before upserting.
            if rel.upper() in _LTM_SINGLE_VALUED_RELATIONS:
                await self._supersede_ltm_edge(safe, src_canon, tgt_canon, now_ts)
            await self._cmd(
                f"MERGE (a:ENTITY {{name: $src}}) "
                f"SET a.type = coalesce(a.type, $src_t) "
                f"MERGE (b:ENTITY {{name: $tgt}}) "
                f"SET b.type = coalesce(b.type, $tgt_t) "
                f"MERGE (a)-[r:{safe}]->(b) "
                f"SET r.weight = coalesce(r.weight, 0) + 1, "
                f"    r.session_id = $sid, "
                f"    r.valid_at = coalesce(r.valid_at, null), "
                f"    r.invalid_at = null, "
                f"    r.is_current = true, "
                f"    r.created_at = coalesce(r.created_at, $now)",
                src=src_canon,
                tgt=tgt_canon,
                sid=session_id,
                src_t=entity_by_name.get(src_canon, {}).get("type", "ENTITY"),
                tgt_t=entity_by_name.get(tgt_canon, {}).get("type", "ENTITY"),
                now=now_ts,
            )
            edges += 1
        return edges

    # ------------------------------------------------------------------
    # Topic communities + summarization
    # ------------------------------------------------------------------

    async def detect_communities(self) -> CommunityResult:
        """Detect communities and write ``community_id`` back to ENTITY nodes.

        Runs in ALL relation-extraction modes (llm-free and llm-assisted). Its
        sole job is to cluster the entity-relation graph and persist the cluster
        id on each entity, so retrieval can weight chunks by community
        (``_community_boost``). Topic summarization (LLM) is a separate,
        llm-assisted-only step layered on top of these communities.
        """
        communities, modularity, silhouette, algorithm = await self._topic_communities()
        return CommunityResult(
            community_count=sum(1 for members in communities.values() if len(members) >= 2),
            modularity=modularity or 0.0,
            algorithm_used=algorithm,
            silhouette_cosine=silhouette,
        )

    async def _topic_communities(
        self,
    ) -> tuple[dict[int, list[str]], float | None, float | None, str]:
        """Detect entity communities and return (communities, modularity, silhouette, algorithm).

        Primary path clusters the entity graph built from ArcadeDB ``node2vec``
        embeddings (which encode each entity's relations + co-occurrence) — this
        gives high, meaningful modularity. Falls back to chunk co-occurrence Leiden
        when node2vec is unavailable.
        """
        gds_rows = await self._qry(
            "MATCH (e:ENTITY) WHERE e.community_id IS NOT NULL "
            "RETURN e.community_id AS cid, collect(DISTINCT e.name) AS members"
        )
        gds_communities: dict[int, list[str]] = {}
        for r in gds_rows:
            cid = r.get("cid")
            members = r.get("members") or []
            if cid is not None:
                gds_communities[int(cid)] = sorted(n for n in members if n)
        if any(len(m) >= 2 for m in gds_communities.values()):
            modularity = await self._partition_modularity(gds_communities)
            return gds_communities, modularity, None, "gds_leiden"

        try:
            import igraph as ig
            import leidenalg
        except ImportError:
            logger.warning("igraph/leidenalg not available; skipping community detection")
            return {}, None, None, "none"

        graph = await self._entity_graph()
        if graph is None or not graph[1]:
            return {}, None, None, "none"
        names, edge_pairs, weights, algorithm = graph
        g = ig.Graph(n=len(names), directed=False)
        g.vs["name"] = names
        g.add_edges(edge_pairs)
        partition = leidenalg.find_partition(
            g, leidenalg.ModularityVertexPartition, weights=weights if weights else None, seed=42
        )
        communities: dict[int, list[str]] = {
            i: sorted(g.vs[vi]["name"] for vi in partition[i]) for i in range(len(partition))
        }
        await self._write_community_ids(communities)
        return communities, float(partition.modularity), None, algorithm

    async def _write_community_ids(self, communities: dict[int, list[str]]) -> None:
        for cid, members in communities.items():
            for name in members:
                try:
                    await self._cmd(
                        "MATCH (e:ENTITY {name: $name}) SET e.community_id = $cid",
                        name=name,
                        cid=cid,
                    )
                except Exception as exc:
                    logger.warning(
                        "_write_community_ids failed for %r (cid=%s): %s", name, cid, exc
                    )

    async def _ensure_analytical_view(self) -> None:
        """Create + refresh the Graph OLAP view so CALL algo.* uses the CSR engine.

        Without a covering Graph Analytical View, ArcadeDB runs graph algorithms
        (node2vec) through the OLTP traversal path — orders of magnitude slower,
        enough to exceed the client read timeout and look hung. Requires ArcadeDB
        >=26.3.2. A full-graph view (no VERTEX/EDGE TYPES clause) covers every
        type node2vec touches. Raises on failure so the caller falls back to the
        pure-Python co-occurrence graph.
        """
        try:
            await self._c.command(
                self._db,
                f"CREATE GRAPH ANALYTICAL VIEW {_OLAP_VIEW} UPDATE MODE OFF",
                language="sql",
            )
        except Exception as exc:
            if "already exists" not in str(exc).lower():
                raise
        # UPDATE MODE OFF marks the view stale on each commit; rebuild it against
        # current data before running the algorithm.
        await self._c.command(
            self._db, f"REBUILD GRAPH ANALYTICAL VIEW {_OLAP_VIEW}", language="sql"
        )

    async def _entity_graph(
        self,
    ) -> tuple[list[str], list[tuple[int, int]], list[float], str] | None:
        """Build the community-detection graph: node2vec kNN + typed relations.

        Returns (names, edge_pairs, weights, algorithm) or None. node2vec runs over
        the whole LTM graph so each entity embedding reflects both its relation
        edges and its chunk context; a kNN graph over those embeddings has real
        structure where the near-complete co-occurrence graph did not. The explicit
        typed relation edges are added on top. Falls back to co-occurrence when
        node2vec/numpy are unavailable.
        """
        try:
            import numpy as np
        except ImportError:
            return await self._cooccurrence_graph()
        try:
            await self._ensure_analytical_view()
            rows = await self._qry(
                f"CALL algo.node2vec({{walkLength:20,walks:20,dimensions:{_N2V_DIM}}}) "
                "YIELD node, embedding WHERE node.name IS NOT NULL "
                "RETURN node.name AS name, embedding"
            )
        except Exception as exc:
            logger.info("node2vec unavailable (%s); using co-occurrence", exc)
            return await self._cooccurrence_graph()
        rows = [r for r in rows if r.get("name") and r.get("embedding")]
        if len(rows) < 3:
            return await self._cooccurrence_graph()

        names = [r["name"] for r in rows]
        name_idx = {n: i for i, n in enumerate(names)}
        mat = np.asarray([r["embedding"] for r in rows], dtype=float)
        mat = mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-9)
        sim = mat @ mat.T

        edges: list[tuple[int, int]] = []
        weights: list[float] = []
        seen: set[tuple[int, int]] = set()

        def _add(i: int, j: int, w: float) -> None:
            if i == j:
                return
            key = (i, j) if i < j else (j, i)
            if key in seen:
                return
            seen.add(key)
            edges.append((i, j))
            weights.append(float(w))

        # kNN over node2vec embeddings.
        for i in range(len(names)):
            order = np.argsort(-sim[i])
            count = 0
            for j in order:
                if j == i:
                    continue
                if sim[i][j] < _N2V_MIN_SIM:
                    break
                _add(i, int(j), sim[i][j])
                count += 1
                if count >= _N2V_KNN:
                    break

        # Explicit typed relation edges (structural signal).
        rel_rows = await self._qry(
            "MATCH (a:ENTITY)-[r]->(b:ENTITY) WHERE a.name < b.name "
            f"AND NOT type(r) IN {_NON_COMMUNITY_EDGE_TYPES} "
            "RETURN DISTINCT a.name AS a, b.name AS b"
        )
        for r in rel_rows:
            ai, bi = name_idx.get(r["a"]), name_idx.get(r["b"])
            if ai is not None and bi is not None:
                _add(ai, bi, max(_N2V_MIN_SIM, float(sim[ai][bi])))

        return names, edges, weights, "node2vec"

    async def _cooccurrence_graph(
        self,
    ) -> tuple[list[str], list[tuple[int, int]], list[float], str] | None:
        """Fallback community graph from entity co-occurrence in chunks."""
        rel_rows = await self._qry(
            "MATCH (c:CHUNK)-[:MENTIONS]->(a:ENTITY), (c)-[:MENTIONS]->(b:ENTITY) "
            "WHERE a.name < b.name "
            "RETURN a.name AS a, b.name AS b, count(DISTINCT c) AS weight"
        )
        if not rel_rows:
            return None
        names = sorted({r["a"] for r in rel_rows} | {r["b"] for r in rel_rows})
        name_idx = {n: i for i, n in enumerate(names)}
        edges: list[tuple[int, int]] = []
        weights: list[float] = []
        seen: set[tuple[int, int]] = set()
        for r in rel_rows:
            ai, bi = name_idx.get(r["a"]), name_idx.get(r["b"])
            if ai is None or bi is None or ai == bi:
                continue
            key = (min(ai, bi), max(ai, bi))
            if key in seen:
                continue
            seen.add(key)
            edges.append((ai, bi))
            weights.append(max(0.0, float(r["weight"])))
        return names, edges, weights, "leidenalg"

    async def _partition_modularity(self, communities: dict[int, list[str]]) -> float | None:
        """Modularity of a given partition over the node2vec/co-occurrence graph."""
        try:
            import igraph as ig
        except ImportError:
            return None
        graph = await self._entity_graph()
        if graph is None:
            return None
        names, edge_pairs, weights, _ = graph
        name_to_comm: dict[str, int] = {
            name: cid for cid, members in communities.items() for name in members
        }
        g = ig.Graph(n=len(names), directed=False)
        if edge_pairs:
            g.add_edges(edge_pairs)
        next_id = (max(communities) if communities else -1) + 1
        membership: list[int] = []
        for name in names:
            if name in name_to_comm:
                membership.append(name_to_comm[name])
            else:
                membership.append(next_id)
                next_id += 1
        try:
            return float(g.modularity(membership, weights=weights if weights else None))
        except Exception as exc:
            logger.debug("modularity computation failed: %s", exc)
            return None

    async def write_topic_summaries(
        self,
        session_id: str,
        summarizer: Any = None,
        vec_store: Any = None,
        gds_modularity: float | None = None,
        community_algorithm: str | None = None,
    ) -> int:
        communities, local_modularity, silhouette, algorithm = await self._topic_communities()
        if not communities:
            logger.info("write_topic_summaries %s: no communities; skipping", session_id)
            return 0

        modularity = gds_modularity if gds_modularity is not None else local_modularity
        algorithm = (community_algorithm or "").strip() or algorithm

        # Load existing community hashes to skip unchanged communities
        existing_hashes: dict[int, str] = {}
        try:
            rows = await self._qry(
                "MATCH (t:TOPIC) WHERE t.community_id IS NOT NULL AND t.community_hash IS NOT NULL "
                "RETURN t.community_id AS cid, t.community_hash AS h"
            )
            existing_hashes = {int(r["cid"]): r["h"] for r in rows if r.get("cid") is not None}
        except Exception as exc:
            logger.warning("community-hash preload failed; recomputing all topics: %s", exc)

        created = 0
        for comm_id, members in communities.items():
            if len(members) < 2:
                continue
            comm_hash = hashlib.sha1(
                "|".join(sorted(members)).encode(), usedforsecurity=False
            ).hexdigest()
            if existing_hashes.get(comm_id) == comm_hash:
                # Refresh metadata only
                await self._cmd(
                    "MATCH (t:TOPIC {community_id: $cid}) "
                    "SET t.modularity = $mod, t.gds_modularity = $gmod, t.community_algorithm = $alg",
                    cid=comm_id,
                    mod=modularity,
                    gmod=gds_modularity,
                    alg=algorithm,
                )
                continue

            # Evidence passages
            evidence_rows = await self._qry(
                "MATCH (c:CHUNK)-[:MENTIONS]->(e:ENTITY) WHERE e.name IN $members "
                "WITH DISTINCT c RETURN c.text AS text, c.heading_path AS hp ORDER BY rand() LIMIT 100",
                members=members,
            )
            passages = "\n\n".join(
                (f"[{r['hp']}] " if r.get("hp") else "") + (r.get("text") or "")
                for r in evidence_rows
            )

            fallback_title = ", ".join(members[:5])
            title, summary = fallback_title, ""
            if summarizer is not None:
                try:
                    title, summary = await summarizer.summarize(
                        members=members, passages=passages, fallback_title=fallback_title
                    )
                except Exception as exc:
                    logger.warning("topic summarizer failed: %s", exc)

            topic_id = str(uuid.uuid5(_GK_NS, f"topic:{session_id}:{comm_id}"))
            # Embed the topic so it is matchable by semantic retrieval. A topic
            # summary is pre-aggregated multi-entity content — exactly the high-
            # level context that synthesis/multi-hop questions need. Falls back to
            # the member-name bag when there is no LLM summary (llm-free).
            embedding: list[float] = []
            emb_text = f"{title}. {summary}".strip().strip(".").strip() or ", ".join(members[:20])
            if emb_text:
                try:
                    from graphknows.storage.embedder import embed_one

                    embedding = await asyncio.to_thread(embed_one, emb_text)
                except Exception as exc:
                    logger.debug("topic embed failed: %s", exc)
            topic = {
                "id": topic_id,
                "topic_id": topic_id,
                "label": title,
                "summary": summary,
                "embedding": embedding,
                "community_id": comm_id,
                "community_hash": comm_hash,
                "entity_count": len(members),
                "entity_names": members,
                "session_ids": [session_id],
                "modularity": modularity,
                "gds_modularity": gds_modularity,
                "community_algorithm": algorithm,
                "silhouette": silhouette,
            }
            await self.upsert_topics([topic])

            # IN_TOPIC edges
            for name in members:
                try:
                    await self._cmd(
                        "MATCH (tp:TOPIC {id: $tid}), (e:ENTITY {name: $name}) MERGE (e)-[:IN_TOPIC]->(tp)",
                        tid=topic_id,
                        name=name,
                    )
                except Exception as exc:
                    logger.warning("IN_TOPIC edge failed for %r -> %s: %s", name, topic_id, exc)

            # Write to vector store if provided
            if vec_store is not None:
                try:
                    result = vec_store.write_ltm_topic(topic)
                    if asyncio.iscoroutine(result):
                        await result
                except Exception as exc:
                    logger.warning("write_ltm_topic failed: %s", exc)

            created += 1
        return created

    # ------------------------------------------------------------------
    # FRAME layer (deterministic FrameNet channel)
    # ------------------------------------------------------------------

    async def write_frames(self, session_id: str, frames: list[dict[str, Any]]) -> int:
        """Bulk-write FRAME instances for a session.

        Each ``frames`` item: ``{frame_type, chunk_id, lu, speaker,
        fes: [[role, filler], ...], fillers: [str, ...]}``. ``id`` is deterministic
        per (session, chunk, index) so a re-flush MERGEs idempotently.
        """
        if not frames:
            return 0
        rows = [
            {
                "id": f"{session_id}:{f['chunk_id']}:{i}",
                "sid": session_id,
                "ft": f["frame_type"],
                "cid": f["chunk_id"],
                "lu": f.get("lu", ""),
                "spk": f.get("speaker", "") or "",
                "fes": json.dumps(f.get("fes", [])),
                "fillers": [str(x).lower() for x in f.get("fillers", [])],
            }
            for i, f in enumerate(frames)
        ]
        await self._cmd(
            "UNWIND $rows AS r "
            "MERGE (n:FRAME {id: r.id}) "
            "SET n.session_id = r.sid, n.frame_type = r.ft, n.chunk_id = r.cid, "
            "n.lu = r.lu, n.speaker = r.spk, n.fes = r.fes, n.fillers = r.fillers",
            rows=rows,
        )
        return len(rows)

    async def query_frame_chunks(
        self, session_id: str, frame_types: list[str]
    ) -> list[dict[str, Any]]:
        """Return FRAME rows for *session_id* whose frame_type is in *frame_types*."""
        if not frame_types:
            return []
        rows = await self._qry(
            "MATCH (n:FRAME) WHERE n.session_id = $sid AND n.frame_type IN $fts "
            "RETURN n.chunk_id AS chunk_id, n.frame_type AS frame_type, "
            "n.speaker AS speaker, n.fillers AS fillers",
            sid=session_id,
            fts=list(frame_types),
        )
        return [
            {
                "chunk_id": r.get("chunk_id"),
                "frame_type": r.get("frame_type"),
                "speaker": (r.get("speaker") or "").lower(),
                "fillers": list(r.get("fillers") or []),
            }
            for r in rows
            if r.get("chunk_id")
        ]

    async def upsert_topics(self, topics: list[dict[str, Any]]) -> None:
        for t in topics:
            tid = t.get("id") or t.get("topic_id", "")
            if not tid:
                continue
            await self._cmd(
                "MERGE (tp:TOPIC {id: $id}) "
                "SET tp.topic_id = $tid, tp.label = $label, tp.summary = $summary, "
                "tp.community_id = $cid, tp.community_hash = $ch, "
                "tp.entity_count = $ec, tp.entity_names = $en, "
                "tp.session_ids = $sids, tp.modularity = $mod, "
                "tp.gds_modularity = $gmod, tp.community_algorithm = $alg, "
                "tp.silhouette = $sil",
                id=tid,
                tid=tid,
                label=t.get("label", ""),
                summary=t.get("summary", ""),
                cid=t.get("community_id"),
                ch=t.get("community_hash", ""),
                ec=t.get("entity_count", 0),
                en=t.get("entity_names") or [],
                sids=t.get("session_ids") or [],
                mod=t.get("modularity"),
                gmod=t.get("gds_modularity"),
                alg=t.get("community_algorithm", ""),
                sil=t.get("silhouette"),
            )

    # ------------------------------------------------------------------
    # Corpus reads (retrieval)
    # ------------------------------------------------------------------

    async def full_text_search_chunks(
        self,
        terms: list[str],
        session_id: str | None = None,
        top_k: int = 5,
        require_all: bool = False,
    ) -> list[dict[str, Any]]:
        """Return LTM CHUNK nodes whose text contains the given terms.

        Uses dynamic OR/AND WHERE conditions (ArcadeDB does not support the
        any() list predicate). Filters by session_id when provided.
        """
        if not terms:
            return []
        safe = [t.replace("'", "") for t in terms]
        op = " AND " if require_all else " OR "
        text_where = op.join(f"toLower(c.text) CONTAINS '{t}'" for t in safe)
        if session_id:
            safe_sid = session_id.replace("'", "")
            full_where = f"({text_where}) AND c.session_id = '{safe_sid}'"
        else:
            full_where = text_where
        rows = await self._qry(
            f"MATCH (c:CHUNK) WHERE {full_where} RETURN c.id AS chunk_id, c.text AS text LIMIT $k",
            k=top_k,
        )
        return [{"chunk_id": r["chunk_id"], "text": r.get("text", "")} for r in rows]

    def get_entity_by_name(self, name: str) -> dict[str, Any] | None:
        """Sync wrapper — runs async query in a new event loop or worker thread."""
        import asyncio as _asyncio
        import concurrent.futures

        try:
            _asyncio.get_running_loop()
        except RuntimeError:
            return _asyncio.run(self._get_entity_by_name_async(name))
        # A loop is already running; offload to a worker thread to avoid
        # RuntimeError("This event loop is already running").
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(_asyncio.run, self._get_entity_by_name_async(name))
            return future.result()

    async def _get_entity_by_name_async(self, name: str) -> dict[str, Any] | None:
        rows = await self._qry(
            "MATCH (e:ENTITY) WHERE toLower(e.name) = toLower($name) "
            "OPTIONAL MATCH (e)-[:APPEARED_IN]->(seg:SEGMENT) "
            "RETURN e.id AS id, e.name AS name, e.label AS label, e.type AS type, "
            "e.confidence AS confidence, e.community_id AS community_id, "
            "e.pagerank AS pagerank, e.betweenness AS betweenness, "
            "e.ingested_at AS ingested_at, e.merged_session_count AS merged_session_count, "
            "count(DISTINCT seg) AS segment_count",
            name=name,
        )
        if not rows:
            return None
        row = {k: v for k, v in rows[0].items() if not k.startswith("@")}
        return row

    def list_topic_nodes(self, min_entities: int = 3) -> list[dict[str, Any]]:
        import asyncio as _asyncio
        import concurrent.futures

        try:
            _asyncio.get_running_loop()
        except RuntimeError:
            return _asyncio.run(self._list_topic_nodes_async(min_entities))
        # A loop is already running; offload to a worker thread to avoid
        # RuntimeError("This event loop is already running").
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(_asyncio.run, self._list_topic_nodes_async(min_entities))
            return future.result()

    async def _list_topic_nodes_async(self, min_entities: int) -> list[dict[str, Any]]:
        rows = await self._qry(
            "MATCH (t:TOPIC) "
            "OPTIONAL MATCH (e)-[:IN_TOPIC]->(t) "
            "WITH t, count(DISTINCT e) AS computed_entity_count "
            "WHERE computed_entity_count >= $min OR t.entity_count >= $min "
            "RETURN t.id AS id, t.label AS label, t.community_id AS community_id, "
            "t.summary AS summary, t.entity_count AS entity_count, "
            "t.entity_names AS entity_names, t.community_hash AS community_hash, "
            "t.modularity AS modularity, t.community_algorithm AS community_algorithm, "
            "computed_entity_count",
            min=min_entities,
        )
        topics = []
        for r in rows:
            ec = int(r.get("entity_count") or r.get("computed_entity_count") or 0)
            if ec < min_entities:
                continue
            topics.append(
                {
                    "topic_id": r.get("id"),
                    "label": r.get("label"),
                    "community_id": r.get("community_id"),
                    "summary": r.get("summary"),
                    "entity_count": ec,
                    "entity_names": r.get("entity_names") or [],
                    "community_hash": r.get("community_hash"),
                    "modularity": r.get("modularity"),
                    "community_algorithm": r.get("community_algorithm"),
                }
            )
        return sorted(topics, key=lambda t: t["entity_count"], reverse=True)

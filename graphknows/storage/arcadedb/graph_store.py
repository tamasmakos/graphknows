"""ArcadeDB single-database graph store (golden layer).

One ``GraphStore`` per namespace database (``mem`` / ``mem_<ns>``) absorbs the
roles of the legacy STM, LTM, and vector stores: SESSION/FILE/CHUNK spine,
ENTITY graph keyed on ``name_norm``, dynamic relation edges with chunk-level
``evidence`` provenance, TEMPORAL/TOPIC/FRAME layers, and vector search over
embeddings stored directly on CHUNK/ENTITY/TOPIC vertices.

Conventions (schema-wide):
- ``type`` not ``label``; ``confidence`` not ``score``;
  ``valid_at``/``invalid_at``/``is_current`` for bi-temporal relation edges.
- Lifecycle ``state`` in {``raw``, ``consolidated``} on SESSION/CHUNK/ENTITY
  and relation edges; writes always land ``raw``.

ArcadeDB constraints honoured here:
- Cypher has no DDL surface — schema uses SQL.
- Embedding values MUST be written as inline SQL literals (LIST params
  conflict with the LSM_VECTOR index); vector search uses ``vectorNeighbors``
  with no FROM clause (see ``_vec_search_sql``).
- openCypher has no ``IN $list`` / ``any(x IN $list)`` — every interpolated
  string goes through :mod:`graphknows.storage.arcadedb._sql`.
- Dynamic relation edge types must be declared before MERGE.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from typing import Any

from graphknows.storage.arcadedb._labels import _safe_rel_label
from graphknows.storage.arcadedb._sql import quote, quoted_list, sanitize
from graphknows.storage.arcadedb.client import ArcadeDBClient

logger = logging.getLogger(__name__)

_DB = "mem"

# Bulk CHUNK writes are capped per statement to keep statement/body size bounded
# (embeddings are large; see plan §8 statement-size risk). Tunable.
_CHUNK_BATCH = 20

# Vector-index dimension token, replaced with the real dimension in
# ensure_schema(dims) so the index matches the configured embedder.
_VEC_DIM = "__VEC_DIM__"

# Graph OLAP analytical view backing `CALL algo.*` (pageRank/node2vec/louvain).
# Ported from the retired ltm_store._ensure_analytical_view — see that method's
# docstring for why the view is required (CSR engine vs. slow OLTP traversal).
_OLAP_VIEW = "mem_olap"

# Known single-valued relation predicates — a new fact tombstones the old edge
# (sets invalid_at / is_current=false) instead of coexisting with it.
# The FrameNet-derived labels (RESIDENCE, BEING_EMPLOYED) map the obvious
# open-vocabulary frame names onto the same behavior. Phase 3 consolidation
# replaces this hardcoded set with frame-semantics-driven functional detection.
_SINGLE_VALUED_RELATIONS: frozenset[str] = frozenset(
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
        "RESIDENCE",
        "BEING_EMPLOYED",
    }
)

# Vertex labels wiped by clear_all() / walked by delete_session().
_VERTEX_LABELS = ("FRAME", "CHUNK", "FILE", "TOPIC", "TEMPORAL", "ENTITY", "SESSION")

# ---------------------------------------------------------------------------
# DDL — golden-layer schema (plan §1). Order matters: types before properties,
# properties before indexes. Executed idempotently in ensure_schema().
# ---------------------------------------------------------------------------

_CORE_DDL: list[tuple[str, str]] = [
    # ---- Vertex types -----------------------------------------------------
    ("sql", "CREATE VERTEX TYPE SESSION IF NOT EXISTS"),
    ("sql", "CREATE PROPERTY SESSION.session_id IF NOT EXISTS STRING"),
    ("sql", "CREATE PROPERTY SESSION.started_at IF NOT EXISTS STRING"),
    ("sql", "CREATE PROPERTY SESSION.state IF NOT EXISTS STRING"),
    ("sql", "CREATE PROPERTY SESSION.consolidated_at IF NOT EXISTS STRING"),
    ("sql", "CREATE PROPERTY SESSION.label IF NOT EXISTS STRING"),
    ("sql", "CREATE INDEX ON SESSION(session_id) UNIQUE"),
    ("sql", "CREATE VERTEX TYPE FILE IF NOT EXISTS"),
    ("sql", "CREATE PROPERTY FILE.file_id IF NOT EXISTS STRING"),
    ("sql", "CREATE PROPERTY FILE.name IF NOT EXISTS STRING"),
    ("sql", "CREATE PROPERTY FILE.media_type IF NOT EXISTS STRING"),
    ("sql", "CREATE PROPERTY FILE.session_id IF NOT EXISTS STRING"),
    ("sql", "CREATE INDEX ON FILE(file_id) UNIQUE"),
    ("sql", "CREATE VERTEX TYPE CHUNK IF NOT EXISTS"),
    ("sql", "CREATE PROPERTY CHUNK.chunk_id IF NOT EXISTS STRING"),
    ("sql", "CREATE PROPERTY CHUNK.text IF NOT EXISTS STRING"),
    ("sql", "CREATE PROPERTY CHUNK.seq IF NOT EXISTS INTEGER"),
    ("sql", "CREATE PROPERTY CHUNK.session_id IF NOT EXISTS STRING"),
    ("sql", "CREATE PROPERTY CHUNK.kind IF NOT EXISTS STRING"),
    ("sql", "CREATE PROPERTY CHUNK.speaker IF NOT EXISTS STRING"),
    ("sql", "CREATE PROPERTY CHUNK.ts IF NOT EXISTS STRING"),
    ("sql", "CREATE PROPERTY CHUNK.state IF NOT EXISTS STRING"),
    ("sql", "CREATE PROPERTY CHUNK.created_at IF NOT EXISTS STRING"),
    ("sql", "CREATE PROPERTY CHUNK.memory_class IF NOT EXISTS STRING"),
    ("sql", "CREATE PROPERTY CHUNK.atom_type IF NOT EXISTS STRING"),
    ("sql", "CREATE PROPERTY CHUNK.entity_names IF NOT EXISTS LIST"),
    ("sql", "CREATE PROPERTY CHUNK.embedding IF NOT EXISTS LIST"),
    ("sql", "CREATE INDEX ON CHUNK(chunk_id) UNIQUE"),
    ("sql", "CREATE INDEX ON CHUNK(session_id) NOTUNIQUE"),
    ("sql", "CREATE INDEX ON CHUNK(state) NOTUNIQUE"),
    (
        "sql",
        f"CREATE INDEX IF NOT EXISTS ON CHUNK(embedding) LSM_VECTOR "
        f"METADATA {{dimensions:{_VEC_DIM},distanceFunction:'COSINE'}}",
    ),
    ("sql", "CREATE VERTEX TYPE ENTITY IF NOT EXISTS"),
    ("sql", "CREATE PROPERTY ENTITY.name_norm IF NOT EXISTS STRING"),
    ("sql", "CREATE PROPERTY ENTITY.name IF NOT EXISTS STRING"),
    ("sql", "CREATE PROPERTY ENTITY.type IF NOT EXISTS STRING"),
    ("sql", "CREATE PROPERTY ENTITY.confidence IF NOT EXISTS DOUBLE"),
    ("sql", "CREATE PROPERTY ENTITY.aliases IF NOT EXISTS LIST"),
    ("sql", "CREATE PROPERTY ENTITY.state IF NOT EXISTS STRING"),
    ("sql", "CREATE PROPERTY ENTITY.pagerank IF NOT EXISTS DOUBLE"),
    ("sql", "CREATE PROPERTY ENTITY.community_id IF NOT EXISTS INTEGER"),
    ("sql", "CREATE PROPERTY ENTITY.graph_embedding IF NOT EXISTS LIST"),
    ("sql", "CREATE PROPERTY ENTITY.embedding IF NOT EXISTS LIST"),
    ("sql", "CREATE INDEX ON ENTITY(name_norm) UNIQUE"),
    ("sql", "CREATE INDEX ON ENTITY(state) NOTUNIQUE"),
    (
        "sql",
        f"CREATE INDEX IF NOT EXISTS ON ENTITY(embedding) LSM_VECTOR "
        f"METADATA {{dimensions:{_VEC_DIM},distanceFunction:'COSINE'}}",
    ),
    # Canonical temporal nodes — one global node per resolved ISO date string.
    ("sql", "CREATE VERTEX TYPE TEMPORAL IF NOT EXISTS"),
    ("sql", "CREATE PROPERTY TEMPORAL.date IF NOT EXISTS STRING"),
    ("sql", "CREATE INDEX ON TEMPORAL(date) UNIQUE"),
    ("sql", "CREATE VERTEX TYPE TOPIC IF NOT EXISTS"),
    ("sql", "CREATE PROPERTY TOPIC.topic_id IF NOT EXISTS STRING"),
    ("sql", "CREATE PROPERTY TOPIC.summary IF NOT EXISTS STRING"),
    ("sql", "CREATE PROPERTY TOPIC.modularity IF NOT EXISTS DOUBLE"),
    ("sql", "CREATE PROPERTY TOPIC.embedding IF NOT EXISTS LIST"),
    ("sql", "CREATE INDEX ON TOPIC(topic_id) UNIQUE"),
    (
        "sql",
        f"CREATE INDEX IF NOT EXISTS ON TOPIC(embedding) LSM_VECTOR "
        f"METADATA {{dimensions:{_VEC_DIM},distanceFunction:'COSINE'}}",
    ),
    # Semantic-frame instances (deterministic FrameNet layer). Evidence is the
    # EVOKED_BY edge to the CHUNK — no chunk_id string property.
    ("sql", "CREATE VERTEX TYPE FRAME IF NOT EXISTS"),
    ("sql", "CREATE PROPERTY FRAME.id IF NOT EXISTS STRING"),
    ("sql", "CREATE PROPERTY FRAME.session_id IF NOT EXISTS STRING"),
    ("sql", "CREATE PROPERTY FRAME.frame_type IF NOT EXISTS STRING"),
    ("sql", "CREATE PROPERTY FRAME.lu IF NOT EXISTS STRING"),
    ("sql", "CREATE PROPERTY FRAME.speaker IF NOT EXISTS STRING"),
    ("sql", "CREATE PROPERTY FRAME.fes IF NOT EXISTS STRING"),
    ("sql", "CREATE PROPERTY FRAME.fillers IF NOT EXISTS LIST"),
    ("sql", "CREATE INDEX ON FRAME(id) UNIQUE"),
    ("sql", "CREATE INDEX ON FRAME(session_id) NOTUNIQUE"),
    ("sql", "CREATE INDEX ON FRAME(frame_type) NOTUNIQUE"),
    # Community-quality vertex — one per Louvain community_id, holding the
    # native modularityScore/conductance metrics (see compute_community_quality).
    ("sql", "CREATE VERTEX TYPE COMMUNITY IF NOT EXISTS"),
    ("sql", "CREATE PROPERTY COMMUNITY.community_id IF NOT EXISTS INTEGER"),
    ("sql", "CREATE PROPERTY COMMUNITY.conductance IF NOT EXISTS DOUBLE"),
    ("sql", "CREATE PROPERTY COMMUNITY.internal_edges IF NOT EXISTS INTEGER"),
    ("sql", "CREATE PROPERTY COMMUNITY.boundary_edges IF NOT EXISTS INTEGER"),
    ("sql", "CREATE PROPERTY COMMUNITY.node_count IF NOT EXISTS INTEGER"),
    ("sql", "CREATE PROPERTY COMMUNITY.modularity IF NOT EXISTS DOUBLE"),
    ("sql", "CREATE PROPERTY COMMUNITY.size IF NOT EXISTS INTEGER"),
    ("sql", "CREATE INDEX ON COMMUNITY(community_id) UNIQUE"),
    # ---- Edge types -------------------------------------------------------
    ("sql", "CREATE EDGE TYPE PART_OF IF NOT EXISTS"),  # CHUNK -> FILE
    ("sql", "CREATE EDGE TYPE IN_SESSION IF NOT EXISTS"),  # FILE -> SESSION
    ("sql", "CREATE EDGE TYPE MENTIONS IF NOT EXISTS"),  # CHUNK -> ENTITY
    ("sql", "CREATE PROPERTY MENTIONS.confidence IF NOT EXISTS DOUBLE"),
    ("sql", "CREATE EDGE TYPE MENTIONS_DATE IF NOT EXISTS"),  # CHUNK -> TEMPORAL
    ("sql", "CREATE EDGE TYPE HAS_TOPIC IF NOT EXISTS"),  # CHUNK -> TOPIC
    ("sql", "CREATE EDGE TYPE EVOKED_BY IF NOT EXISTS"),  # FRAME -> CHUNK
    ("sql", "CREATE PROPERTY EVOKED_BY.confidence IF NOT EXISTS DOUBLE"),
    ("sql", "CREATE EDGE TYPE IN_COMMUNITY IF NOT EXISTS"),  # ENTITY -> COMMUNITY
]


def _emb_sql(emb: list[float]) -> str:
    """Format an embedding as an ArcadeDB SQL array literal (inline, not a param)."""
    return "[" + ",".join(str(x) for x in emb) + "]"


def _vec_search_sql(index: str, emb: list[float], k: int) -> str:
    """Return SQL for a vectorNeighbors ANN search.

    No FROM clause — the index already knows its type. Adding FROM <type>
    makes ArcadeDB invoke vectorNeighbors once *per vertex*, returning N*k rows
    and making deduplication order-dependent.
    """
    return f'SELECT expand(vectorNeighbors("{index}", {_emb_sql(emb)}, {k}))'


def _name_norm(name: str) -> str:
    """Canonical ENTITY merge key: casefolded, whitespace-collapsed name."""
    return " ".join(name.split()).casefold()


def _structural_types_literal() -> str:
    """Structural edge types as a Cypher list literal (no IN $list support)."""
    from graphknows.ingestion.models.taxonomy import STRUCTURAL_REL_TYPES

    return quoted_list(sorted(STRUCTURAL_REL_TYPES))


class GraphStore:
    """Single-database ArcadeDB store for one namespace (golden layer).

    Args:
        client: Shared ``ArcadeDBClient`` (connect() is idempotent).
        db: Database name, from :func:`graphknows.storage.namespace.db_name`.
    """

    def __init__(self, client: ArcadeDBClient, db: str = _DB) -> None:
        self._c = client
        self._db = db
        self._declared_edge_types: set[str] = set()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @property
    def client(self) -> ArcadeDBClient:
        """The underlying ArcadeDB client (may be shared)."""
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

    async def drop_database(self) -> None:
        """Drop this store's entire ArcadeDB database (destructive)."""
        await self._c.drop_database(self._db)

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    async def ensure_schema(self, dims: int) -> None:
        """Create the database and the golden-layer schema idempotently.

        Args:
            dims: Embedding dimension for the LSM_VECTOR indexes (must match
                the configured embedder).
        """
        await self._c.create_database(self._db)
        for lang, stmt in _CORE_DDL:
            stmt = stmt.replace(_VEC_DIM, str(dims))
            try:
                await self._c.command(self._db, stmt, language=lang)
            except Exception as exc:
                exc_s = str(exc).lower()
                if "already exists" in exc_s or "duplicate" in exc_s:
                    logger.debug("GraphStore DDL already exists (idempotent): %s", stmt[:60])
                else:
                    logger.warning("GraphStore DDL failed: %s | %s", stmt[:60], exc)
                    raise
        logger.info("GraphStore schema ensured (db=%s, dims=%d)", self._db, dims)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _cmd(self, cypher: str, **params: Any) -> list[dict]:
        return await self._c.command(self._db, cypher, params=params or None)

    async def _qry(self, cypher: str, **params: Any) -> list[dict]:
        return await self._c.query(self._db, cypher, params=params or None)

    async def _ensure_edge_type(self, safe_label: str) -> None:
        """Declare a dynamic relation edge type before first MERGE (idempotent)."""
        if safe_label in self._declared_edge_types:
            return
        try:
            await self._c.command(
                self._db, f"CREATE EDGE TYPE {safe_label} IF NOT EXISTS", language="sql"
            )
            self._declared_edge_types.add(safe_label)
        except Exception as exc:
            exc_s = str(exc).lower()
            if "already exists" in exc_s or "duplicate" in exc_s:
                self._declared_edge_types.add(safe_label)
            else:
                logger.warning("ensure_edge_type %s failed: %s", safe_label, exc)
                raise

    async def _set_embedding(
        self, vertex_type: str, key_prop: str, key: str, emb: list[float]
    ) -> None:
        """Set an embedding on a vertex matched by its key property.

        The match key is a Cypher parameter (handles quotes/newlines safely);
        the embedding stays an inline literal because LIST params conflict with
        the LSM_VECTOR index.
        """
        if not emb:
            return
        stmt = (
            f"MATCH (n:{vertex_type} {{{key_prop}: $key}}) SET n.embedding = {_emb_sql(emb)}"
        )  # nosec B608
        await self._c.command(self._db, stmt, params={"key": key})

    @staticmethod
    def _dedup_rows(rows: list[dict]) -> list[dict]:
        """Deduplicate vectorNeighbors results by @rid."""
        seen: set[str] = set()
        out: list[dict] = []
        for r in rows:
            rid = r.get("@rid", r.get("chunk_id", ""))
            if rid not in seen:
                seen.add(rid)
                out.append(r)
        return out

    # ------------------------------------------------------------------
    # Write API — spine (SESSION / FILE / CHUNK)
    # ------------------------------------------------------------------

    async def ensure_session(self, session_id: str, started_at: str = "") -> None:
        """Upsert a SESSION vertex; state starts ``raw``."""
        await self._cmd(
            "MERGE (s:SESSION {session_id: $sid}) "
            "SET s.started_at = coalesce(s.started_at, $sa), "
            "    s.state = coalesce(s.state, 'raw')",
            sid=session_id,
            sa=started_at,
        )

    async def ensure_file(
        self, file_id: str, session_id: str, name: str = "", media_type: str = ""
    ) -> None:
        """Upsert a FILE vertex linked to its SESSION via FILE-[:IN_SESSION]->SESSION."""
        await self._cmd(
            "MERGE (f:FILE {file_id: $fid}) "
            "SET f.name = $name, f.media_type = $mt, f.session_id = $sid "
            "WITH f MATCH (s:SESSION {session_id: $sid}) "
            "MERGE (f)-[:IN_SESSION]->(s)",
            fid=file_id,
            name=name,
            mt=media_type,
            sid=session_id,
        )

    async def write_chunk(self, chunk: dict[str, Any]) -> None:
        """Upsert one CHUNK vertex (state='raw') with its PART_OF edge and embedding.

        ``chunk`` keys: chunk_id (required), session_id, file_id, text, seq,
        kind, speaker, ts, memory_class, atom_type, entity_names, embedding.
        """
        await self.write_chunks_bulk([chunk])

    async def write_chunks_bulk(self, chunks: list[dict[str, Any]]) -> None:
        """Upsert CHUNK vertices in batches of ``_CHUNK_BATCH`` per statement.

        Per batch: one UNWIND MERGE for scalar properties, one UNWIND MERGE for
        PART_OF edges. Embeddings are written per chunk afterwards — they must
        be inline SQL literals, so they cannot ride in the batched params.
        """
        if not chunks:
            return
        now = datetime.now(UTC).isoformat()
        for start in range(0, len(chunks), _CHUNK_BATCH):
            batch = chunks[start : start + _CHUNK_BATCH]
            rows = [
                {
                    "cid": c["chunk_id"],
                    "sid": c.get("session_id", ""),
                    "fid": c.get("file_id", ""),
                    "text": c.get("text", ""),
                    "seq": int(c.get("seq") or 0),
                    "kind": c.get("kind", "text"),
                    "speaker": c.get("speaker", "") or "",
                    "ts": c.get("ts", "") or "",
                    "mc": c.get("memory_class", "") or "",
                    "at": c.get("atom_type", "") or "",
                    "en": list(c.get("entity_names") or []),
                }
                for c in batch
            ]
            await self._cmd(
                "UNWIND $rows AS row "
                "MERGE (c:CHUNK {chunk_id: row.cid}) "
                "SET c.session_id = row.sid, c.text = row.text, c.seq = row.seq, "
                "    c.kind = row.kind, c.speaker = row.speaker, c.ts = row.ts, "
                "    c.memory_class = row.mc, c.atom_type = row.at, "
                "    c.entity_names = row.en, "
                "    c.state = coalesce(c.state, 'raw'), "
                "    c.created_at = coalesce(c.created_at, $now)",
                rows=rows,
                now=now,
            )
            edge_rows = [r for r in rows if r["fid"]]
            if edge_rows:
                await self._cmd(
                    "UNWIND $rows AS row "
                    "MATCH (c:CHUNK {chunk_id: row.cid}), (f:FILE {file_id: row.fid}) "
                    "MERGE (c)-[:PART_OF]->(f)",
                    rows=edge_rows,
                )
            for c in batch:
                if c.get("embedding"):
                    await self._set_embedding("CHUNK", "chunk_id", c["chunk_id"], c["embedding"])

    # ------------------------------------------------------------------
    # Write API — entities / relations
    # ------------------------------------------------------------------

    async def write_entities_bulk(
        self,
        session_id: str,
        chunk_id: str,
        entities: list[dict[str, Any]],
    ) -> None:
        """Upsert ENTITY vertices (MERGE on ``name_norm``) + MENTIONS edges.

        Callers pre-filter entities through the writer-layer filter
        (``graph_entities`` — drops temporal/numeric spans); this store does
        not duplicate that policy.

        Semantics per entity (dict with "name"; optional "type", "confidence"):
        - MERGE key is the casefolded name; ``name`` keeps the first-seen surface.
        - When a later surface differs from the canonical name it is appended
          to ``aliases`` (deduplicated).
        - ``confidence`` keeps the max seen; ``state`` starts ``raw``.
        - MENTIONS edge CHUNK->ENTITY carries ``confidence``.
        """
        if not entities:
            return
        rows: list[dict[str, Any]] = []
        for e in entities:
            name = str(e.get("name") or "").strip()
            if not name:
                continue
            conf = e.get("confidence", e.get("score", 1.0))
            rows.append(
                {
                    "nn": _name_norm(name),
                    "name": name,
                    "etype": e.get("type") or "ENTITY",
                    "conf": float(conf) if isinstance(conf, (int, float)) else 1.0,
                }
            )
        if not rows:
            return

        # Pass 1: upsert ENTITY nodes on name_norm.
        await self._cmd(
            "UNWIND $rows AS row "
            "MERGE (e:ENTITY {name_norm: row.nn}) "
            "SET e.name = coalesce(e.name, row.name), "
            "    e.type = coalesce(e.type, row.etype), "
            "    e.state = coalesce(e.state, 'raw'), "
            "    e.confidence = CASE "
            "        WHEN row.conf > coalesce(e.confidence, 0) THEN row.conf "
            "        ELSE e.confidence "
            "    END, "
            "    e.aliases = CASE "
            "        WHEN row.name <> e.name AND NOT row.name IN coalesce(e.aliases, []) "
            "        THEN coalesce(e.aliases, []) + [row.name] "
            "        ELSE coalesce(e.aliases, []) "
            "    END",
            rows=rows,
        )

        # Pass 2: MENTIONS edges with confidence.
        await self._cmd(
            "MATCH (c:CHUNK {chunk_id: $cid}) "
            "UNWIND $rows AS row "
            "MATCH (e:ENTITY {name_norm: row.nn}) "
            "MERGE (c)-[m:MENTIONS]->(e) "
            "SET m.confidence = CASE "
            "    WHEN row.conf > coalesce(m.confidence, 0) THEN row.conf "
            "    ELSE m.confidence "
            "END",
            cid=chunk_id,
            rows=rows,
        )

    async def _supersede_edge(self, safe_rel: str, src_norm: str, dst_norm: str, now: str) -> None:
        """Tombstone any current edge for (src, predicate) pointing elsewhere.

        Called only for single-valued predicates. Sets ``invalid_at``/
        ``is_current=false`` instead of deleting, preserving history.
        """
        try:
            await self._cmd(
                f"MATCH (a:ENTITY {{name_norm: $src}})"
                f"-[r:{safe_rel} {{is_current: true}}]"
                f"->(b:ENTITY) "
                f"WHERE b.name_norm <> $dst "
                f"SET r.invalid_at = $now, r.is_current = false",
                src=src_norm,
                dst=dst_norm,
                now=now,
            )
        except Exception as exc:
            logger.debug(
                "_supersede_edge %s (%s -> !=%s) skipped: %s", safe_rel, src_norm, dst_norm, exc
            )

    async def write_relation(
        self,
        src_name: str,
        predicate: str,
        dst_name: str,
        fact: str = "",
        chunk_id: str = "",
        confidence: float = 0.5,
        valid_at: str | None = None,
        source: str = "",
    ) -> None:
        """MERGE a dynamic relation edge (type = taxonomy predicate).

        On a duplicate (src, predicate, dst) the ``chunk_id`` is appended to the
        edge's ``evidence`` list (deduplicated) and ``confidence`` keeps the max.
        Single-valued predicates tombstone the previous current edge first.
        """
        safe = _safe_rel_label(predicate)
        if not safe or not src_name or not dst_name:
            return
        src_norm = _name_norm(src_name)
        dst_norm = _name_norm(dst_name)
        now = datetime.now(UTC).isoformat()

        await self._ensure_edge_type(safe)
        if predicate.upper() in _SINGLE_VALUED_RELATIONS:
            await self._supersede_edge(safe, src_norm, dst_norm, now)

        # Evidence is appended via an interpolated quoted literal — openCypher
        # cannot express "$param IN list-property" reliably.
        cid_lit = quote(chunk_id) if chunk_id else ""
        evidence_set = (
            f"r.evidence = CASE "
            f"WHEN {cid_lit} IN coalesce(r.evidence, []) THEN r.evidence "
            f"ELSE coalesce(r.evidence, []) + [{cid_lit}] END, "
            if chunk_id
            else "r.evidence = coalesce(r.evidence, []), "
        )
        await self._cmd(
            f"MERGE (a:ENTITY {{name_norm: $src_nn}}) "
            f"SET a.name = coalesce(a.name, $src), a.state = coalesce(a.state, 'raw') "
            f"MERGE (b:ENTITY {{name_norm: $dst_nn}}) "
            f"SET b.name = coalesce(b.name, $dst), b.state = coalesce(b.state, 'raw') "
            f"MERGE (a)-[r:{safe}]->(b) "
            f"SET r.predicate = $predicate, "
            f"    r.fact = coalesce(r.fact, $fact), "
            f"    {evidence_set}"
            f"    r.confidence = CASE "
            f"        WHEN $conf > coalesce(r.confidence, 0) THEN $conf "
            f"        ELSE r.confidence "
            f"    END, "
            f"    r.valid_at = coalesce(r.valid_at, $valid_at), "
            f"    r.invalid_at = null, "
            f"    r.is_current = true, "
            f"    r.state = coalesce(r.state, 'raw'), "
            f"    r.source = coalesce(r.source, $source), "
            f"    r.created_at = coalesce(r.created_at, $now)",
            src_nn=src_norm,
            dst_nn=dst_norm,
            src=src_name,
            dst=dst_name,
            predicate=predicate,
            fact=fact,
            conf=float(confidence),
            valid_at=valid_at or None,
            source=source,
            now=now,
        )

    # ------------------------------------------------------------------
    # Write API — temporal / frames / topics
    # ------------------------------------------------------------------

    async def write_temporal(self, chunk_id: str, iso_dates: list[str]) -> None:
        """MERGE TEMPORAL nodes (key: ISO ``date`` string) + MENTIONS_DATE edges."""
        dates = [str(d).strip() for d in iso_dates if str(d).strip()]
        if not dates:
            return
        await self._cmd(
            "MATCH (c:CHUNK {chunk_id: $cid}) "
            "UNWIND $dates AS iso "
            "MERGE (d:TEMPORAL {date: iso}) "
            "MERGE (c)-[:MENTIONS_DATE]->(d)",
            cid=chunk_id,
            dates=dates,
        )

    async def write_frames(self, session_id: str, frames: list[dict[str, Any]]) -> int:
        """Bulk-write FRAME vertices + EVOKED_BY evidence edges to their chunks.

        Each ``frames`` item: ``{frame_type, chunk_id, lu, speaker,
        fes: [[role, filler], ...], fillers: [str, ...], confidence?}``.
        ``id`` is deterministic per (session, chunk, index) so re-runs MERGE
        idempotently. Evidence is the FRAME-[:EVOKED_BY]->CHUNK edge — there is
        no ``chunk_id`` property on FRAME.
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
                "conf": float(f.get("confidence", 1.0)),
            }
            for i, f in enumerate(frames)
        ]
        await self._cmd(
            "UNWIND $rows AS r "
            "MERGE (n:FRAME {id: r.id}) "
            "SET n.session_id = r.sid, n.frame_type = r.ft, "
            "n.lu = r.lu, n.speaker = r.spk, n.fes = r.fes, n.fillers = r.fillers",
            rows=rows,
        )
        await self._cmd(
            "UNWIND $rows AS r "
            "MATCH (n:FRAME {id: r.id}), (c:CHUNK {chunk_id: r.cid}) "
            "MERGE (n)-[ev:EVOKED_BY]->(c) "
            "SET ev.confidence = r.conf",
            rows=rows,
        )
        return len(rows)

    async def write_topics(self, topics: list[dict[str, Any]]) -> int:
        """Upsert TOPIC vertices + HAS_TOPIC edges from their evidence chunks.

        Each ``topics`` item: ``{topic_id, summary, modularity, embedding?,
        chunk_ids?: [str, ...]}``.
        """
        written = 0
        for t in topics:
            tid = str(t.get("topic_id") or "").strip()
            if not tid:
                continue
            await self._cmd(
                "MERGE (tp:TOPIC {topic_id: $tid}) "
                "SET tp.summary = $summary, tp.modularity = $mod",
                tid=tid,
                summary=t.get("summary", ""),
                mod=t.get("modularity"),
            )
            if t.get("embedding"):
                await self._set_embedding("TOPIC", "topic_id", tid, t["embedding"])
            chunk_ids = [c for c in (t.get("chunk_ids") or []) if c]
            if chunk_ids:
                await self._cmd(
                    "MATCH (tp:TOPIC {topic_id: $tid}) "
                    "UNWIND $cids AS cid "
                    "MATCH (c:CHUNK {chunk_id: cid}) "
                    "MERGE (c)-[:HAS_TOPIC]->(tp)",
                    tid=tid,
                    cids=chunk_ids,
                )
            written += 1
        return written

    # ------------------------------------------------------------------
    # Read API — chunks
    # ------------------------------------------------------------------

    async def search_chunks_ann(
        self,
        embedding: list[float],
        top_k: int,
        session_id: str | None = None,
        state: str | None = None,
    ) -> list[dict[str, Any]]:
        """ANN search over CHUNK.embedding via vectorNeighbors.

        session/state filters are applied client-side over an over-fetched
        candidate set (vectorNeighbors has no WHERE support).
        """
        if not embedding:
            return []
        k_pre = max(top_k * 20, 200) if (session_id or state) else top_k
        sql = _vec_search_sql("CHUNK[embedding]", embedding, k_pre)
        try:
            rows = await self._c.query(self._db, sql, language="sql")
        except Exception as exc:
            logger.warning("search_chunks_ann failed: %s", exc)
            return []
        rows = self._dedup_rows(rows)
        results: list[dict[str, Any]] = []
        for r in rows:
            if session_id and r.get("session_id") != session_id:
                continue
            if state and r.get("state") != state:
                continue
            dist = float(r.get("distance") or 1.0)
            results.append(
                {
                    "chunk_id": r.get("chunk_id", ""),
                    "session_id": r.get("session_id") or "",
                    "text": r.get("text", ""),
                    "seq": r.get("seq"),
                    "kind": r.get("kind") or "",
                    "speaker": r.get("speaker") or "",
                    "ts": r.get("ts") or "",
                    "state": r.get("state") or "",
                    "memory_class": r.get("memory_class") or "",
                    "cosine_score": max(0.0, 1.0 - dist),
                }
            )
            if len(results) >= top_k:
                break
        return results

    @staticmethod
    def _chunk_filters(session_id: str | None, state: str | None) -> str:
        """Optional AND-filter tail for CHUNK queries (interpolated via _sql)."""
        parts = []
        if session_id:
            parts.append(f"c.session_id = {quote(session_id)}")
        if state:
            parts.append(f"c.state = {quote(state)}")
        return (" AND " + " AND ".join(parts)) if parts else ""

    async def query_entity_chunks(
        self,
        entity_names: list[str],
        session_id: str | None = None,
        top_k: int = 5,
        state: str | None = None,
    ) -> list[dict[str, Any]]:
        """CHUNKs linked (via MENTIONS) to any of the given entity names.

        Short tokens (< 5 chars) use exact match to avoid substring false
        positives ("war" matching "edward"); longer tokens use CONTAINS.
        """
        if not entity_names:
            return []
        safe = [sanitize(n) for n in entity_names if n]
        if not safe:
            return []
        where_entity = " OR ".join(
            f"e.name_norm = '{n.casefold()}'"
            if len(n) < 5
            else f"e.name_norm CONTAINS '{n.casefold()}'"
            for n in safe
        )
        filters = self._chunk_filters(session_id, state)
        rows = await self._qry(
            f"MATCH (c:CHUNK)-[:MENTIONS]->(e:ENTITY) "
            f"WHERE ({where_entity}){filters} "
            f"RETURN DISTINCT c.chunk_id AS chunk_id, c.text AS text LIMIT $k",
            k=top_k,
        )
        return [{"chunk_id": r["chunk_id"], "text": r.get("text", "")} for r in rows]

    async def full_text_search_chunks(
        self,
        terms: list[str],
        session_id: str | None = None,
        top_k: int = 5,
        require_all: bool = False,
        state: str | None = None,
    ) -> list[dict[str, Any]]:
        """CHUNKs whose text contains the given terms (CONTAINS scan for now;
        a FULL_TEXT index replaces this in Phase 4).

        require_all=True uses AND (all terms must appear); False uses OR.
        """
        if not terms:
            return []
        safe = [sanitize(t) for t in terms if t]
        if not safe:
            return []
        op = " AND " if require_all else " OR "
        text_where = op.join(f"toLower(c.text) CONTAINS '{t.lower()}'" for t in safe)
        filters = self._chunk_filters(session_id, state)
        rows = await self._qry(
            f"MATCH (c:CHUNK) WHERE ({text_where}){filters} "
            f"RETURN c.chunk_id AS chunk_id, c.text AS text LIMIT $k",
            k=top_k,
        )
        return [{"chunk_id": r["chunk_id"], "text": r.get("text", "")} for r in rows]

    async def query_temporal_chunks(
        self,
        iso_dates: list[str],
        session_id: str | None = None,
        top_k: int = 10,
        state: str | None = None,
    ) -> list[dict[str, Any]]:
        """CHUNKs linked to TEMPORAL nodes matching the given ISO dates.

        Exact match per date; 4-digit years also prefix-match ``YYYY-``.
        """
        dates = [sanitize(str(d).strip()) for d in iso_dates if str(d).strip()]
        if not dates:
            return []
        conds: list[str] = []
        for d in dates:
            conds.append(f"d.date = '{d}'")
            if len(d) == 4 and d.isdigit():
                conds.append(f"d.date STARTS WITH '{d}-'")
        where_iso = " OR ".join(conds)
        filters = self._chunk_filters(session_id, state)
        rows = await self._qry(
            f"MATCH (c:CHUNK)-[:MENTIONS_DATE]->(d:TEMPORAL) "
            f"WHERE ({where_iso}){filters} "
            f"RETURN DISTINCT c.chunk_id AS chunk_id, c.text AS text, d.date AS date "
            f"LIMIT $k",
            k=top_k,
        )
        return [
            {"chunk_id": r["chunk_id"], "text": r.get("text", ""), "date": r.get("date", "")}
            for r in rows
        ]

    async def get_chunks_by_ids(self, chunk_ids: list[str]) -> dict[str, dict[str, Any]]:
        """Fetch chunks by id. Returns {chunk_id: {text, session_id, ...}}."""
        ids = [c for c in chunk_ids if c]
        if not ids:
            return {}
        rows = await self._qry(
            f"MATCH (c:CHUNK) WHERE c.chunk_id IN {quoted_list(ids)} "
            f"RETURN c.chunk_id AS chunk_id, c.session_id AS session_id, c.text AS text, "
            f"c.seq AS seq, c.kind AS kind, c.speaker AS speaker, c.ts AS ts, "
            f"c.state AS state, c.memory_class AS memory_class"
        )
        result: dict[str, dict[str, Any]] = {}
        for r in rows:
            cid = r.get("chunk_id", "")
            if cid and cid not in result:
                result[cid] = {
                    "text": r.get("text", ""),
                    "session_id": r.get("session_id") or "",
                    "seq": r.get("seq"),
                    "kind": r.get("kind") or "",
                    "speaker": r.get("speaker") or "",
                    "ts": r.get("ts") or "",
                    "state": r.get("state") or "",
                    "memory_class": r.get("memory_class") or "",
                }
        return result

    # ------------------------------------------------------------------
    # Read API — topics / frames / entity graph
    # ------------------------------------------------------------------

    async def search_topics(self, embedding: list[float], top_k: int = 5) -> list[dict[str, Any]]:
        """ANN search over TOPIC.embedding."""
        if not embedding:
            return []
        sql = _vec_search_sql("TOPIC[embedding]", embedding, top_k * 2)
        try:
            rows = await self._c.query(self._db, sql, language="sql")
        except Exception as exc:
            logger.debug("search_topics failed (non-fatal): %s", exc)
            return []
        rows = self._dedup_rows(rows)
        results: list[dict[str, Any]] = []
        for r in rows:
            dist = float(r.get("distance") or 1.0)
            results.append(
                {
                    "topic_id": r.get("topic_id", ""),
                    "summary": r.get("summary", ""),
                    "modularity": r.get("modularity"),
                    "confidence": max(0.0, 1.0 - dist),
                }
            )
            if len(results) >= top_k:
                break
        return results

    async def query_frame_chunks(
        self,
        frame_types: list[str],
        session_id: str | None = None,
        state: str | None = None,
    ) -> list[dict[str, Any]]:
        """FRAME rows (with evidence chunk) whose frame_type matches, via EVOKED_BY."""
        types = [t for t in frame_types if t]
        if not types:
            return []
        where = f"n.frame_type IN {quoted_list(types)}"
        if session_id:
            where += f" AND n.session_id = {quote(session_id)}"
        if state:
            where += f" AND c.state = {quote(state)}"
        rows = await self._qry(
            f"MATCH (n:FRAME)-[:EVOKED_BY]->(c:CHUNK) WHERE {where} "
            f"RETURN c.chunk_id AS chunk_id, n.frame_type AS frame_type, "
            f"n.speaker AS speaker, n.fillers AS fillers"
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

    async def entity_relation_graph(
        self, session_id: str | None = None
    ) -> list[dict[str, Any]]:
        """Semantic ENTITY->ENTITY relation rows, excluding structural edges.

        Returns rows of {src, type, dst, weight, is_current} for PPR/pagerank.
        With ``session_id`` set, rows are restricted to entities mentioned by
        that session's chunks.
        """
        rows = await self._qry(
            "MATCH (a:ENTITY)-[r]->(b:ENTITY) "
            "WHERE a.name IS NOT NULL AND b.name IS NOT NULL AND a.name <> b.name "
            f"AND NOT type(r) IN {_structural_types_literal()} "
            "AND (r.is_current = true OR r.is_current IS NULL) "
            "RETURN a.name AS src, type(r) AS type, b.name AS dst, "
            "coalesce(r.confidence, 1.0) AS weight, r.is_current AS is_current"
        )
        result = [
            {
                "src": r["src"],
                "type": r.get("type", ""),
                "dst": r["dst"],
                "weight": float(r.get("weight") or 1.0),
                "is_current": r.get("is_current") is not False,
            }
            for r in rows
            if r.get("src") and r.get("dst")
        ]
        if not session_id:
            return result
        # Restrict both endpoints to entities mentioned by this session's chunks.
        ent_rows = await self._qry(
            "MATCH (c:CHUNK)-[:MENTIONS]->(e:ENTITY) WHERE c.session_id = $sid "
            "RETURN DISTINCT e.name AS name",
            sid=session_id,
        )
        session_names = {r["name"] for r in ent_rows if r.get("name")}
        return [r for r in result if r["src"] in session_names and r["dst"] in session_names]

    # ------------------------------------------------------------------
    # Graph analytics
    # ------------------------------------------------------------------

    async def _ensure_analytical_view(self) -> None:
        """Create + rebuild the Graph OLAP view so ``CALL algo.*`` uses the CSR engine.

        Ported from the retired ``ltm_store._ensure_analytical_view``. Without a
        covering Graph Analytical View, ArcadeDB runs graph algorithms through
        the OLTP traversal path — orders of magnitude slower, enough to exceed
        the client read timeout and look hung. Requires ArcadeDB >=26.3.2. A
        full-graph view (no VERTEX/EDGE TYPES clause) covers every type the
        native algos touch. ``UPDATE MODE OFF`` marks the view stale on each
        commit, so it is rebuilt against current data on every call — cheap
        relative to a wrong/stale result, but not free, so callers running
        several algos in one pass (see ``memory.flush_memory``) should call
        this once and pass ``skip_view_ensure=True`` to the rest. Raises on
        failure so callers can decide how to react (pagerank falls back to
        numpy; node2vec/louvain soft-fail to 0).
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
        await self._c.command(
            self._db, f"REBUILD GRAPH ANALYTICAL VIEW {_OLAP_VIEW}", language="sql"
        )

    async def _entity_degree_map(self) -> dict[str, int]:
        """Total (in+out) semantic-relation degree per ENTITY, keyed by ``name_norm``.

        Structural edge types are excluded (same policy as ``entity_relation_graph``
        / the numpy pagerank). Two directed queries avoid relying on ArcadeDB's
        undirected Cypher pattern support.
        """
        struct = _structural_types_literal()
        out_rows = await self._qry(
            f"MATCH (a:ENTITY)-[r]->(b:ENTITY) WHERE NOT type(r) IN {struct} "
            "RETURN a.name_norm AS nn, count(r) AS deg"
        )
        in_rows = await self._qry(
            f"MATCH (a:ENTITY)<-[r]-(b:ENTITY) WHERE NOT type(r) IN {struct} "
            "RETURN a.name_norm AS nn, count(r) AS deg"
        )
        degree: dict[str, int] = {}
        for r in out_rows + in_rows:
            nn = r.get("nn")
            if nn:
                degree[nn] = degree.get(nn, 0) + int(r.get("deg") or 0)
        return degree

    async def compute_and_persist_pagerank(self, skip_view_ensure: bool = False) -> int:
        """Compute PageRank over the entity-relation graph and persist ``pagerank``.

        Tries ArcadeDB's native ``CALL algo.pageRank()`` (scoped to ENTITY rows
        via ``node.name_norm IS NOT NULL`` — ``name_norm`` is ENTITY-only in
        this schema, and is the scoping filter that actually works: ArcadeDB's
        Cypher does not honour a ``node:ENTITY`` label predicate inside a
        ``CALL ... YIELD ... WHERE`` clause, it silently matches zero rows)
        first. **On ANY failure** (view creation, unsupported algo, timeout,
        or empty graph) this falls back to the in-process numpy power-iteration
        implementation (``_pagerank_numpy_fallback``) so a flush never comes up
        empty.

        Args:
            skip_view_ensure: Skip rebuilding the analytical view (the caller
                already did so in this pass — see ``_ensure_analytical_view``).

        Returns:
            Number of ENTITY nodes updated with a pagerank score.
        """
        try:
            if not skip_view_ensure:
                await self._ensure_analytical_view()
            # ArcadeDB's algo.pageRank yields the score column as `score`, not
            # `rank` — YIELDing `rank` binds an unknown field that comes back
            # null for every row, which the null-filter below then discards,
            # silently forcing the numpy fallback. Alias `score AS rank` so the
            # rest of this method keeps reading `rank`.
            rows = await self._qry(
                "CALL algo.pageRank() YIELD node, score "
                "WHERE node.name_norm IS NOT NULL "
                "RETURN node.name_norm AS nn, score AS rank"
            )
            # A row with a null rank is not a usable result — treat "every rank
            # is null" the same as "no rows" so we don't persist meaningless
            # zeros over a graph that never actually got ranked.
            rows = [r for r in rows if r.get("nn") and r.get("rank") is not None]
            if not rows:
                raise RuntimeError("native algo.pageRank returned no usable (non-null) ranks")
        except Exception as exc:
            logger.info(
                "compute_and_persist_pagerank: native algo.pageRank unavailable (%s); "
                "falling back to numpy",
                exc,
            )
            return await self._pagerank_numpy_fallback()

        updated = 0
        for r in rows:
            try:
                await self._cmd(
                    "MATCH (e:ENTITY {name_norm: $nn}) SET e.pagerank = $pr",
                    nn=r["nn"],
                    pr=float(r.get("rank") or 0.0),
                )
                updated += 1
            except Exception as exc:
                logger.warning(
                    "compute_and_persist_pagerank: entity %r failed: %s", r["nn"], exc
                )
        logger.info(
            "compute_and_persist_pagerank: native algo.pageRank updated %d ENTITY nodes", updated
        )
        return updated

    async def compute_graph_embeddings(
        self, dimensions: int = 32, skip_view_ensure: bool = False
    ) -> int:
        """Compute node2vec embeddings over the entity-relation graph -> ``ENTITY.graph_embedding``.

        New capability (no numpy equivalent — this is not a fallback-guarded
        replacement of an existing feature). Soft-fails to 0 on any error or
        an empty result; entities with fewer than 3 semantic-relation edges are
        skipped (too sparse for a meaningful walk-based embedding).

        Args:
            dimensions: node2vec embedding dimension.
            skip_view_ensure: Skip rebuilding the analytical view (see
                ``_ensure_analytical_view``).

        Returns:
            Number of ENTITY nodes updated with a ``graph_embedding``.
        """
        try:
            if not skip_view_ensure:
                await self._ensure_analytical_view()
            # Scoped via node.name_norm IS NOT NULL, not node:ENTITY — see
            # compute_and_persist_pagerank's docstring for why the label
            # predicate doesn't work in ArcadeDB's Cypher.
            rows = await self._qry(
                f"CALL algo.node2vec({{walkLength:20,walks:20,dimensions:{int(dimensions)}}}) "
                "YIELD node, embedding "
                "WHERE node.name_norm IS NOT NULL "
                "RETURN node.name_norm AS nn, embedding"
            )
        except Exception as exc:
            logger.info("compute_graph_embeddings: native node2vec unavailable (%s); skipping", exc)
            return 0

        rows = [r for r in rows if r.get("nn") and r.get("embedding")]
        if not rows:
            logger.info("compute_graph_embeddings: no embeddings returned; skipping")
            return 0

        # A genuinely empty degree map (no entity has any semantic relation)
        # still means every entity is sparse — only a *failed* lookup disables
        # the filter, tracked separately so it isn't mistaken for "empty".
        degree_available = True
        try:
            degree = await self._entity_degree_map()
        except Exception as exc:
            logger.warning(
                "compute_graph_embeddings: degree lookup failed (%s); skipping sparsity filter",
                exc,
            )
            degree = {}
            degree_available = False

        updated = 0
        for r in rows:
            nn = r["nn"]
            if degree_available and degree.get(nn, 0) < 3:
                continue
            try:
                # Inline literal, not a $param — verified live that ArcadeDB's
                # Cypher SET double-wraps a LIST-typed param (n.prop becomes
                # [[...]] instead of [...]), same reason _set_embedding uses
                # an inline literal for CHUNK/ENTITY/TOPIC.embedding.
                emb_literal = _emb_sql([float(x) for x in r["embedding"]])
                await self._cmd(
                    f"MATCH (e:ENTITY {{name_norm: $nn}}) SET e.graph_embedding = {emb_literal}",
                    nn=nn,
                )
                updated += 1
            except Exception as exc:
                logger.warning("compute_graph_embeddings: entity %r failed: %s", nn, exc)
        logger.info("compute_graph_embeddings: wrote %d ENTITY graph_embedding(s)", updated)
        return updated

    async def compute_communities(self, skip_view_ensure: bool = False) -> int:
        """Compute community_id via ArcadeDB's native Louvain -> ``ENTITY.community_id``.

        Replaces the retired igraph/leidenalg community step; runs in both
        llm_free and llm_assisted modes (plan §1: ``graph_embedding``/community
        are both-modes signals now). New capability — soft-fails to 0 on any
        error or an empty result; no fallback required.

        Args:
            skip_view_ensure: Skip rebuilding the analytical view (see
                ``_ensure_analytical_view``).

        Returns:
            Number of ENTITY nodes updated with a ``community_id``.
        """
        try:
            if not skip_view_ensure:
                await self._ensure_analytical_view()
            rows = await self._qry(
                "CALL algo.louvain() YIELD node, communityId "
                "WHERE node.name_norm IS NOT NULL "
                "RETURN node.name_norm AS nn, communityId AS cid"
            )
        except Exception as exc:
            logger.info("compute_communities: native louvain unavailable (%s); skipping", exc)
            return 0

        rows = [r for r in rows if r.get("nn") and r.get("cid") is not None]
        if not rows:
            logger.info("compute_communities: no rows returned; skipping")
            return 0

        updated = 0
        for r in rows:
            try:
                await self._cmd(
                    "MATCH (e:ENTITY {name_norm: $nn}) SET e.community_id = $cid",
                    nn=r["nn"],
                    cid=int(r["cid"]),
                )
                updated += 1
            except Exception as exc:
                logger.warning("compute_communities: entity %r failed: %s", r["nn"], exc)
        logger.info("compute_communities: wrote %d ENTITY community_id(s)", updated)
        return updated

    async def _semantic_rel_types_literal(self) -> str:
        """Cypher list literal of distinct ENTITY->ENTITY semantic relation types.

        Structural types (MENTIONS/PART_OF/…) are excluded — the same policy as
        the pagerank path — so the community-quality algos score the meaningful
        relation graph, not CHUNK->ENTITY provenance or ENTITY->COMMUNITY links.
        Returns ``[]`` when the graph has no semantic entity-entity edges.
        """
        struct = _structural_types_literal()
        rows = await self._qry(
            f"MATCH (a:ENTITY)-[r]->(b:ENTITY) WHERE NOT type(r) IN {struct} "
            "RETURN DISTINCT type(r) AS t"
        )
        return quoted_list(sorted({r["t"] for r in rows if r.get("t")}))

    async def compute_community_quality(self, skip_view_ensure: bool = False) -> dict[str, Any]:
        """Score community structure via native modularityScore + conductance.

        Runs after ``compute_communities`` (reads the ``community_id`` it wrote).
        Both algos are scoped to the semantic entity-entity relation types (see
        ``_semantic_rel_types_literal``) so CHUNK->ENTITY / ENTITY->COMMUNITY
        edges don't pollute the metrics. Persists a **COMMUNITY vertex** per
        ``community_id`` (chosen over per-ENTITY props: it makes community-level
        quality directly queryable and enables the quality-adjusted hop signal in
        ``chunk_community_weights``), each carrying its conductance / edge counts
        / size and the shared global modularity, plus an ``IN_COMMUNITY`` edge
        ENTITY->COMMUNITY. Soft-fails to a zero result on any algo error — never
        raises out of flush.

        Args:
            skip_view_ensure: Skip rebuilding the analytical view (the caller
                already did so in this pass).

        Returns:
            ``{"modularity": float, "communities": int, "communities_scored": int}``.
        """
        empty = {"modularity": 0.0, "communities": 0, "communities_scored": 0}
        try:
            if not skip_view_ensure:
                await self._ensure_analytical_view()
            rel_types = await self._semantic_rel_types_literal()
            if rel_types == "[]":
                logger.info("compute_community_quality: no semantic edges; skipping")
                return empty
            mod_rows = await self._qry(
                f"CALL algo.modularityScore('community_id', {rel_types}) "
                "YIELD modularity, communities, edgeCount "
                "RETURN modularity, communities, edgeCount"
            )
            cond_rows = await self._qry(
                f"CALL algo.conductance('community_id', {rel_types}) "
                "YIELD community, conductance, internalEdges, boundaryEdges, nodeCount "
                "RETURN community, conductance, internalEdges, boundaryEdges, nodeCount"
            )
        except Exception as exc:
            logger.info("compute_community_quality: native algo unavailable (%s); skipping", exc)
            return empty

        if not mod_rows:
            logger.info("compute_community_quality: modularityScore returned no rows; skipping")
            return empty
        modularity = float(mod_rows[0].get("modularity") or 0.0)
        communities = int(mod_rows[0].get("communities") or 0)

        # Per-community entity counts (the persisted ``size``; conductance's
        # nodeCount can differ by the isolated COMMUNITY vertex, so we keep both).
        size_rows = await self._qry(
            "MATCH (e:ENTITY) WHERE e.community_id IS NOT NULL "
            "RETURN e.community_id AS cid, count(e) AS size"
        )
        sizes = {r["cid"]: int(r.get("size") or 0) for r in size_rows if r.get("cid") is not None}

        scored = 0
        for r in cond_rows:
            cid = r.get("community")
            if cid is None:
                continue
            try:
                await self._cmd(
                    "MERGE (k:COMMUNITY {community_id: $cid}) "
                    "SET k.conductance = $cond, k.internal_edges = $ie, "
                    "    k.boundary_edges = $be, k.node_count = $nc, "
                    "    k.modularity = $mod, k.size = $size",
                    cid=int(cid),
                    cond=float(r.get("conductance") or 0.0),
                    ie=int(r.get("internalEdges") or 0),
                    be=int(r.get("boundaryEdges") or 0),
                    nc=int(r.get("nodeCount") or 0),
                    mod=modularity,
                    size=sizes.get(int(cid), 0),
                )
                scored += 1
            except Exception as exc:
                logger.warning("compute_community_quality: community %r failed: %s", cid, exc)

        # Link entities to their community vertex (only communities that got a
        # COMMUNITY vertex will MERGE — arbitrary Louvain ids are reused, so this
        # is self-healing across flushes).
        try:
            await self._cmd(
                "MATCH (e:ENTITY) WHERE e.community_id IS NOT NULL "
                "MATCH (k:COMMUNITY {community_id: e.community_id}) "
                "MERGE (e)-[:IN_COMMUNITY]->(k)"
            )
        except Exception as exc:
            logger.warning("compute_community_quality: IN_COMMUNITY link failed: %s", exc)

        logger.info(
            "compute_community_quality: modularity=%.4f, scored %d/%d communities",
            modularity, scored, communities,
        )
        return {"modularity": modularity, "communities": communities, "communities_scored": scored}

    async def chunk_community_weights(self, chunk_ids: list[str]) -> dict[str, float]:
        """Quality-adjusted community boost weight per chunk, from its entities.

        For each chunk, takes the majority (most-mentioned) non-null
        ``community_id`` among the entities it mentions and returns a weight that
        blends the community's size with its **conductance** (a hop-reasoning
        signal): ``size * (1 - conductance)``. A tight, well-separated community
        (low conductance) is a trustworthy hub for intra-community expansion, so
        chunks anchored in it get a larger multi-hop boost; a loose community
        (high conductance) is down-weighted. Communities without a COMMUNITY
        quality vertex fall back to raw size, so this never regresses when the
        quality algos didn't run. Bounded downstream by
        ``ranking.rrf.community_boost``; chunks with no community-tagged entities
        are absent (callers default to 1.0, a no-op). Feeds
        ``DETRetriever._community_boost`` / ``AgenticRetriever
        ._apply_community_weights`` in both profiles.
        """
        ids = [c for c in chunk_ids if c]
        if not ids:
            return {}
        try:
            rows = await self._qry(
                f"MATCH (c:CHUNK)-[:MENTIONS]->(e:ENTITY) "
                f"WHERE c.chunk_id IN {quoted_list(ids)} AND e.community_id IS NOT NULL "
                f"RETURN c.chunk_id AS chunk_id, e.community_id AS cid, count(e) AS cnt "
                f"ORDER BY cnt DESC"
            )
            size_rows = await self._qry(
                "MATCH (e:ENTITY) WHERE e.community_id IS NOT NULL "
                "RETURN e.community_id AS cid, count(e) AS size"
            )
            quality_rows = await self._qry(
                "MATCH (k:COMMUNITY) WHERE k.community_id IS NOT NULL "
                "RETURN k.community_id AS cid, k.conductance AS conductance"
            )
        except Exception as exc:
            logger.debug("chunk_community_weights failed (non-fatal): %s", exc)
            return {}
        sizes = {r["cid"]: int(r.get("size") or 1) for r in size_rows if r.get("cid") is not None}
        conductance = {
            r["cid"]: float(r["conductance"])
            for r in quality_rows
            if r.get("cid") is not None and r.get("conductance") is not None
        }
        # Rows are globally sorted by cnt desc, so the first row seen for a given
        # chunk is that chunk's majority community (highest-cnt row wins).
        majority: dict[str, Any] = {}
        for r in rows:
            cid = r.get("chunk_id")
            if cid and cid not in majority:
                majority[cid] = r.get("cid")

        def _weight(comm: Any) -> float:
            size = float(sizes.get(comm, 1))
            cond = conductance.get(comm)
            if cond is None:
                return size  # no quality vertex — raw size (no regression)
            return size * max(0.0, 1.0 - cond)

        return {cid: _weight(comm) for cid, comm in majority.items()}

    async def similar_entities_by_embedding(
        self, name: str, top_k: int = 5
    ) -> list[dict[str, Any]]:
        """Entities most similar to *name* by cosine over ``graph_embedding``.

        In-Python cosine over every ENTITY.graph_embedding — entity counts are
        small (tens-hundreds per namespace), so no vector index is needed.
        Returns rows of ``{name, name_norm, similarity}`` excluding the seed
        itself, empty when the seed or the corpus lacks embeddings.
        """
        try:
            import numpy as np
        except ImportError:
            return []
        seed_nn = _name_norm(name)
        rows = await self._qry(
            "MATCH (e:ENTITY) WHERE e.graph_embedding IS NOT NULL "
            "RETURN e.name_norm AS nn, e.name AS name, e.graph_embedding AS emb"
        )
        seed_emb: list[float] | None = None
        candidates: list[tuple[str, str, list[float]]] = []
        for r in rows:
            emb = r.get("emb")
            if not emb:
                continue
            if r.get("nn") == seed_nn:
                seed_emb = emb
            else:
                candidates.append((r.get("nn"), r.get("name") or r.get("nn"), emb))
        if seed_emb is None or not candidates:
            return []
        seed = np.asarray(seed_emb, dtype=float)
        seed = seed / (np.linalg.norm(seed) + 1e-9)
        scored: list[tuple[float, str, str]] = []
        for nn, disp_name, emb in candidates:
            v = np.asarray(emb, dtype=float)
            v = v / (np.linalg.norm(v) + 1e-9)
            scored.append((float(seed @ v), nn, disp_name))
        scored.sort(key=lambda x: -x[0])
        return [
            {"name_norm": nn, "name": disp_name, "similarity": sim}
            for sim, nn, disp_name in scored[:top_k]
        ]

    async def _pagerank_numpy_fallback(self) -> int:
        """In-process PageRank over the entity-relation graph (numpy power iteration).

        The pre-WS4 implementation, kept as the safety net for
        ``compute_and_persist_pagerank`` when ArcadeDB's native
        ``CALL algo.pageRank()`` is unavailable or fails. Persists ``pagerank``
        on ENTITY nodes. Returns the number of ENTITY nodes updated. Soft-fails
        to 0 when numpy is unavailable or the graph is empty.
        """
        try:
            import numpy as np
        except ImportError:
            logger.info("_pagerank_numpy_fallback: numpy unavailable, skipping")
            return 0

        try:
            entity_rows = await self._qry(
                "MATCH (e:ENTITY) WHERE e.name_norm IS NOT NULL RETURN e.name_norm AS nn"
            )
            edge_rows = await self._qry(
                "MATCH (a:ENTITY)-[r]->(b:ENTITY) "
                "WHERE (r.is_current = true OR r.is_current IS NULL) "
                f"AND NOT type(r) IN {_structural_types_literal()} "
                "RETURN DISTINCT a.name_norm AS src, b.name_norm AS dst, "
                "coalesce(r.confidence, 1.0) AS w"
            )
        except Exception as exc:
            logger.warning("_pagerank_numpy_fallback: graph query failed: %s", exc)
            return 0

        names = [r["nn"] for r in entity_rows if r.get("nn")]
        if not names:
            return 0
        name_idx = {n: i for i, n in enumerate(names)}
        n = len(names)

        adj = np.zeros((n, n), dtype=float)
        for row in edge_rows:
            src, dst = row.get("src"), row.get("dst")
            w = float(row.get("w") or 1.0)
            si, ti = name_idx.get(src), name_idx.get(dst)  # type: ignore[arg-type]
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
        new_rank = rank
        for _ in range(50):
            dangling_sum = rank[dangling_nodes].sum() / n if len(dangling_nodes) else 0.0
            new_rank = (1 - damping) / n + damping * (adj @ rank + dangling_sum)
            if np.linalg.norm(new_rank - rank, 1) < 1e-6:
                break
            rank = new_rank
        rank = new_rank

        updated = 0
        for i, nn in enumerate(names):
            try:
                await self._cmd(
                    "MATCH (e:ENTITY {name_norm: $nn}) SET e.pagerank = $pr",
                    nn=nn,
                    pr=float(rank[i]),
                )
                updated += 1
            except Exception as exc:
                logger.warning("_pagerank_numpy_fallback: entity %r failed: %s", nn, exc)
        logger.info("_pagerank_numpy_fallback: updated %d ENTITY nodes", updated)
        return updated

    # ------------------------------------------------------------------
    # Entity resolution primitives (consolidation / flush path)
    # ------------------------------------------------------------------

    async def list_entities(self) -> list[dict[str, Any]]:
        """List every ENTITY with its incoming-MENTIONS count (namespace-wide).

        Returns rows of ``{name_norm, name, type, mention_count}``. Entities are
        namespace-global, so this is not session-scoped — it feeds the layered
        entity resolver, which decides drops/merges over the whole graph.
        """
        rows = await self._qry(
            "MATCH (e:ENTITY) WHERE e.name_norm IS NOT NULL "
            "OPTIONAL MATCH (c:CHUNK)-[:MENTIONS]->(e) "
            "RETURN e.name_norm AS name_norm, e.name AS name, e.type AS type, "
            "count(c) AS mention_count"
        )
        return [
            {
                "name_norm": r["name_norm"],
                "name": r.get("name") or r["name_norm"],
                "type": r.get("type") or "",
                "mention_count": int(r.get("mention_count") or 0),
            }
            for r in rows
            if r.get("name_norm")
        ]

    async def distinct_speakers(self) -> list[str]:
        """Casefolded distinct CHUNK.speaker values (the known session speakers)."""
        rows = await self._qry(
            "MATCH (c:CHUNK) WHERE c.speaker IS NOT NULL AND c.speaker <> '' "
            "RETURN DISTINCT c.speaker AS speaker"
        )
        return [str(r["speaker"]).casefold() for r in rows if r.get("speaker")]

    async def drop_entities(self, name_norms: list[str]) -> int:
        """DETACH DELETE the given ENTITY nodes (removes MENTIONS/RELATION edges).

        Used by the junk/hub filter. Returns the number of nodes deleted.
        """
        norms = [n for n in name_norms if n]
        if not norms:
            return 0
        rows = await self._qry(
            f"MATCH (e:ENTITY) WHERE e.name_norm IN {quoted_list(norms)} "
            "RETURN count(e) AS n"
        )
        await self._cmd(
            f"MATCH (e:ENTITY) WHERE e.name_norm IN {quoted_list(norms)} DETACH DELETE e"
        )
        return int(rows[0].get("n") or 0) if rows else 0

    async def _repoint_relation(
        self, src_name: str, dst_name: str, row: dict[str, Any]
    ) -> None:
        """Re-create one loser relation edge on the winner via write_relation.

        Reuses the audited write_relation (evidence-append dedup, max confidence,
        edge-type declaration, single-valued supersede). Each evidence chunk is
        replayed so the winner edge accumulates the loser's full provenance.
        """
        predicate = row.get("predicate") or row.get("t") or ""
        if not predicate or not src_name or not dst_name:
            return
        conf = float(row.get("confidence") or 0.5)
        fact = row.get("fact") or ""
        valid_at = row.get("valid_at") or None
        source = row.get("source") or ""
        evidence = [e for e in (row.get("evidence") or []) if e]
        if not evidence:
            await self.write_relation(
                src_name, predicate, dst_name, fact=fact, confidence=conf,
                valid_at=valid_at, source=source,
            )
            return
        for cid in evidence:
            await self.write_relation(
                src_name, predicate, dst_name, fact=fact, chunk_id=cid,
                confidence=conf, valid_at=valid_at, source=source,
            )

    async def merge_entities(self, winner_norm: str, loser_norm: str) -> bool:
        """Merge the loser ENTITY into the winner. Idempotent and self-safe.

        - Re-points loser MENTIONS (CHUNK->loser) onto the winner (MERGE dedups;
          max confidence kept).
        - Re-points loser semantic RELATION edges (both directions) onto the
          winner, skipping self-loops, via write_relation (merges evidence /
          keeps max confidence).
        - Appends the loser's name + aliases to the winner's aliases (deduped).
        - DETACH DELETE the loser.

        Returns True when a merge happened, False when it was a no-op (loser or
        winner missing, or same node).
        """
        if not winner_norm or not loser_norm or winner_norm == loser_norm:
            return False
        loser = await self._qry(
            "MATCH (e:ENTITY {name_norm: $nn}) RETURN e.name AS name, e.aliases AS aliases LIMIT 1",
            nn=loser_norm,
        )
        if not loser:
            return False  # already gone — idempotent
        winner = await self._qry(
            "MATCH (e:ENTITY {name_norm: $nn}) RETURN e.name AS name, e.aliases AS aliases LIMIT 1",
            nn=winner_norm,
        )
        if not winner:
            return False  # winner must exist
        loser_name = loser[0].get("name") or loser_norm
        loser_aliases = list(loser[0].get("aliases") or [])
        winner_name = winner[0].get("name") or winner_norm
        aliases = list(winner[0].get("aliases") or [])

        # 1. Re-point MENTIONS onto the winner (old loser edges vanish on delete).
        await self._cmd(
            "MATCH (c:CHUNK)-[m:MENTIONS]->(:ENTITY {name_norm: $loser}) "
            "MATCH (w:ENTITY {name_norm: $winner}) "
            "MERGE (c)-[nm:MENTIONS]->(w) "
            "SET nm.confidence = CASE "
            "    WHEN coalesce(m.confidence, 0) > coalesce(nm.confidence, 0) "
            "    THEN m.confidence ELSE nm.confidence END",
            loser=loser_norm,
            winner=winner_norm,
        )

        # 2. Re-point semantic RELATION edges (both directions), skip self-loops.
        struct = _structural_types_literal()
        out_rows = await self._qry(
            "MATCH (:ENTITY {name_norm: $loser})-[r]->(b:ENTITY) "
            f"WHERE NOT type(r) IN {struct} "
            "RETURN type(r) AS t, b.name AS dst, b.name_norm AS dst_nn, "
            "r.predicate AS predicate, r.fact AS fact, r.evidence AS evidence, "
            "r.confidence AS confidence, r.valid_at AS valid_at, r.source AS source",
            loser=loser_norm,
        )
        in_rows = await self._qry(
            "MATCH (a:ENTITY)-[r]->(:ENTITY {name_norm: $loser}) "
            f"WHERE NOT type(r) IN {struct} "
            "RETURN type(r) AS t, a.name AS src, a.name_norm AS src_nn, "
            "r.predicate AS predicate, r.fact AS fact, r.evidence AS evidence, "
            "r.confidence AS confidence, r.valid_at AS valid_at, r.source AS source",
            loser=loser_norm,
        )
        for row in out_rows:
            if row.get("dst_nn") == winner_norm:
                continue
            await self._repoint_relation(winner_name, row.get("dst") or row.get("dst_nn"), row)
        for row in in_rows:
            if row.get("src_nn") == winner_norm:
                continue
            await self._repoint_relation(row.get("src") or row.get("src_nn"), winner_name, row)

        # 3. Fold the loser's surface + aliases into the winner's aliases.
        for a in [loser_name, *loser_aliases]:
            if a and a != winner_name and a not in aliases:
                aliases.append(a)
        await self._cmd(
            "MATCH (w:ENTITY {name_norm: $winner}) SET w.aliases = $aliases",
            winner=winner_norm,
            aliases=aliases,
        )

        # 4. Remove the loser (and its now-orphaned original edges).
        await self._cmd(
            "MATCH (lo:ENTITY {name_norm: $loser}) DETACH DELETE lo", loser=loser_norm
        )
        logger.info("merge_entities: %s -> %s", loser_norm, winner_norm)
        return True

    async def consolidate_session(self, session_id: str) -> int:
        """Flip lifecycle ``state`` raw -> consolidated in place (Phase-1 shim).

        This is the thin structural consolidation for Phase 1 — NOT the full
        Phase-3 pipeline (entity merge / relation canonicalize / topics). It
        flips the session's SESSION + CHUNK nodes, and every ``raw`` ENTITY and
        relation edge (entities/relations are namespace-global, so any still-raw
        artifact is promoted). Nothing is deleted; data stays queryable.

        Returns the number of CHUNK nodes flipped for this session.
        """
        now = datetime.now(UTC).isoformat()
        # Count first (query endpoint rejects a mutating SET...RETURN as
        # non-idempotent), then flip via the command endpoint.
        chunk_rows = await self._qry(
            "MATCH (c:CHUNK {session_id: $sid}) WHERE c.state = 'raw' RETURN count(c) AS n",
            sid=session_id,
        )
        await self._cmd(
            "MATCH (c:CHUNK {session_id: $sid}) WHERE c.state = 'raw' "
            "SET c.state = 'consolidated'",
            sid=session_id,
        )
        await self._cmd(
            "MATCH (s:SESSION {session_id: $sid}) "
            "SET s.state = 'consolidated', s.consolidated_at = $now",
            sid=session_id,
            now=now,
        )
        await self._cmd("MATCH (e:ENTITY) WHERE e.state = 'raw' SET e.state = 'consolidated'")
        await self._cmd(
            "MATCH (:ENTITY)-[r]->(:ENTITY) WHERE r.state = 'raw' SET r.state = 'consolidated'"
        )
        flipped = int(chunk_rows[0].get("n") or 0) if chunk_rows else 0
        logger.info("consolidate_session %s: flipped %d chunks", session_id, flipped)
        return flipped

    # ------------------------------------------------------------------
    # Deletion
    # ------------------------------------------------------------------

    async def delete_session(self, session_id: str) -> None:
        """Delete a session's FRAME/CHUNK/FILE vertices and its SESSION node.

        Namespace-global vertices (ENTITY, TOPIC, TEMPORAL) are untouched —
        they may be evidenced by other sessions.
        """
        for label in ("FRAME", "CHUNK", "FILE"):
            try:
                await self._cmd(
                    f"MATCH (n:{label} {{session_id: $sid}}) DETACH DELETE n",
                    sid=session_id,
                )
            except Exception:
                logger.debug("delete_session: %s delete ignored", label, exc_info=True)
        try:
            await self._cmd(
                "MATCH (n:SESSION {session_id: $sid}) DETACH DELETE n",
                sid=session_id,
            )
        except Exception:
            logger.debug("delete_session: SESSION delete ignored", exc_info=True)

    async def clear_all(self) -> None:
        """Delete every vertex (and, via DETACH, every edge) in this database."""
        for label in _VERTEX_LABELS:
            try:
                await self._cmd(f"MATCH (n:{label}) DETACH DELETE n")
            except Exception:
                logger.debug("clear_all: %s delete ignored", label, exc_info=True)


__all__ = ["GraphStore"]

"""ArcadeDB vector store.

Uses the `stm` database for session-scoped vectors and the `ltm` database
for corpus-level vectors, storing embeddings as LIST properties on vertex types.

ArcadeDB constraints:
- Embedding writes use SQL UPDATE with literal float arrays (Cypher params for LIST
  properties conflict with the LSM_VECTOR index).
- Vector search uses SQL vectorNeighbors() with the 3-arg form (index provides
  the distance function).
- Index creation requires the property to be declared as LIST first.
"""

from __future__ import annotations

import contextlib
import json
import logging
from typing import Any

from graphknows.memory.stores.arcadedb.client import ArcadeDBClient

logger = logging.getLogger(__name__)

_STM_DB = "stm"
_LTM_DB = "ltm"
_VEC_DIM = 1536

# ---------------------------------------------------------------------------
# DDL helpers
# ---------------------------------------------------------------------------

# Embedding properties + indexes added to existing STM types.
# CHUNK and TURN types are created by ArcadeDBSTMStore.ensure_schema().
_STM_EMB_DDL: list[tuple[str, str]] = [
    ("sql", "CREATE PROPERTY CHUNK.embedding IF NOT EXISTS LIST"),
    (
        "sql",
        f"CREATE INDEX IF NOT EXISTS ON CHUNK(embedding) LSM_VECTOR METADATA {{dimensions:{_VEC_DIM},distanceFunction:'COSINE'}}",
    ),
    ("sql", "CREATE PROPERTY CHUNK.memory_class IF NOT EXISTS STRING"),
    ("sql", "CREATE PROPERTY CHUNK.metadata IF NOT EXISTS STRING"),
    ("sql", "CREATE PROPERTY TURN.embedding IF NOT EXISTS LIST"),
    ("sql", "CREATE PROPERTY TURN.metadata IF NOT EXISTS STRING"),
    (
        "sql",
        f"CREATE INDEX IF NOT EXISTS ON TURN(embedding) LSM_VECTOR METADATA {{dimensions:{_VEC_DIM},distanceFunction:'COSINE'}}",
    ),
]

# LTM vertex types that hold vectors (ENTITY and CHUNK managed by ArcadeDBLTMStore;
# TOPIC also lives in ltm).  We only add the embedding property + index here.
_LTM_VEC_TYPES: list[tuple[str, str]] = [
    # (vertex_type, index_name)
    ("CHUNK", "CHUNK[embedding]"),
    ("TURN", "TURN[embedding]"),
    ("ENTITY", "ENTITY[embedding]"),
    ("TOPIC", "TOPIC[embedding]"),
]

_LTM_BASE_DDL: list[tuple[str, str]] = [
    # Minimal vertex types (full schema created by ArcadeDBLTMStore)
    ("sql", "CREATE VERTEX TYPE CHUNK IF NOT EXISTS"),
    ("sql", "CREATE PROPERTY CHUNK.id IF NOT EXISTS STRING"),
    ("sql", "CREATE PROPERTY CHUNK.session_id IF NOT EXISTS STRING"),
    ("sql", "CREATE PROPERTY CHUNK.text IF NOT EXISTS STRING"),
    ("sql", "CREATE PROPERTY CHUNK.heading_path IF NOT EXISTS STRING"),
    ("sql", "CREATE PROPERTY CHUNK.file_id IF NOT EXISTS STRING"),
    ("sql", "CREATE PROPERTY CHUNK.community_id IF NOT EXISTS INTEGER"),
    ("sql", "CREATE PROPERTY CHUNK.community_weight IF NOT EXISTS DOUBLE"),
    ("sql", "CREATE PROPERTY CHUNK.embedding IF NOT EXISTS LIST"),
    ("sql", "CREATE PROPERTY CHUNK.metadata IF NOT EXISTS STRING"),
    ("sql", "CREATE INDEX ON CHUNK(id) UNIQUE"),
    ("sql", "CREATE INDEX ON CHUNK(session_id)"),
    (
        "sql",
        f"CREATE INDEX IF NOT EXISTS ON CHUNK(embedding) LSM_VECTOR METADATA {{dimensions:{_VEC_DIM},distanceFunction:'COSINE'}}",
    ),
    ("sql", "CREATE VERTEX TYPE TURN IF NOT EXISTS"),
    ("sql", "CREATE PROPERTY TURN.id IF NOT EXISTS STRING"),
    ("sql", "CREATE PROPERTY TURN.session_id IF NOT EXISTS STRING"),
    ("sql", "CREATE PROPERTY TURN.dialog_id IF NOT EXISTS STRING"),
    ("sql", "CREATE PROPERTY TURN.turn_index IF NOT EXISTS INTEGER"),
    ("sql", "CREATE PROPERTY TURN.role IF NOT EXISTS STRING"),
    ("sql", "CREATE PROPERTY TURN.text IF NOT EXISTS STRING"),
    ("sql", "CREATE PROPERTY TURN.community_id IF NOT EXISTS INTEGER"),
    ("sql", "CREATE PROPERTY TURN.community_weight IF NOT EXISTS DOUBLE"),
    ("sql", "CREATE PROPERTY TURN.embedding IF NOT EXISTS LIST"),
    ("sql", "CREATE PROPERTY TURN.metadata IF NOT EXISTS STRING"),
    ("sql", "CREATE INDEX ON TURN(id) UNIQUE"),
    ("sql", "CREATE INDEX ON TURN(session_id)"),
    (
        "sql",
        f"CREATE INDEX IF NOT EXISTS ON TURN(embedding) LSM_VECTOR METADATA {{dimensions:{_VEC_DIM},distanceFunction:'COSINE'}}",
    ),
    ("sql", "CREATE VERTEX TYPE ENTITY IF NOT EXISTS"),
    ("sql", "CREATE PROPERTY ENTITY.name IF NOT EXISTS STRING"),
    ("sql", "CREATE PROPERTY ENTITY.label IF NOT EXISTS STRING"),
    ("sql", "CREATE PROPERTY ENTITY.embedding IF NOT EXISTS LIST"),
    ("sql", "CREATE INDEX ON ENTITY(name) UNIQUE"),
    (
        "sql",
        f"CREATE INDEX IF NOT EXISTS ON ENTITY(embedding) LSM_VECTOR METADATA {{dimensions:{_VEC_DIM},distanceFunction:'COSINE'}}",
    ),
    ("sql", "CREATE VERTEX TYPE TOPIC IF NOT EXISTS"),
    ("sql", "CREATE PROPERTY TOPIC.id IF NOT EXISTS STRING"),
    ("sql", "CREATE PROPERTY TOPIC.community_id IF NOT EXISTS INTEGER"),
    ("sql", "CREATE PROPERTY TOPIC.summary IF NOT EXISTS STRING"),
    ("sql", "CREATE PROPERTY TOPIC.embedding IF NOT EXISTS LIST"),
    ("sql", "CREATE PROPERTY TOPIC.entity_names IF NOT EXISTS LIST"),
    ("sql", "CREATE PROPERTY TOPIC.session_ids IF NOT EXISTS LIST"),
    ("sql", "CREATE INDEX ON TOPIC(id) UNIQUE"),
    (
        "sql",
        f"CREATE INDEX IF NOT EXISTS ON TOPIC(embedding) LSM_VECTOR METADATA {{dimensions:{_VEC_DIM},distanceFunction:'COSINE'}}",
    ),
]


def _emb_sql(emb: list[float]) -> str:
    """Format embedding list as ArcadeDB SQL array literal."""
    return "[" + ",".join(str(x) for x in emb) + "]"


def _dump_meta(metadata: dict[str, Any] | None) -> str:
    """Serialise a chunk/turn metadata dict to a JSON string for storage."""
    if not metadata:
        return ""
    try:
        return json.dumps(metadata, default=str)
    except (TypeError, ValueError):
        return ""


def _load_meta(raw: Any) -> dict[str, Any]:
    """Parse a stored metadata JSON string back into a dict (lenient)."""
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str) or not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except (TypeError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _vec_search_sql(index: str, emb: list[float], k: int) -> str:
    """Return SQL for vectorNeighbors ANN search.

    No FROM clause — the index already knows its type.  Adding FROM <type>
    causes ArcadeDB to invoke vectorNeighbors once *per vertex*, returning
    N*k rows and making deduplication order-dependent.
    """
    return f'SELECT expand(vectorNeighbors("{index}", {_emb_sql(emb)}, {k}))'


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------


class ArcadeDBVectorStore:
    """Async ArcadeDB-backed vector store.

    Attributes:
        _c: ArcadeDBClient (shared with STM and LTM stores, not owned here).
    """

    def __init__(self, client: ArcadeDBClient) -> None:
        self._c = client
        self._schema_ready = False

    # ------------------------------------------------------------------
    # Lifecycle (connect/close delegated to client)
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        await self._c.connect()
        if not self._schema_ready:
            await self.ensure_schema()
            self._schema_ready = True

    async def close(self) -> None:
        await self._c.close()

    async def ensure_schema(self) -> None:
        """Add embedding DDL to STM and create LTM vector schema. Idempotent."""
        # STM: extend existing CHUNK/TURN with embedding property + index
        for lang, stmt in _STM_EMB_DDL:
            try:
                await self._c.command(_STM_DB, stmt, language=lang)
            except Exception as exc:
                if "already exists" not in str(exc).lower():
                    logger.debug("STM emb DDL (non-fatal): %s | %s", stmt[:60], exc)

        # LTM: create database + schema
        await self._c.create_database(_LTM_DB)
        for lang, stmt in _LTM_BASE_DDL:
            try:
                await self._c.command(_LTM_DB, stmt, language=lang)
            except Exception as exc:
                if "already exists" not in str(exc).lower():
                    logger.debug("LTM DDL (non-fatal): %s | %s", stmt[:60], exc)

        logger.info("ArcadeDBVectorStore schema ensured (stm + ltm)")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _set_embedding(
        self, db: str, vertex_type: str, entity_id: str, emb: list[float]
    ) -> None:
        """Set embedding on a vertex matched by id.

        The match key is passed as a Cypher parameter (handles quotes/newlines/
        backslashes safely); the embedding stays an inline literal because LIST
        params conflict with the LSM_VECTOR index.
        """
        if not emb:
            return
        stmt = f"MATCH (n:{vertex_type} {{id: $key}}) SET n.embedding = {_emb_sql(emb)}"  # nosec B608
        await self._c.command(db, stmt, params={"key": entity_id})

    async def _set_embedding_by_name(
        self, db: str, vertex_type: str, name: str, emb: list[float]
    ) -> None:
        """Set embedding on a vertex matched by name (see :meth:`_set_embedding`)."""
        if not emb:
            return
        stmt = f"MATCH (n:{vertex_type} {{name: $key}}) SET n.embedding = {_emb_sql(emb)}"  # nosec B608
        await self._c.command(db, stmt, params={"key": name})

    def _dedup_rows(self, rows: list[dict]) -> list[dict]:
        """Deduplicate vectorNeighbors results by @rid."""
        seen: set[str] = set()
        out: list[dict] = []
        for r in rows:
            rid = r.get("@rid", r.get("id", ""))
            if rid not in seen:
                seen.add(rid)
                out.append(r)
        return out

    # ------------------------------------------------------------------
    # STM chunk writes
    # ------------------------------------------------------------------

    async def write_stm_chunk(self, chunk: dict[str, Any]) -> None:
        """Merge a chunk into STM and set its embedding."""
        cid = chunk["id"]
        mc = (chunk.get("metadata") or {}).get("memory_class") or "raw_evidence"
        await self._c.command(
            _STM_DB,
            "MERGE (n:CHUNK {id: $id}) SET n.session_id = $sid, n.text = $text, "
            "n.file_id = $fid, n.heading_path = $hp, n.memory_class = $mc, "
            "n.metadata = $meta",
            params={
                "id": cid,
                "sid": chunk.get("session_id", ""),
                "text": chunk.get("text", ""),
                "fid": chunk.get("file_id") or "",
                "hp": chunk.get("heading_path") or "",
                "mc": mc,
                "meta": _dump_meta(chunk.get("metadata")),
            },
        )
        if chunk.get("embedding"):
            await self._set_embedding(_STM_DB, "CHUNK", cid, chunk["embedding"])

    async def write_stm_turn(self, turn: dict[str, Any]) -> None:
        """Merge a turn into STM and set its embedding."""
        tid = turn["id"]
        await self._c.command(
            _STM_DB,
            "MERGE (n:TURN {id: $id}) SET n.session_id = $sid, n.role = $role, "
            "n.text = $text, n.dialog_id = $did, n.turn_index = $ti, "
            "n.metadata = $meta",
            params={
                "id": tid,
                "sid": turn.get("session_id", ""),
                "role": turn.get("role", "unknown"),
                "text": turn.get("text", ""),
                "did": turn.get("dialog_id") or "",
                "ti": turn.get("turn_index") or 0,
                "meta": _dump_meta(turn.get("metadata")),
            },
        )
        if turn.get("embedding"):
            await self._set_embedding(_STM_DB, "TURN", tid, turn["embedding"])

    # ------------------------------------------------------------------
    # STM search
    # ------------------------------------------------------------------

    async def search_stm_chunks(
        self, session_id: str, embedding: list[float], top_k: int
    ) -> list[dict[str, Any]]:
        """ANN search on STM chunks filtered to session_id."""
        if not embedding:
            return []
        # Over-fetch: multiple old sessions share the same corpus, so we need
        # enough candidates to survive the session filter.  k=200 reliably
        # captures all chunks in any realistic eval database.
        k_pre = max(top_k * 20, 200)
        sql = _vec_search_sql("CHUNK[embedding]", embedding, k_pre)
        try:
            rows = await self._c.query(_STM_DB, sql, language="sql")
        except Exception as exc:
            logger.warning("search_stm_chunks failed: %s", exc)
            return []
        rows = self._dedup_rows(rows)
        results = []
        for r in rows:
            if r.get("session_id") != session_id:
                continue
            dist = float(r.get("distance") or 1.0)
            results.append(
                {
                    "id": r.get("id", ""),
                    "session_id": r.get("session_id", ""),
                    "file_id": r.get("file_id") or "",
                    "text": r.get("text", ""),
                    "heading_path": r.get("heading_path") or "",
                    "metadata": _load_meta(r.get("metadata")),
                    "cosine_score": max(0.0, 1.0 - dist),
                }
            )
            if len(results) >= top_k:
                break
        return results

    async def search_stm_turns(
        self, session_id: str, embedding: list[float], top_k: int
    ) -> list[dict[str, Any]]:
        """ANN search on STM turns filtered to session_id."""
        if not embedding:
            return []
        k_pre = max(top_k * 20, 200)
        sql = _vec_search_sql("TURN[embedding]", embedding, k_pre)
        try:
            rows = await self._c.query(_STM_DB, sql, language="sql")
        except Exception as exc:
            logger.warning("search_stm_turns failed: %s", exc)
            return []
        rows = self._dedup_rows(rows)
        results = []
        for r in rows:
            if r.get("session_id") != session_id:
                continue
            dist = float(r.get("distance") or 1.0)
            results.append(
                {
                    "id": r.get("id", ""),
                    "session_id": r.get("session_id", ""),
                    "dialog_id": r.get("dialog_id") or "",
                    "turn_index": r.get("turn_index"),
                    "role": r.get("role", ""),
                    "text": r.get("text", ""),
                    "metadata": _load_meta(r.get("metadata")),
                    "cosine_score": max(0.0, 1.0 - dist),
                }
            )
            if len(results) >= top_k:
                break
        return results

    # ------------------------------------------------------------------
    # LTM chunk/turn search
    # ------------------------------------------------------------------

    async def search_ltm_chunks(
        self,
        embedding: list[float],
        top_k: int,
        session_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """ANN search on LTM chunks."""
        if not embedding:
            return []
        k_pre = max(top_k * 20, 200) if session_id else top_k
        sql = _vec_search_sql("CHUNK[embedding]", embedding, k_pre)
        try:
            rows = await self._c.query(_LTM_DB, sql, language="sql")
        except Exception as exc:
            logger.warning("search_ltm_chunks failed: %s", exc)
            return []
        rows = self._dedup_rows(rows)
        results = []
        for r in rows:
            if session_id and r.get("session_id") != session_id:
                continue
            dist = float(r.get("distance") or 1.0)
            results.append(
                {
                    "id": r.get("id", ""),
                    "session_id": r.get("session_id") or "",
                    "file_id": r.get("file_id") or "",
                    "text": r.get("text", ""),
                    "heading_path": r.get("heading_path") or "",
                    "metadata": _load_meta(r.get("metadata")),
                    "cosine_score": max(0.0, 1.0 - dist),
                }
            )
            if len(results) >= top_k:
                break
        return results

    async def search_ltm_turns(
        self,
        embedding: list[float],
        top_k: int,
        session_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """ANN search on LTM turns."""
        if not embedding:
            return []
        k_pre = max(top_k * 20, 200) if session_id else top_k
        sql = _vec_search_sql("TURN[embedding]", embedding, k_pre)
        try:
            rows = await self._c.query(_LTM_DB, sql, language="sql")
        except Exception as exc:
            logger.warning("search_ltm_turns failed: %s", exc)
            return []
        rows = self._dedup_rows(rows)
        results = []
        for r in rows:
            if session_id and r.get("session_id") != session_id:
                continue
            dist = float(r.get("distance") or 1.0)
            results.append(
                {
                    "id": r.get("id", ""),
                    "session_id": r.get("session_id") or "",
                    "dialog_id": r.get("dialog_id") or "",
                    "turn_index": r.get("turn_index"),
                    "role": r.get("role", ""),
                    "text": r.get("text", ""),
                    "metadata": _load_meta(r.get("metadata")),
                    "cosine_score": max(0.0, 1.0 - dist),
                }
            )
            if len(results) >= top_k:
                break
        return results

    async def search_chunks(
        self,
        query_emb: list[float],
        session_id: str,
        k: int = 20,
    ) -> list[dict[str, Any]]:
        """ANN search routing: STM when session_id set, LTM otherwise."""
        if session_id:
            rows = await self.search_stm_chunks(session_id, query_emb, k)
        else:
            rows = await self.search_ltm_chunks(query_emb, k)
        return [
            {
                "chunk_id": r["id"],
                "doc_id": r.get("file_id", ""),
                "text": r["text"],
                "metadata": r.get("metadata", {}),
                "cosine_score": r.get("cosine_score", 0.0),
            }
            for r in rows
        ]

    async def search_entities(
        self,
        query_emb: list[float],
        k: int = 10,
        entity_types: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """ANN search on corpus-level entity embeddings."""
        if not query_emb:
            return []
        sql = _vec_search_sql("ENTITY[embedding]", query_emb, k * 2)
        try:
            rows = await self._c.query(_LTM_DB, sql, language="sql")
        except Exception as exc:
            logger.debug("search_entities failed (non-fatal): %s", exc)
            return []
        rows = self._dedup_rows(rows)
        results = []
        for r in rows:
            label = r.get("label") or r.get("type") or "Entity"
            if entity_types and label not in entity_types:
                continue
            dist = float(r.get("distance") or 1.0)
            results.append(
                {
                    "name": r.get("name", ""),
                    "label": label,
                    "score": max(0.0, 1.0 - dist),
                }
            )
            if len(results) >= k:
                break
        return results

    # ------------------------------------------------------------------
    # Bulk fetch by IDs
    # ------------------------------------------------------------------

    async def get_chunks_by_ids(self, chunk_ids: list[str]) -> dict[str, dict[str, Any]]:
        """Fetch chunks by ID from both STM and LTM."""
        if not chunk_ids:
            return {}
        result: dict[str, dict[str, Any]] = {}
        for db in (_STM_DB, _LTM_DB):
            try:
                rows = await self._c.query(
                    db,
                    "MATCH (n:CHUNK) WHERE n.id IN $ids RETURN n.id AS id, "
                    "n.file_id AS file_id, n.text AS text, "
                    "n.memory_class AS memory_class, n.metadata AS metadata",
                    params={"ids": chunk_ids},
                )
            except Exception:
                continue
            for r in rows:
                cid = r.get("id", "")
                if cid and cid not in result:
                    meta = _load_meta(r.get("metadata"))
                    meta.setdefault("memory_class", r.get("memory_class") or "raw_evidence")
                    result[cid] = {
                        "doc_id": r.get("file_id") or "",
                        "text": r.get("text", ""),
                        "metadata": meta,
                    }
        return result

    # ------------------------------------------------------------------
    # STM cleanup
    # ------------------------------------------------------------------

    async def clear_stm(self, session_id: str) -> None:
        """Delete all STM chunks and turns for session_id."""
        for vtype in ("CHUNK", "TURN"):
            try:
                await self._c.command(
                    _STM_DB,
                    f"MATCH (n:{vtype} {{session_id: $sid}}) DETACH DELETE n",
                    params={"sid": session_id},
                )
            except Exception as exc:
                logger.debug("clear_stm %s %s: %s", vtype, session_id, exc)

    async def delete_session(self, session_id: str) -> int:
        """Delete session data from both STM and LTM."""
        deleted = 0
        for db in (_STM_DB, _LTM_DB):
            for vtype in ("CHUNK", "TURN"):
                try:
                    rows = await self._c.query(
                        db,
                        f"MATCH (n:{vtype} {{session_id: $sid}}) RETURN count(n) AS c",
                        params={"sid": session_id},
                    )
                    n = int((rows[0].get("c") or 0) if rows else 0)
                    if n > 0:
                        await self._c.command(
                            db,
                            f"MATCH (n:{vtype} {{session_id: $sid}}) DETACH DELETE n",
                            params={"sid": session_id},
                        )
                        deleted += n
                except Exception as exc:
                    logger.debug("delete_session %s %s %s: %s", db, vtype, session_id, exc)
        return deleted

    # ------------------------------------------------------------------
    # Promotion: STM → LTM
    # ------------------------------------------------------------------

    async def promote_to_ltm(self, session_id: str) -> tuple[int, int]:
        """Copy STM chunks and turns to LTM."""
        chunks_promoted = await self._promote_type(session_id, "CHUNK")
        turns_promoted = await self._promote_type(session_id, "TURN")
        logger.info(
            "promote_to_ltm session=%s chunks=%d turns=%d",
            session_id,
            chunks_promoted,
            turns_promoted,
        )
        return chunks_promoted, turns_promoted

    async def _promote_type(self, session_id: str, vtype: str) -> int:
        """Read vertices from STM and write to LTM for a given type."""
        try:
            rows = await self._c.query(
                _STM_DB,
                f"MATCH (n:{vtype} {{session_id: $sid}}) "
                f"RETURN n.id AS id, n.session_id AS session_id, n.text AS text, "
                f"n.file_id AS file_id, n.heading_path AS heading_path, "
                f"n.dialog_id AS dialog_id, n.turn_index AS turn_index, "
                f"n.role AS role, n.embedding AS embedding, n.memory_class AS memory_class",
                params={"sid": session_id},
            )
        except Exception as exc:
            logger.warning("promote_to_ltm read %s: %s", vtype, exc)
            return 0

        count = 0
        for r in rows:
            rid = r.get("id", "")
            if not rid:
                continue
            # MERGE in LTM (without embedding)
            props = {
                k: v
                for k, v in r.items()
                if v is not None and not k.startswith("@") and k != "embedding"
            }
            set_clauses = ", ".join(f"n.{k} = ${k}" for k in props if k != "id")
            if set_clauses:
                await self._c.command(
                    _LTM_DB,
                    f"MERGE (n:{vtype} {{id: $id}}) SET {set_clauses}",
                    params={"id": rid, **props},
                )
            else:
                await self._c.command(
                    _LTM_DB,
                    f"MERGE (n:{vtype} {{id: $id}})",
                    params={"id": rid},
                )
            # Set embedding via SQL
            emb = r.get("embedding")
            if emb:
                await self._set_embedding(_LTM_DB, vtype, rid, emb)
            count += 1
        return count

    # ------------------------------------------------------------------
    # LTM corpus writes
    # ------------------------------------------------------------------

    async def upsert_chunks(self, session_id: str, chunks: list[dict[str, Any]]) -> None:
        """Upsert chunks into LTM."""
        for c in chunks:
            cid = c.get("id", "")
            if not cid:
                continue
            await self._c.command(
                _LTM_DB,
                "MERGE (n:CHUNK {id: $id}) SET n.session_id = $sid, n.text = $text, "
                "n.file_id = $fid",
                params={
                    "id": cid,
                    "sid": session_id,
                    "text": c.get("text", ""),
                    "fid": c.get("doc_id", ""),
                },
            )
            if c.get("embedding"):
                await self._set_embedding(_LTM_DB, "CHUNK", cid, c["embedding"])

    async def upsert_entities(self, entities: list[dict[str, Any]]) -> None:
        """Upsert corpus-level entity embeddings."""
        for e in entities:
            name = e.get("name", "")
            if not name:
                continue
            await self._c.command(
                _LTM_DB,
                "MERGE (n:ENTITY {name: $name}) SET n.label = $label, n.chunk_count = $cc",
                params={
                    "name": name,
                    "label": e.get("label", "Entity"),
                    "cc": e.get("chunk_count", 0),
                },
            )
            if e.get("embedding"):
                await self._set_embedding_by_name(_LTM_DB, "ENTITY", name, e["embedding"])

    async def write_ltm_topic(self, topic: dict[str, Any]) -> None:
        """Upsert a topic into LTM."""
        tid = topic.get("topic_id") or topic.get("id", "")
        if not tid:
            return
        await self._c.command(
            _LTM_DB,
            "MERGE (n:TOPIC {id: $id}) SET n.community_id = $cid, n.summary = $summary",
            params={
                "id": tid,
                "cid": topic.get("community_id"),
                "summary": topic.get("summary", ""),
            },
        )
        if topic.get("embedding"):
            await self._set_embedding(_LTM_DB, "TOPIC", tid, topic["embedding"])

    async def write_community_assignments(
        self,
        session_id: str,
        assignments: list[dict[str, Any]],
    ) -> None:
        """Write community metadata onto LTM CHUNK and TURN vertices."""
        if not assignments:
            return
        for assignment in assignments:
            community_id = assignment.get("community_id")
            weight = float(assignment.get("weight", 1.0))
            chunk_ids = assignment.get("chunk_ids") or []
            turn_ids = assignment.get("turn_ids") or []

            for cid in chunk_ids:
                with contextlib.suppress(Exception):
                    await self._c.command(
                        _LTM_DB,
                        "MATCH (n:CHUNK {id: $id, session_id: $sid}) "
                        "SET n.community_id = $cid, n.community_weight = $w",
                        params={"id": cid, "sid": session_id, "cid": community_id, "w": weight},
                    )
            for tid in turn_ids:
                with contextlib.suppress(Exception):
                    await self._c.command(
                        _LTM_DB,
                        "MATCH (n:TURN {id: $id, session_id: $sid}) "
                        "SET n.community_id = $cid, n.community_weight = $w",
                        params={"id": tid, "sid": session_id, "cid": community_id, "w": weight},
                    )

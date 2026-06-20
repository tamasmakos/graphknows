"""ArcadeDB STM store — session-scoped entity graph.

Data lives in an ArcadeDB database named stm.  All operations use
openCypher via the ArcadeDB HTTP REST API.  Schema DDL (CREATE VERTEX TYPE,
indexes) uses ArcadeDB SQL because Cypher has no DDL surface.
"""

from __future__ import annotations

import datetime
import logging
from typing import Any

from graphknows.memory.stores.arcadedb._stm_schema import (
    _DB,
    _SCHEMA_DDL,
    _SINGLE_VALUED_RELATIONS,
    _safe_rel_label,
)
from graphknows.memory.stores.arcadedb.client import ArcadeDBClient

logger = logging.getLogger(__name__)

class ArcadeDBSTMStore:
    """Session-scoped graph store backed by ArcadeDB.

    Connects via the ArcadeDB HTTP REST API using openCypher for graph
    operations and SQL for schema DDL.

    Args:
        client: Connected ``ArcadeDBClient`` instance.
    """

    def __init__(self, client: ArcadeDBClient) -> None:
        self._db = _DB
        self._c = client

    def close(self) -> None:
        """No-op — lifecycle is managed by the client passed in."""

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    async def ensure_schema(self) -> None:
        """Create vertex types, edge types, and indexes idempotently."""
        await self._c.create_database(self._db)
        for stmt in _SCHEMA_DDL:
            try:
                await self._c.command(self._db, stmt, language="sql")
            except Exception:
                logger.debug("STM schema DDL skipped: %s", stmt, exc_info=True)

    # ------------------------------------------------------------------
    # Session metadata
    # ------------------------------------------------------------------

    async def write_session_metadata(
        self,
        session_id: str,
        metadata: dict[str, Any],
    ) -> None:
        """Upsert a SESSION node with arbitrary metadata properties."""
        # Build SET clauses for each metadata key individually — ArcadeDB
        # Cypher may not support the `+=` map-merge operator.
        set_clauses = ", ".join(f"n.{k} = ${k}" for k in metadata)
        cypher = "MERGE (n:SESSION {id: $sid})"
        if set_clauses:
            cypher += f" SET {set_clauses}"
        params: dict[str, Any] = {"sid": session_id, **metadata}
        await self._c.command(self._db, cypher, params=params)

    # ------------------------------------------------------------------
    # File nodes
    # ------------------------------------------------------------------

    async def write_file_node(
        self,
        session_id: str,
        file_id: str,
        title: str,
        parser_name: str,
        source_hint: str,
    ) -> None:
        """Upsert a FILE node linked to a SESSION via IN_SESSION."""
        await self._c.command(
            self._db,
            """
            MERGE (f:FILE {id: $fid})
            SET f.session_id = $sid,
                f.title = $title,
                f.parser_name = $parser_name,
                f.source_hint = $source_hint
            WITH f
            MATCH (sess:SESSION {id: $sid})
            MERGE (sess)-[:IN_SESSION]->(f)
            """,
            params={
                "fid": file_id,
                "sid": session_id,
                "title": title,
                "parser_name": parser_name,
                "source_hint": source_hint,
            },
        )

    # ------------------------------------------------------------------
    # Chunk nodes
    # ------------------------------------------------------------------

    async def write_chunk_node(
        self,
        session_id: str,
        file_id: str,
        chunk_id: str,
        text: str,
        heading_path: str,
        metadata: dict[str, Any],
    ) -> None:
        """Upsert a CHUNK node linked to its FILE via HAS_CHUNK."""
        chunk_index = metadata.get("chunk_index")
        await self._c.command(
            self._db,
            """
            MERGE (c:CHUNK {id: $cid})
            SET c.session_id = $sid,
                c.file_id = $fid,
                c.text = $text,
                c.heading_path = $heading_path,
                c.chunk_index = $chunk_index,
                c.metadata = $metadata
            WITH c
            MATCH (f:FILE {id: $fid})
            MERGE (f)-[:HAS_CHUNK]->(c)
            """,
            params={
                "cid": chunk_id,
                "sid": session_id,
                "fid": file_id,
                "text": text,
                "heading_path": heading_path,
                "chunk_index": chunk_index,
                "metadata": str(metadata),
            },
        )

    # ------------------------------------------------------------------
    # Entity nodes
    # ------------------------------------------------------------------

    async def write_entity_node(
        self,
        session_id: str,
        entity_name: str,
        entity_type: str | None,
        confidence: float,
    ) -> None:
        """Upsert an ENTITY node scoped to a session."""
        etype = entity_type or "ENTITY"
        await self._c.command(
            self._db,
            """
            MERGE (e:ENTITY {name: $name, session_id: $sid})
            SET e.type = $etype,
                e.confidence = CASE
                    WHEN $conf > coalesce(e.confidence, 0) THEN $conf
                    ELSE e.confidence
                END
            """,
            params={
                "name": entity_name,
                "sid": session_id,
                "etype": etype,
                "conf": confidence,
            },
        )

    async def write_chunk_entity_edge(
        self,
        chunk_id: str,
        entity_name: str,
        session_id: str,
    ) -> None:
        """Create a MENTIONS edge from CHUNK to ENTITY."""
        await self._c.command(
            self._db,
            """
            MATCH (c:CHUNK {id: $cid}),
                  (e:ENTITY {name: $name, session_id: $sid})
            MERGE (c)-[:MENTIONS]->(e)
            """,
            params={"cid": chunk_id, "name": entity_name, "sid": session_id},
        )

    # ------------------------------------------------------------------
    # Dialog and turn nodes
    # ------------------------------------------------------------------

    async def write_dialog_node(
        self,
        session_id: str,
        dialog_id: str,
        started_at: str,
    ) -> None:
        """Upsert a DIALOG node linked to a SESSION."""
        await self._c.command(
            self._db,
            """
            MERGE (d:DIALOG {id: $did})
            SET d.session_id = $sid, d.started_at = $started_at
            WITH d
            MATCH (sess:SESSION {id: $sid})
            MERGE (sess)-[:IN_SESSION]->(d)
            """,
            params={"did": dialog_id, "sid": session_id, "started_at": started_at},
        )

    async def write_turn_node(
        self,
        dialog_id: str,
        session_id: str,
        turn_id: str,
        turn_index: int,
        role: str,
        text: str,
        timestamp: str,
    ) -> None:
        """Upsert a TURN node linked to its DIALOG."""
        await self._c.command(
            self._db,
            """
            MERGE (t:TURN {id: $tid})
            SET t.session_id = $sid,
                t.dialog_id = $did,
                t.turn_index = $idx,
                t.role = $role,
                t.text = $text,
                t.timestamp = $ts
            WITH t
            MATCH (d:DIALOG {id: $did})
            MERGE (d)-[:HAS_TURN]->(t)
            """,
            params={
                "tid": turn_id,
                "sid": session_id,
                "did": dialog_id,
                "idx": turn_index,
                "role": role,
                "text": text,
                "ts": timestamp,
            },
        )

    async def write_turn_sequence(self, prev_turn_id: str, next_turn_id: str) -> None:
        """Create a NEXT edge between consecutive TURNs."""
        await self._c.command(
            self._db,
            """
            MATCH (a:TURN {id: $aid}), (b:TURN {id: $bid})
            MERGE (a)-[:NEXT]->(b)
            """,
            params={"aid": prev_turn_id, "bid": next_turn_id},
        )

    async def write_turn_entity_edge(
        self,
        turn_id: str,
        entity_name: str,
        session_id: str,
        confidence: float,
    ) -> None:
        """Create a MENTIONS edge from TURN to ENTITY."""
        await self._c.command(
            self._db,
            """
            MATCH (t:TURN {id: $tid}),
                  (e:ENTITY {name: $name, session_id: $sid})
            MERGE (t)-[r:MENTIONS]->(e)
            SET r.confidence = $conf
            """,
            params={
                "tid": turn_id,
                "name": entity_name,
                "sid": session_id,
                "conf": confidence,
            },
        )

    # ------------------------------------------------------------------
    # Bi-temporal relation edges
    # ------------------------------------------------------------------

    async def write_relation_edge(
        self,
        session_id: str,
        subject_name: str,
        relation_name: str,
        object_name: str,
        fact: str = "",
        confidence: float = 0.5,
        valid_at: Any | None = None,
        created_at: Any | None = None,
    ) -> None:
        """Write a bi-temporal relation edge between two named entities.

        For single-valued relation types any existing current edge pointing
        to a different object is tombstoned atomically within a transaction.
        """
        safe = _safe_rel_label(relation_name)
        if not safe:
            return

        now = datetime.datetime.utcnow().isoformat()

        # Ensure the edge type is declared — ArcadeDB requires this before MERGE.
        try:
            await self._c.command(
                self._db,
                f"CREATE EDGE TYPE {safe} IF NOT EXISTS",
                language="sql",
            )
        except Exception:
            logger.debug("Edge type DDL skipped for %s", safe, exc_info=True)

        tx = await self._c.begin(self._db)
        try:
            # Tombstone existing current edge for single-valued types.
            if relation_name in _SINGLE_VALUED_RELATIONS:
                await self._c.command(
                    self._db,
                    f"""
                    MATCH (a:ENTITY {{name: $subject, session_id: $session_id}})
                          -[r:{safe} {{session_id: $session_id, is_current: true}}]
                          ->(b:ENTITY)
                    WHERE toLower(b.name) <> toLower($object)
                    SET r.is_current = false, r.valid_to = $now
                    """,
                    params={
                        "subject": subject_name,
                        "session_id": session_id,
                        "object": object_name,
                        "now": now,
                    },
                    session_id=tx,
                )

            # Upsert endpoint entities and MERGE the relation edge.
            await self._c.command(
                self._db,
                f"""
                MERGE (a:ENTITY {{name: $subject, session_id: $session_id}})
                MERGE (b:ENTITY {{name: $object, session_id: $session_id}})
                MERGE (a)-[r:{safe} {{session_id: $session_id}}]->(b)
                SET r.fact       = $fact,
                    r.confidence = $confidence,
                    r.valid_from = coalesce(r.valid_from, $now),
                    r.valid_to   = null,
                    r.is_current = true,
                    r.created_at = coalesce(r.created_at, $now)
                """,
                params={
                    "subject": subject_name,
                    "object": object_name,
                    "session_id": session_id,
                    "fact": fact,
                    "confidence": confidence,
                    "now": now,
                },
                session_id=tx,
            )
            await self._c.commit(self._db, tx)
        except Exception:
            await self._c.rollback(self._db, tx)
            raise

    # ------------------------------------------------------------------
    # Deprecated write methods (backward compatibility with old ingest paths)
    # ------------------------------------------------------------------

    async def write_entities(self, session_id: str, entities: list[Any]) -> None:
        """Upsert entity nodes under *session_id* (deprecated, use write_entity_node)."""
        for e in entities:
            await self._c.command(
                self._db,
                """
                MERGE (n:ENTITY {name: $name, session_id: $sid})
                ON CREATE SET
                    n.label = $label,
                    n.id = $id,
                    n.confidence = $confidence
                ON MATCH SET
                    n.confidence = CASE
                        WHEN $confidence > n.confidence THEN $confidence
                        ELSE n.confidence
                    END
                """,
                params={
                    "name": e.name,
                    "label": e.label,
                    "id": e.id,
                    "sid": session_id,
                    "confidence": e.confidence,
                },
            )

    async def write_relations(self, session_id: str, relations: list[Any]) -> None:
        """Upsert RELATES edges (deprecated, use write_relation_edge)."""
        for r in relations:
            await self._c.command(
                self._db,
                """
                MATCH (a:ENTITY {id: $src, session_id: $sid}),
                      (b:ENTITY {id: $tgt, session_id: $sid})
                MERGE (a)-[rel:RELATES {type: $rtype, session_id: $sid}]->(b)
                """,
                params={
                    "src": r.source,
                    "tgt": r.target,
                    "rtype": r.type,
                    "sid": session_id,
                },
            )

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------

    async def query_entity_chunks(
        self,
        session_id: str,
        entity_names: list[str],
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """Return CHUNK nodes linked to any of the given entity names in *session_id*.

        Uses dynamic OR conditions instead of the any() list predicate because
        ArcadeDB's openCypher does not support any(x IN $list WHERE ...) syntax.
        """
        if not entity_names:
            return []
        safe = [n.replace("'", "") for n in entity_names]
        where_parts = " OR ".join(f"toLower(e.name) CONTAINS '{n}'" for n in safe)
        rows = await self._c.query(
            self._db,
            f"MATCH (e:ENTITY {{session_id: $sid}}) WHERE {where_parts} "
            f"MATCH (c:CHUNK)-[:MENTIONS]->(e) "
            f"RETURN DISTINCT c.id AS chunk_id, c.text AS text LIMIT $k",
            params={"sid": session_id, "k": top_k},
        )
        return [{"chunk_id": r["chunk_id"], "text": r["text"]} for r in rows]

    async def full_text_search_chunks(
        self,
        session_id: str,
        terms: list[str],
        top_k: int = 5,
        require_all: bool = False,
    ) -> list[dict[str, Any]]:
        """Return CHUNK nodes whose text contains the given terms.

        require_all=True uses AND (all terms must appear); False uses OR.
        """
        if not terms:
            return []
        safe = [t.replace("'", "") for t in terms]
        op = " AND " if require_all else " OR "
        where_parts = op.join(f"toLower(c.text) CONTAINS '{t}'" for t in safe)
        rows = await self._c.query(
            self._db,
            f"MATCH (c:CHUNK {{session_id: $sid}}) WHERE {where_parts} "
            f"RETURN c.id AS chunk_id, c.text AS text LIMIT $k",
            params={"sid": session_id, "k": top_k},
        )
        return [{"chunk_id": r["chunk_id"], "text": r["text"]} for r in rows]

    async def get_session_nodes(self, session_id: str) -> list[dict[str, Any]]:
        """Return all entity nodes for *session_id* as plain dicts."""
        rows = await self._c.query(
            self._db,
            "MATCH (n:ENTITY {session_id: $sid}) RETURN n",
            params={"sid": session_id},
        )
        # ArcadeDB REST returns each row as a dict of projected columns.
        # When projecting a node with RETURN n, the value is a nested dict.
        results = []
        for r in rows:
            node = r.get("n", r)
            if isinstance(node, dict):
                results.append({k: v for k, v in node.items() if not k.startswith("@")})
            else:
                results.append(r)
        return results

    async def get_session_edges(self, session_id: str) -> list[dict[str, Any]]:
        """Return all RELATES edges for *session_id* as plain dicts."""
        rows = await self._c.query(
            self._db,
            """
            MATCH (a:ENTITY {session_id: $sid})-[r:RELATES]->(b:ENTITY {session_id: $sid})
            RETURN a.name AS from, b.name AS to, r.type AS rel_type
            """,
            params={"sid": session_id},
        )
        return [{"from": r["from"], "to": r["to"], "rel_type": r["rel_type"]} for r in rows]

    async def get_session_graph(self, session_id: str) -> dict[str, Any]:
        """Return nodes and edges for *session_id* as a graph dict."""
        nodes = await self.get_session_nodes(session_id)
        edges = await self.get_session_edges(session_id)
        return {
            "session_id": session_id,
            "node_count": len(nodes),
            "edge_count": len(edges),
            "nodes": nodes,
            "edges": edges,
        }

    async def read_session_subgraph(self, session_id: str) -> dict[str, Any]:
        """Read the full STM subgraph for *session_id*.

        Returns a structured dict consumed by SessionFlusher:
        {
            entities, files, chunks, dialogs, turns,
            chunk_entity_edges, turn_entity_edges, next_edges,
            relation_edges, _merge_count
        }
        """
        # Each query is run independently — ArcadeDB Cypher does not support
        # UNION across multiple MATCH patterns in a single query reliably.
        entities_primary = await self._c.query(
            self._db,
            """
            MATCH (sess:SESSION)-[:IN_SESSION]->(f:FILE)-[:HAS_CHUNK]->(c:CHUNK)
                  -[:HAS_ENTITY|MENTIONS]->(e:ENTITY)
            WHERE sess.id = $sid
            RETURN DISTINCT e.name AS name, e.type AS type, e.confidence AS confidence
            """,
            params={"sid": session_id},
        )
        entities_fallback = await self._c.query(
            self._db,
            """
            MATCH (e:ENTITY)
            WHERE e.session_id = $sid
            RETURN DISTINCT e.name AS name, e.type AS type, e.confidence AS confidence
            """,
            params={"sid": session_id},
        )
        primary_names = {r["name"] for r in entities_primary}
        fallback_only = [r for r in entities_fallback if r["name"] not in primary_names]
        if fallback_only and not entities_primary:
            logger.warning(
                "Session %s: CHUNK→ENTITY chain broken — %d entities via fallback only.",
                session_id,
                len(fallback_only),
            )
        entities = [
            {"name": r["name"], "type": r["type"], "confidence": r["confidence"] or 1.0}
            for r in entities_primary + fallback_only
        ]

        files_rows = await self._c.query(
            self._db,
            """
            MATCH (sess:SESSION)-[:IN_SESSION]->(f:FILE)
            WHERE sess.id = $sid
            RETURN f.id AS id, f.title AS title,
                   f.parser_name AS parser_name, f.source_hint AS source_hint
            """,
            params={"sid": session_id},
        )
        files = [
            {
                "id": r["id"],
                "title": r["title"],
                "parser_name": r["parser_name"],
                "source_hint": r["source_hint"],
            }
            for r in files_rows
        ]

        chunks_rows = await self._c.query(
            self._db,
            """
            MATCH (sess:SESSION)-[:IN_SESSION]->(f:FILE)-[:HAS_CHUNK]->(c:CHUNK)
            WHERE sess.id = $sid
            RETURN c.id AS id, f.id AS file_id, c.text AS text,
                   c.heading_path AS heading_path, c.chunk_index AS chunk_index
            """,
            params={"sid": session_id},
        )
        chunks = [
            {
                "id": r["id"],
                "file_id": r["file_id"],
                "text": r["text"],
                "heading_path": r["heading_path"],
                "chunk_index": r["chunk_index"] if r["chunk_index"] is not None else 0,
            }
            for r in chunks_rows
        ]

        dialogs_rows = await self._c.query(
            self._db,
            """
            MATCH (sess:SESSION)-[:IN_SESSION]->(d:DIALOG)
            WHERE sess.id = $sid
            RETURN d.id AS id, d.session_id AS session_id, d.started_at AS started_at
            """,
            params={"sid": session_id},
        )
        dialogs = [
            {"id": r["id"], "session_id": r["session_id"], "started_at": r["started_at"]}
            for r in dialogs_rows
        ]

        turns_rows = await self._c.query(
            self._db,
            """
            MATCH (sess:SESSION)-[:IN_SESSION]->(d:DIALOG)-[:HAS_TURN]->(t:TURN)
            WHERE sess.id = $sid
            RETURN t.id AS id, d.id AS dialog_id, t.session_id AS session_id,
                   t.turn_index AS turn_index, t.role AS role, t.text AS text
            """,
            params={"sid": session_id},
        )
        turns = [
            {
                "id": r["id"],
                "dialog_id": r["dialog_id"],
                "session_id": r["session_id"],
                "turn_index": r["turn_index"],
                "role": r["role"],
                "text": r["text"],
            }
            for r in turns_rows
        ]

        # Chunk-entity edges — union of chain path and fallback.
        ce_primary = await self._c.query(
            self._db,
            """
            MATCH (sess:SESSION)-[:IN_SESSION]->(f:FILE)-[:HAS_CHUNK]->(c:CHUNK)
                  -[:HAS_ENTITY|MENTIONS]->(e:ENTITY)
            WHERE sess.id = $sid
            RETURN c.id AS chunk_id, e.name AS entity_name
            """,
            params={"sid": session_id},
        )
        ce_fallback = await self._c.query(
            self._db,
            """
            MATCH (c:CHUNK)-[:MENTIONS|HAS_ENTITY]->(e:ENTITY)
            WHERE e.session_id = $sid
            RETURN c.id AS chunk_id, e.name AS entity_name
            """,
            params={"sid": session_id},
        )
        seen_ce: set[tuple[str, str]] = set()
        chunk_entity_edges = []
        for row in ce_primary + ce_fallback:
            key = (row["chunk_id"], row["entity_name"])
            if key not in seen_ce:
                seen_ce.add(key)
                chunk_entity_edges.append(
                    {"chunk_id": row["chunk_id"], "entity_name": row["entity_name"]}
                )

        turn_entity_rows = await self._c.query(
            self._db,
            """
            MATCH (sess:SESSION)-[:IN_SESSION]->(d:DIALOG)-[:HAS_TURN]->(t:TURN)
                  -[:HAS_ENTITY|MENTIONS]->(e:ENTITY)
            WHERE sess.id = $sid
            RETURN t.id AS turn_id, e.name AS entity_name
            """,
            params={"sid": session_id},
        )
        turn_entity_edges = [
            {"turn_id": r["turn_id"], "entity_name": r["entity_name"]} for r in turn_entity_rows
        ]

        next_rows = await self._c.query(
            self._db,
            """
            MATCH (sess:SESSION)-[:IN_SESSION]->(d:DIALOG)-[:HAS_TURN]->(t1:TURN)
                  -[:NEXT]->(t2:TURN)
            WHERE sess.id = $sid
            RETURN t1.id AS from_turn_id, t2.id AS to_turn_id
            """,
            params={"sid": session_id},
        )
        next_edges = [
            {"from_turn_id": r["from_turn_id"], "to_turn_id": r["to_turn_id"]} for r in next_rows
        ]

        relation_rows = await self._c.query(
            self._db,
            """
            MATCH (a:ENTITY {session_id: $sid})-[r]->(b:ENTITY {session_id: $sid})
            WHERE r.session_id = $sid AND r.is_current = true AND type(r) <> 'REL'
            RETURN a.name AS source, type(r) AS relation,
                   b.name AS target, r.confidence AS confidence
            """,
            params={"sid": session_id},
        )
        relation_edges = [
            {
                "source": r["source"],
                "relation": r["relation"],
                "target": r["target"],
                "confidence": r["confidence"],
            }
            for r in relation_rows
        ]

        return {
            "entities": entities,
            "files": files,
            "chunks": chunks,
            "dialogs": dialogs,
            "turns": turns,
            "chunk_entity_edges": chunk_entity_edges,
            "turn_entity_edges": turn_entity_edges,
            "next_edges": next_edges,
            "relation_edges": relation_edges,
            "_merge_count": 0,
        }

    async def get_session_node_count(self, session_id: str) -> int:
        """Return the number of ENTITY nodes stored for *session_id*."""
        rows = await self._c.query(
            self._db,
            "MATCH (n:ENTITY {session_id: $sid}) RETURN count(n) AS c",
            params={"sid": session_id},
        )
        return int(rows[0]["c"]) if rows else 0

    async def check_chain_integrity(self, session_id: str) -> dict[str, Any]:
        """Check SESSION→FILE→CHUNK chain integrity for *session_id*."""

        async def _count(cypher: str) -> int:
            rows = await self._c.query(self._db, cypher, params={"sid": session_id})
            return int(rows[0][next(iter(rows[0]))]) if rows else 0

        session_count = await _count("MATCH (sess:SESSION {id: $sid}) RETURN count(sess) AS n")
        file_count = await _count(
            "MATCH (sess:SESSION {id: $sid})-[:IN_SESSION]->(f:FILE) RETURN count(f) AS n"
        )
        chunk_count = await _count(
            "MATCH (sess:SESSION {id: $sid})-[:IN_SESSION]->(f:FILE)-[:HAS_CHUNK]->(c:CHUNK) "
            "RETURN count(c) AS n"
        )
        entity_count = await _count(
            "MATCH (sess:SESSION {id: $sid})-[:IN_SESSION]->(f:FILE)-[:HAS_CHUNK]->(c:CHUNK)"
            "-[:HAS_ENTITY|MENTIONS]->(e:ENTITY) RETURN count(DISTINCT e) AS n"
        )
        fallback_entity_count = await _count(
            "MATCH (e:ENTITY) WHERE e.session_id = $sid RETURN count(e) AS n"
        )
        return {
            "session_id": session_id,
            "has_session_node": session_count > 0,
            "file_count": file_count,
            "chunk_count": chunk_count,
            "entity_count": entity_count,
            "chain_ok": file_count > 0 and chunk_count > 0 and entity_count > 0,
            "fallback_entity_count": fallback_entity_count,
        }

    # ------------------------------------------------------------------
    # Session deletion
    # ------------------------------------------------------------------

    async def delete_session(self, session_id: str) -> None:
        """Delete all nodes for *session_id* from the STM database."""
        # Use Cypher DETACH DELETE per label — handles cascading edge removal.
        for label in ("TURN", "DIALOG", "CHUNK", "ENTITY", "FILE"):
            try:
                await self._c.command(
                    self._db,
                    f"MATCH (n:{label} {{session_id: $sid}}) DETACH DELETE n",
                    params={"sid": session_id},
                )
            except Exception:
                logger.debug("delete_session: %s delete ignored", label, exc_info=True)
        try:
            await self._c.command(
                self._db,
                "MATCH (n:SESSION {id: $sid}) DETACH DELETE n",
                params={"sid": session_id},
            )
        except Exception:
            logger.debug("delete_session: SESSION delete ignored", exc_info=True)

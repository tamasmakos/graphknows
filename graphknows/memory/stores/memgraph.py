"""Optional Memgraph backend for session entity graphs.

Used as an alternative to Neo4j for live session entity storage.
Memgraph uses the Bolt protocol. Authentication is enabled via
``MEMGRAPH_USER`` / ``MEMGRAPH_PASSWORD`` environment variables.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
from typing import Any

from graphknows.ingestion.domain.schema import MEMGRAPH_INDEXES


def _safe_rel_label(raw: str) -> str:
    """Convert a relation string to a valid Cypher relationship type label.

    Uppercases, replaces spaces/hyphens with underscores, strips non-word chars.
    Returns empty string if nothing valid remains.
    """
    label = raw.upper().replace("-", "_").replace(" ", "_")
    label = re.sub(r"[^\w]", "", label)
    return label


logger = logging.getLogger(__name__)


class MemgraphStore:
    """Async Memgraph store for session-scoped entity graphs.

    Attributes:
        uri: Bolt URI for Memgraph, e.g. bolt://memgraph:7687.

    Authentication is read from ``MEMGRAPH_USER`` / ``MEMGRAPH_PASSWORD``
    environment variables.  If ``MEMGRAPH_USER`` is unset, the driver
    connects without credentials (legacy / dev mode).
    """

    def __init__(self, uri: str) -> None:
        from neo4j import GraphDatabase

        user = os.environ.get("MEMGRAPH_USER", "")
        password = os.environ.get("MEMGRAPH_PASSWORD", "")
        auth = (user, password) if user else None
        self._driver = GraphDatabase.driver(uri, auth=auth)

    def close(self) -> None:
        """Close the Bolt driver connection."""
        self._driver.close()

    async def ensure_indexes(self) -> None:
        """Create STM indexes from schema.MEMGRAPH_INDEXES idempotently."""

        def _sync() -> None:
            with self._driver.session() as s:
                for stmt in MEMGRAPH_INDEXES:
                    try:
                        s.run(stmt).consume()
                    except Exception:
                        logger.debug("Memgraph index creation ignored", exc_info=True)

        await asyncio.to_thread(_sync)

    # ------------------------------------------------------------------
    # Session metadata
    # ------------------------------------------------------------------

    async def write_session_metadata(
        self,
        session_id: str,
        metadata: dict[str, Any],
    ) -> None:
        """Upsert a SESSION node with arbitrary metadata properties.

        Args:
            session_id: Unique session identifier.
            metadata: Flat dict of properties to set on the node.
        """

        def _sync() -> None:
            with self._driver.session() as s:
                s.run(
                    "MERGE (n:SESSION {id: $sid}) SET n += $props",
                    sid=session_id,
                    props=metadata,
                ).consume()

        await asyncio.to_thread(_sync)

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
        """Upsert a FILE node linked to a SESSION.

        Args:
            session_id: Owning session.
            file_id: Unique file identifier.
            title: Human-readable file title.
            parser_name: Parser used for extraction.
            source_hint: Original source path or URL hint.
        """

        def _sync() -> None:
            with self._driver.session() as s:
                s.run(
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
                    fid=file_id,
                    sid=session_id,
                    title=title,
                    parser_name=parser_name,
                    source_hint=source_hint,
                ).consume()

        await asyncio.to_thread(_sync)

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
        """Upsert a CHUNK node linked to its FILE.

        Args:
            session_id: Owning session.
            file_id: Parent file identifier.
            chunk_id: Unique chunk identifier.
            text: Chunk text content.
            heading_path: Slash-separated heading breadcrumb.
            metadata: Additional metadata properties.
        """
        chunk_index = metadata.get("chunk_index")

        def _sync() -> None:
            with self._driver.session() as s:
                s.run(
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
                    cid=chunk_id,
                    sid=session_id,
                    fid=file_id,
                    text=text,
                    heading_path=heading_path,
                    chunk_index=chunk_index,
                    metadata=str(metadata),
                ).consume()

        await asyncio.to_thread(_sync)

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
        """Upsert an ENTITY node scoped to a session.

        Args:
            session_id: Owning session.
            entity_name: Canonical entity name (merge key with session).
            entity_type: Entity label/type string.
            confidence: Extraction confidence score.
        """

        def _sync() -> None:
            etype = entity_type or "ENTITY"
            with self._driver.session() as s:
                s.run(
                    """
                    MERGE (e:ENTITY {name: $name, session_id: $sid})
                    SET e.type = $etype,
                        e.confidence = CASE
                            WHEN $conf > coalesce(e.confidence, 0) THEN $conf
                            ELSE e.confidence
                        END
                    """,
                    name=entity_name,
                    sid=session_id,
                    etype=etype,
                    conf=confidence,
                ).consume()

        await asyncio.to_thread(_sync)

    async def write_chunk_entity_edge(
        self,
        chunk_id: str,
        entity_name: str,
        session_id: str,
    ) -> None:
        """Create a MENTIONS edge from CHUNK to ENTITY.

        Args:
            chunk_id: Source chunk identifier.
            entity_name: Target entity name.
            session_id: Session namespace for entity lookup.
        """

        def _sync() -> None:
            with self._driver.session() as s:
                s.run(
                    """
                    MATCH (c:CHUNK {id: $cid}),
                          (e:ENTITY {name: $name, session_id: $sid})
                    MERGE (c)-[:MENTIONS]->(e)
                    """,
                    cid=chunk_id,
                    name=entity_name,
                    sid=session_id,
                ).consume()

        await asyncio.to_thread(_sync)

    # ------------------------------------------------------------------
    # Dialog and turn nodes
    # ------------------------------------------------------------------

    async def write_dialog_node(
        self,
        session_id: str,
        dialog_id: str,
        started_at: str,
    ) -> None:
        """Upsert a DIALOG node linked to a SESSION.

        Args:
            session_id: Owning session.
            dialog_id: Unique dialog identifier.
            started_at: ISO-8601 timestamp string.
        """

        def _sync() -> None:
            with self._driver.session() as s:
                s.run(
                    """
                    MERGE (d:DIALOG {id: $did})
                    SET d.session_id = $sid, d.started_at = $started_at
                    WITH d
                    MATCH (sess:SESSION {id: $sid})
                    MERGE (sess)-[:IN_SESSION]->(d)
                    """,
                    did=dialog_id,
                    sid=session_id,
                    started_at=started_at,
                ).consume()

        await asyncio.to_thread(_sync)

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
        """Upsert a TURN node linked to its DIALOG.

        Args:
            dialog_id: Parent dialog identifier.
            session_id: Owning session.
            turn_id: Unique turn identifier.
            turn_index: Ordinal position within dialog.
            role: Speaker role: 'user', 'agent', or 'system'.
            text: Turn text content.
            timestamp: ISO-8601 timestamp string.
        """

        def _sync() -> None:
            with self._driver.session() as s:
                s.run(
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
                    tid=turn_id,
                    sid=session_id,
                    did=dialog_id,
                    idx=turn_index,
                    role=role,
                    text=text,
                    ts=timestamp,
                ).consume()

        await asyncio.to_thread(_sync)

    async def write_turn_sequence(self, prev_turn_id: str, next_turn_id: str) -> None:
        """Create a NEXT edge between consecutive TURNs.

        Args:
            prev_turn_id: Source turn identifier.
            next_turn_id: Destination turn identifier.
        """

        def _sync() -> None:
            with self._driver.session() as s:
                s.run(
                    """
                    MATCH (a:TURN {id: $aid}), (b:TURN {id: $bid})
                    MERGE (a)-[:NEXT]->(b)
                    """,
                    aid=prev_turn_id,
                    bid=next_turn_id,
                ).consume()

        await asyncio.to_thread(_sync)

    async def query_entity_chunks(
        self,
        session_id: str,
        entity_names: list[str],
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """Return CHUNK nodes linked to any of the given entity names in *session_id*.

        Args:
            session_id: Session namespace for entity lookup.
            entity_names: List of entity names to match.
            top_k: Maximum number of chunk results to return.

        Returns:
            List of dicts with chunk_id and text.
        """

        def _sync() -> list[dict[str, Any]]:
            with self._driver.session() as s:
                rows = list(
                    s.run(
                        """
                        UNWIND $names AS n
                        MATCH (c:CHUNK)-[:MENTIONS]->(e:ENTITY {name: n, session_id: $sid})
                        RETURN DISTINCT c.id AS chunk_id, c.text AS text
                        LIMIT $k
                        """,
                        names=entity_names,
                        sid=session_id,
                        k=top_k,
                    )
                )
            return [{"chunk_id": r["chunk_id"], "text": r["text"]} for r in rows]

        return await asyncio.to_thread(_sync)

    # ------------------------------------------------------------------
    async def write_turn_entity_edge(
        self,
        turn_id: str,
        entity_name: str,
        session_id: str,
        confidence: float,
    ) -> None:
        """Create a MENTIONS edge from TURN to ENTITY.

        Args:
            turn_id: Source turn identifier.
            entity_name: Target entity name.
            session_id: Session namespace for entity lookup.
            confidence: Mention confidence score.
        """

        def _sync() -> None:
            with self._driver.session() as s:
                s.run(
                    """
                    MATCH (t:TURN {id: $tid}),
                          (e:ENTITY {name: $name, session_id: $sid})
                    MERGE (t)-[r:MENTIONS]->(e)
                    SET r.confidence = $conf
                    """,
                    tid=turn_id,
                    name=entity_name,
                    sid=session_id,
                    conf=confidence,
                ).consume()

        await asyncio.to_thread(_sync)

    # ------------------------------------------------------------------
    # Deprecated — kept for backward compatibility, do not use in new code
    # ------------------------------------------------------------------

    async def write_entities(self, session_id: str, entities: list[Any]) -> None:
        """Upsert entity nodes under *session_id*.

        .. deprecated::
            Use :meth:`write_entity_node` for new code.
        """
        if not entities:
            return

        def _sync() -> None:
            with self._driver.session() as s:
                for e in entities:
                    s.run(
                        """MERGE (n:ENTITY {name: $name, session_id: $sid})
                           ON CREATE SET
                               n.label = $label,
                               n.all_labels = [$label],
                               n.id = $id,
                               n.confidence = $confidence,
                               n.doc_ids = $doc_ids
                           ON MATCH SET
                               n.all_labels = CASE
                                   WHEN $label IN n.all_labels THEN n.all_labels
                                   ELSE n.all_labels + [$label]
                               END,
                               n.label = CASE
                                   WHEN n.label = 'Entity' THEN $label
                                   ELSE n.label
                               END,
                               n.confidence = CASE
                                   WHEN $confidence > n.confidence THEN $confidence
                                   ELSE n.confidence
                               END""",
                        name=e.name,
                        label=e.label,
                        id=e.id,
                        sid=session_id,
                        confidence=e.confidence,
                        doc_ids=e.doc_ids,
                    )

        await asyncio.to_thread(_sync)

    async def write_relations(self, session_id: str, relations: list[Any]) -> None:
        """Upsert RELATES edges between session entities.

        .. deprecated::
            Use :meth:`write_chunk_entity_edge` for new code.
        """
        if not relations:
            return

        def _sync() -> None:
            with self._driver.session() as s:
                for r in relations:
                    s.run(
                        """MATCH (a:ENTITY {id:$src, session_id:$sid}),
                                (b:ENTITY {id:$tgt, session_id:$sid})
                           MERGE (a)-[rel:RELATES {type:$rtype, session_id:$sid}]->(b)""",
                        src=r.source,
                        tgt=r.target,
                        rtype=r.type,
                        sid=session_id,
                    )

        await asyncio.to_thread(_sync)

    # Relation types where only one target per (head, relation_type) should be
    # current at a time. When a new fact arrives for a single-valued relation,
    # any existing current edge with a *different* tail is tombstoned.
    _SINGLE_VALUED_RELATIONS: frozenset[str] = frozenset(
        {
            "AFFILIATED_WITH",
            "LOCATED_IN",
            "PART_OF",
            "PRODUCED_BY",
            "PRECEDED_BY",
            "SUCCEEDED_BY",
        }
    )

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
        """Write a bi-temporal REL edge between two named entities in Memgraph STM.

        Edge properties added/updated:
          - ``valid_from``: ISO timestamp when this fact became valid (= now on create).
          - ``valid_to``: None while current; set to close timestamp when tombstoned.
          - ``is_current``: True while the fact is the authoritative current state.

        For single-valued relation types (e.g. AFFILIATED_WITH, LOCATED_IN) a new
        incoming edge with a *different* tail automatically tombstones any existing
        current edge for the same (subject, relation_name) pair — no LLM needed.
        """
        import datetime

        now = datetime.datetime.utcnow().isoformat()

        safe = _safe_rel_label(relation_name)
        if not safe:
            return

        def _sync() -> None:
            with self._driver.session() as s:
                # Invalidation: tombstone existing current edges for single-valued types.
                # Match all typed edges from subject with is_current=true pointing at a
                # different object — we use a generic MATCH on any rel type here since
                # Memgraph does not support dynamic labels in MATCH without f-strings, and
                # the session_id + is_current filter is sufficient for correctness.
                if relation_name in self._SINGLE_VALUED_RELATIONS:
                    s.run(
                        f"""MATCH (a:ENTITY {{name: $subject, session_id: $session_id}})
                               -[r:{safe} {{session_id: $session_id, is_current: true}}]
                               ->(b:ENTITY)
                           WHERE toLower(b.name) <> toLower($object)
                           SET r.is_current = false, r.valid_to = $now""",
                        subject=subject_name,
                        session_id=session_id,
                        object=object_name,
                        now=now,
                    )

                # Write new edge with typed label and bi-temporal fields.
                # MERGE cannot use dynamic labels, so we CREATE if not exists via
                # MATCH + MERGE pattern with a discriminating property set.
                s.run(
                    f"""MATCH (a:ENTITY {{name: $subject, session_id: $session_id}})
                       MATCH (b:ENTITY {{name: $object, session_id: $session_id}})
                       MERGE (a)-[r:{safe} {{session_id: $session_id}}]->(b)
                       SET r.fact        = $fact,
                           r.confidence  = $confidence,
                           r.valid_from  = coalesce(r.valid_from, $now),
                           r.valid_to    = null,
                           r.is_current  = true,
                           r.created_at  = coalesce(r.created_at, $now)""",
                    subject=subject_name,
                    object=object_name,
                    session_id=session_id,
                    fact=fact,
                    confidence=confidence,
                    now=now,
                )

        await asyncio.to_thread(_sync)

    async def delete_session(self, session_id: str) -> None:
        """Delete all nodes and edges for *session_id*."""

        def _sync() -> None:
            with self._driver.session() as s:
                s.run("MATCH (n {session_id: $sid}) DETACH DELETE n", sid=session_id)

        await asyncio.to_thread(_sync)

    async def get_session_nodes(self, session_id: str) -> list[dict[str, Any]]:
        """Return all entity nodes for *session_id* as plain dicts."""

        def _sync() -> list[dict[str, Any]]:
            with self._driver.session() as s:
                rows = list(
                    s.run(
                        "MATCH (n:ENTITY {session_id: $sid}) RETURN properties(n) AS props",
                        sid=session_id,
                    )
                )
            return [dict(r["props"]) for r in rows]

        return await asyncio.to_thread(_sync)

    async def get_session_edges(self, session_id: str) -> list[dict[str, Any]]:
        """Return all RELATES edges for *session_id* as plain dicts."""

        def _sync() -> list[dict[str, Any]]:
            with self._driver.session() as s:
                rows = list(
                    s.run(
                        """MATCH (a:ENTITY {session_id:$sid})-[r:RELATES]->(b:ENTITY {session_id:$sid})
                       RETURN a.name AS from, b.name AS to, r.type AS rel_type""",
                        sid=session_id,
                    )
                )
            return [{"from": r["from"], "to": r["to"], "rel_type": r["rel_type"]} for r in rows]

        return await asyncio.to_thread(_sync)

    async def get_session_graph(self, session_id: str) -> dict[str, Any]:
        """Return nodes and edges for *session_id* as a graph dict.

        Args:
            session_id: Session namespace to query.

        Returns:
            Dict with session_id, node_count, edge_count, nodes, and edges.
        """
        nodes, edges = await asyncio.gather(
            self.get_session_nodes(session_id),
            self.get_session_edges(session_id),
        )
        return {
            "session_id": session_id,
            "node_count": len(nodes),
            "edge_count": len(edges),
            "nodes": nodes,
            "edges": edges,
        }

    async def read_session_subgraph(self, session_id: str) -> dict:
        """Read the full STM subgraph for *session_id* from Memgraph.

        Returns a structured dict consumed by SessionFlusher:
        {
            entities, files, chunks, dialogs, turns,
            chunk_entity_edges, turn_entity_edges, next_edges, _merge_count
        }
        """

        def _sync() -> dict:
            with self._driver.session() as s:
                # --- Entities: split UNION into two queries and detect fallback ---
                primary_entities = [
                    {"name": r["name"], "type": r["type"], "confidence": r["confidence"] or 1.0}
                    for r in s.run(
                        "MATCH (sess:SESSION)-[:IN_SESSION]->(f:FILE)-[:HAS_CHUNK]->(c:CHUNK)"
                        "-[:HAS_ENTITY|MENTIONS]->(e:ENTITY) "
                        "WHERE sess.id = $sid "
                        "RETURN DISTINCT e.name AS name, e.type AS type, e.confidence AS confidence",
                        sid=session_id,
                    )
                ]
                fallback_entities = [
                    {"name": r["name"], "type": r["type"], "confidence": r["confidence"] or 1.0}
                    for r in s.run(
                        "MATCH (e:ENTITY) WHERE e.session_id = $sid "
                        "RETURN DISTINCT e.name AS name, e.type AS type, e.confidence AS confidence",
                        sid=session_id,
                    )
                ]
                primary_names = {e["name"] for e in primary_entities}
                fallback_only = [e for e in fallback_entities if e["name"] not in primary_names]
                if fallback_only and not primary_entities:
                    logger.warning(
                        "Session %s: CHUNK→ENTITY chain broken — %d entities found via fallback only. "
                        "SESSION→FILE→CHUNK chain may be missing. Check MCP ingest path.",
                        session_id,
                        len(fallback_only),
                    )
                # Merge: primary takes precedence, add any fallback-only extras
                entities = primary_entities + fallback_only
                files = [
                    {
                        "id": r["id"],
                        "title": r["title"],
                        "parser_name": r["parser_name"],
                        "source_hint": r["source_hint"],
                    }
                    for r in s.run(
                        "MATCH (sess:SESSION)-[:IN_SESSION]->(f:FILE) "
                        "WHERE sess.id = $sid "
                        "RETURN f.id AS id, f.title AS title, f.parser_name AS parser_name, "
                        "f.source_hint AS source_hint",
                        sid=session_id,
                    )
                ]
                chunks = [
                    {
                        "id": r["id"],
                        "file_id": r["file_id"],
                        "text": r["text"],
                        "heading_path": r["heading_path"],
                        "chunk_index": r["chunk_index"] if r["chunk_index"] is not None else 0,
                    }
                    for r in s.run(
                        "MATCH (sess:SESSION)-[:IN_SESSION]->(f:FILE)-[:HAS_CHUNK]->(c:CHUNK) "
                        "WHERE sess.id = $sid "
                        "RETURN c.id AS id, f.id AS file_id, c.text AS text, "
                        "c.heading_path AS heading_path, c.chunk_index AS chunk_index",
                        sid=session_id,
                    )
                ]
                dialogs = [
                    {"id": r["id"], "session_id": r["session_id"], "started_at": r["started_at"]}
                    for r in s.run(
                        "MATCH (sess:SESSION)-[:IN_SESSION]->(d:DIALOG) "
                        "WHERE sess.id = $sid "
                        "RETURN d.id AS id, d.session_id AS session_id, d.started_at AS started_at",
                        sid=session_id,
                    )
                ]
                turns = [
                    {
                        "id": r["id"],
                        "dialog_id": r["dialog_id"],
                        "session_id": r["session_id"],
                        "turn_index": r["turn_index"],
                        "role": r["role"],
                        "text": r["text"],
                    }
                    for r in s.run(
                        "MATCH (sess:SESSION)-[:IN_SESSION]->(d:DIALOG)-[:HAS_TURN]->(t:TURN) "
                        "WHERE sess.id = $sid "
                        "RETURN t.id AS id, d.id AS dialog_id, t.session_id AS session_id, "
                        "t.turn_index AS turn_index, t.role AS role, t.text AS text",
                        sid=session_id,
                    )
                ]
                chunk_entity_edges = [
                    {"chunk_id": r["chunk_id"], "entity_name": r["entity_name"]}
                    for r in s.run(
                        "MATCH (sess:SESSION)-[:IN_SESSION]->(f:FILE)-[:HAS_CHUNK]->(c:CHUNK)"
                        "-[:HAS_ENTITY|MENTIONS]->(e:ENTITY) "
                        "WHERE sess.id = $sid "
                        "RETURN c.id AS chunk_id, e.name AS entity_name"
                        " UNION "
                        "MATCH (c:CHUNK)-[:MENTIONS|HAS_ENTITY]->(e:ENTITY) "
                        "WHERE e.session_id = $sid "
                        "RETURN c.id AS chunk_id, e.name AS entity_name",
                        sid=session_id,
                    )
                ]
                turn_entity_edges = [
                    {"turn_id": r["turn_id"], "entity_name": r["entity_name"]}
                    for r in s.run(
                        "MATCH (sess:SESSION)-[:IN_SESSION]->(d:DIALOG)-[:HAS_TURN]->(t:TURN)"
                        "-[:HAS_ENTITY|MENTIONS]->(e:ENTITY) "
                        "WHERE sess.id = $sid "
                        "RETURN t.id AS turn_id, e.name AS entity_name",
                        sid=session_id,
                    )
                ]
                next_edges = [
                    {"from_turn_id": r["from_turn_id"], "to_turn_id": r["to_turn_id"]}
                    for r in s.run(
                        "MATCH (sess:SESSION)-[:IN_SESSION]->(d:DIALOG)-[:HAS_TURN]->(t1:TURN)"
                        "-[:NEXT]->(t2:TURN) "
                        "WHERE sess.id = $sid "
                        "RETURN t1.id AS from_turn_id, t2.id AS to_turn_id",
                        sid=session_id,
                    )
                ]
                relation_edges = [
                    {
                        "source": r["source"],
                        "relation": r["relation"],
                        "target": r["target"],
                        "confidence": r["confidence"],
                    }
                    for r in s.run(
                        "MATCH (a:ENTITY {session_id: $sid})"
                        "-[r {session_id: $sid, is_current: true}]->"
                        "(b:ENTITY {session_id: $sid}) "
                        "WHERE type(r) <> 'REL' "
                        "RETURN a.name AS source, type(r) AS relation, b.name AS target, "
                        "r.confidence AS confidence",
                        sid=session_id,
                    )
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

        return await asyncio.to_thread(_sync)

    async def get_session_node_count(self, session_id: str) -> int:
        """Return the number of Entity nodes stored for *session_id*."""

        def _sync() -> int:
            with self._driver.session() as s:
                rows = list(
                    s.run(
                        "MATCH (n:ENTITY {session_id:$sid}) RETURN count(n) AS c",
                        sid=session_id,
                    )
                )
            return int(rows[0]["c"]) if rows else 0

        return await asyncio.to_thread(_sync)

    async def check_chain_integrity(self, session_id: str) -> dict:
        """Check SESSION→FILE→CHUNK chain integrity for *session_id*.

        Returns:
            Dict with keys: session_id, has_session_node, file_count, chunk_count,
            entity_count, chain_ok, fallback_entity_count.
        """

        def _sync() -> dict:
            with self._driver.session() as s:

                def _count(cypher: str) -> int:
                    rows = list(s.run(cypher, sid=session_id))
                    return int(rows[0][0]) if rows else 0

                session_count = _count("MATCH (sess:SESSION {id: $sid}) RETURN count(sess) AS n")
                file_count = _count(
                    "MATCH (sess:SESSION {id: $sid})-[:IN_SESSION]->(f:FILE) RETURN count(f) AS n"
                )
                chunk_count = _count(
                    "MATCH (sess:SESSION {id: $sid})-[:IN_SESSION]->(f:FILE)-[:HAS_CHUNK]->(c:CHUNK) "
                    "RETURN count(c) AS n"
                )
                entity_count = _count(
                    "MATCH (sess:SESSION {id: $sid})-[:IN_SESSION]->(f:FILE)-[:HAS_CHUNK]->(c:CHUNK)"
                    "-[:HAS_ENTITY|MENTIONS]->(e:ENTITY) RETURN count(DISTINCT e) AS n"
                )
                fallback_entity_count = _count(
                    "MATCH (e:ENTITY) WHERE e.session_id = $sid RETURN count(e) AS n"
                )
                chain_ok = file_count > 0 and chunk_count > 0 and entity_count > 0
                return {
                    "session_id": session_id,
                    "has_session_node": session_count > 0,
                    "file_count": file_count,
                    "chunk_count": chunk_count,
                    "entity_count": entity_count,
                    "chain_ok": chain_ok,
                    "fallback_entity_count": fallback_entity_count,
                }

        return await asyncio.to_thread(_sync)

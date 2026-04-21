"""
Neo4j Knowledge Graph Uploader.

Uses the official async neo4j driver.
All Cypher uses parameterised queries — never f-string interpolation.
"""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Any, Dict, List, Optional

import networkx as nx
from neo4j import AsyncDriver

from kg.schema import (
    LABEL_CHUNK,
    LABEL_DOCUMENT,
    LABEL_ENTITY,
    REL_CONTAINS,
    REL_MENTIONS,
    REL_RELATED_TO,
)

logger = logging.getLogger(__name__)


def _clean_props(props: Dict[str, Any]) -> Dict[str, Any]:
    """Strip None values and serialise non-primitive types to JSON strings."""
    cleaned: Dict[str, Any] = {}
    for k, v in props.items():
        if v is None:
            continue
        if isinstance(v, (bool, int, float, str)):
            cleaned[k] = v
        elif isinstance(v, (list, tuple)):
            if v and all(isinstance(x, (int, float)) for x in v):
                cleaned[k] = list(v)  # embedding — keep as numeric list
            else:
                try:
                    cleaned[k] = json.dumps(v, ensure_ascii=False)
                except (TypeError, ValueError):
                    pass
        elif isinstance(v, dict):
            try:
                cleaned[k] = json.dumps(v, ensure_ascii=False)
            except (TypeError, ValueError):
                pass
        else:
            cleaned[k] = str(v)
    return cleaned


def _content_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


class Neo4jUploader:
    """
    Upload a NetworkX knowledge graph to Neo4j using the DOCUMENT→CHUNK→ENTITY schema.

    Parameters
    ----------
    driver:
        An open AsyncDriver instance.
    database:
        Target Neo4j database name (default "neo4j").
    batch_size:
        Number of nodes/edges to write per transaction.
    """

    def __init__(
        self,
        driver: AsyncDriver,
        database: str = "neo4j",
        batch_size: int = 500,
    ) -> None:
        self.driver = driver
        self.database = database
        self.batch_size = batch_size

    # ── Public API ─────────────────────────────────────────────────────────────

    async def upload(
        self,
        graph: nx.DiGraph,
        clean_database: bool = False,
    ) -> Dict[str, int]:
        """
        Write *graph* to Neo4j.

        Returns a stats dict: {nodes_created, edges_created, nodes_merged, ...}
        """
        stats: Dict[str, int] = {
            "documents": 0,
            "chunks": 0,
            "entities": 0,
            "contains": 0,
            "mentions": 0,
            "related_to": 0,
        }

        async with self.driver.session(database=self.database) as session:
            if clean_database:
                logger.info("Clearing Neo4j database '%s'…", self.database)
                await session.run("MATCH (n) DETACH DELETE n")

            # Separate nodes by type
            documents: List[Dict[str, Any]] = []
            chunks: List[Dict[str, Any]] = []
            entities: List[Dict[str, Any]] = []

            for node_id, data in graph.nodes(data=True):
                node_type = data.get("node_type", data.get("type", "")).upper()
                props = _clean_props({k: v for k, v in data.items() if k != "node_type"})

                if node_type in {"DOCUMENT"}:
                    props.setdefault("doc_id", str(node_id))
                    documents.append(props)
                elif node_type in {"CHUNK"}:
                    props.setdefault("chunk_id", str(node_id))
                    chunks.append(props)
                elif node_type in {"ENTITY", "ENTITY_CONCEPT"}:
                    props.setdefault("entity_id", str(node_id))
                    props.setdefault("name", props.get("entity_id", ""))
                    entities.append(props)

            # Upsert nodes
            stats["documents"] = await self._upsert_nodes(
                session, LABEL_DOCUMENT, "doc_id", documents
            )
            stats["chunks"] = await self._upsert_nodes(session, LABEL_CHUNK, "chunk_id", chunks)
            stats["entities"] = await self._upsert_nodes(
                session, LABEL_ENTITY, "entity_id", entities
            )

            # Write edges
            for src, dst, data in graph.edges(data=True):
                rel_type = data.get("graph_type", data.get("type", "")).upper()
                props = _clean_props(
                    {k: v for k, v in data.items() if k not in {"graph_type", "type"}}
                )

                if rel_type in {"LEXICAL", "CONTAINS"}:
                    stats["contains"] += await self._upsert_edge(
                        session,
                        LABEL_DOCUMENT,
                        "doc_id",
                        str(src),
                        REL_CONTAINS,
                        LABEL_CHUNK,
                        "chunk_id",
                        str(dst),
                        props,
                    )
                elif rel_type in {"MENTIONS", "LEXICAL_ENTITY"}:
                    stats["mentions"] += await self._upsert_edge(
                        session,
                        LABEL_CHUNK,
                        "chunk_id",
                        str(src),
                        REL_MENTIONS,
                        LABEL_ENTITY,
                        "entity_id",
                        str(dst),
                        props,
                    )
                elif rel_type in {"ENTITY_RELATION", "RELATED_TO"}:
                    stats["related_to"] += await self._upsert_edge(
                        session,
                        LABEL_ENTITY,
                        "entity_id",
                        str(src),
                        REL_RELATED_TO,
                        LABEL_ENTITY,
                        "entity_id",
                        str(dst),
                        props,
                    )

        logger.info("Upload complete: %s", stats)
        return stats

    # ── Helpers ────────────────────────────────────────────────────────────────

    async def _upsert_nodes(
        self,
        session: Any,
        label: str,
        key: str,
        nodes: List[Dict[str, Any]],
    ) -> int:
        if not nodes:
            return 0

        total = 0
        for i in range(0, len(nodes), self.batch_size):
            batch = nodes[i : i + self.batch_size]
            cypher = (
                f"UNWIND $batch AS props MERGE (n:{label} {{{key}: props.{key}}}) SET n += props"
            )
            await session.run(cypher, {"batch": batch})
            total += len(batch)

        logger.debug("Upserted %d %s nodes", total, label)
        return total

    async def _upsert_edge(
        self,
        session: Any,
        from_label: str,
        from_key: str,
        from_val: str,
        rel_type: str,
        to_label: str,
        to_key: str,
        to_val: str,
        props: Optional[Dict[str, Any]] = None,
    ) -> int:
        cypher = (
            f"MATCH (a:{from_label} {{{from_key}: $from_val}}) "
            f"MATCH (b:{to_label} {{{to_key}: $to_val}}) "
            f"MERGE (a)-[r:{rel_type}]->(b) "
            f"SET r += $props"
        )
        await session.run(
            cypher,
            {"from_val": from_val, "to_val": to_val, "props": props or {}},
        )
        return 1

    async def upload_parsed_document(self, parsed: "ParsedDocument") -> None:
        """
        Store a ParsedDocument (from kg.parser) directly in Neo4j
        without running entity extraction.
        Used by the /documents upload endpoint.
        """
        from datetime import datetime, timezone

        now = datetime.now(timezone.utc).isoformat()

        doc_props = _clean_props(
            {
                "doc_id": parsed.doc_id,
                "title": parsed.title,
                "source_path": str(parsed.source_path) if parsed.source_path else "",
                "mime_type": parsed.mime_type,
                "created_at": now,
            }
        )

        chunk_rows = [
            _clean_props(
                {
                    "chunk_id": c.chunk_id,
                    "doc_id": c.doc_id,
                    "position": c.position,
                    "text": c.text,
                    "heading_path": c.heading_path,
                    "token_count": c.token_count,
                }
            )
            for c in parsed.chunks
        ]

        async with self.driver.session(database=self.database) as session:
            # Upsert Document node
            await session.run(
                "MERGE (d:Document {doc_id: $doc_id}) SET d += $props",
                {"doc_id": parsed.doc_id, "props": doc_props},
            )
            # Upsert Chunk nodes + CONTAINS edges
            for i in range(0, len(chunk_rows), self.batch_size):
                batch = chunk_rows[i : i + self.batch_size]
                await session.run(
                    """
                    UNWIND $batch AS c
                    MERGE (ch:Chunk {chunk_id: c.chunk_id})
                    SET ch += c
                    WITH ch, c
                    MATCH (d:Document {doc_id: c.doc_id})
                    MERGE (d)-[:CONTAINS]->(ch)
                    """,
                    {"batch": batch},
                )

        logger.info("Stored parsed document %s (%d chunks)", parsed.doc_id, len(parsed.chunks))

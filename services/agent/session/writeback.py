"""WriteBackService v2 — temporal LTM write-back with Segment nodes."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any

from neo4j import AsyncGraphDatabase

from graphknows.schema import apply_schema

log = logging.getLogger("agent.session.writeback")

_schema_applied = False


def ensure_schema(uri: str, user: str, password: str) -> None:
    """Apply schema (constraints + indexes) once at startup using sync driver. Idempotent."""
    global _schema_applied  # noqa: PLW0603
    if not _schema_applied:
        from neo4j import GraphDatabase as _GDB
        drv = _GDB.driver(uri, auth=(user, password))
        try:
            with drv.session() as s:
                apply_schema(s)
        finally:
            drv.close()
        _schema_applied = True


class WriteBackService:
    def __init__(self, neo4j_uri: str, neo4j_user: str, neo4j_password: str) -> None:
        self._uri = neo4j_uri
        self._user = neo4j_user
        self._password = neo4j_password

    async def write_back(
        self,
        session_id: str,
        entities: list[dict],
        relations: list[dict],
        query: str = "",
        min_confidence: float = 0.6,
    ) -> dict[str, Any]:
        """Extract from Memgraph, merge into Neo4j with Segment node."""
        ensure_schema(self._uri, self._user, self._password)
        ts = datetime.now(timezone.utc).isoformat()
        driver = AsyncGraphDatabase.driver(self._uri, auth=(self._user, self._password))
        try:
            async with driver.session() as s:
                # 1. Create/merge Segment node
                await s.run(
                    "MERGE (seg:Segment {session_id: $sid}) "
                    "SET seg.timestamp = $ts, seg.query = $q",
                    sid=session_id, ts=ts, q=query,
                )

                # 2. Merge entities with temporal tracking + weight
                high_conf = [e for e in entities if e.get("confidence", 1.0) >= min_confidence]
                written_entities = 0
                for e in high_conf:
                    await s.run(
                        "MERGE (ent:Entity {name: $name}) "
                        "ON CREATE SET ent.type=$type, ent.first_seen=$ts, ent.last_seen=$ts, ent.confidence_avg=$conf "
                        "ON MATCH SET ent.last_seen=$ts, ent.type=CASE WHEN ent.type IS NULL THEN $type ELSE ent.type END, "
                        "ent.confidence_avg=(coalesce(ent.confidence_avg,0.0)+$conf)/2.0 "
                        "WITH ent "
                        "MATCH (seg:Segment {session_id: $sid}) "
                        "MERGE (ent)-[:APPEARED_IN {confidence: $conf}]->(seg)",
                        name=e["name"], type=e.get("type", "Entity"),
                        ts=ts, conf=e.get("confidence", 1.0), sid=session_id,
                    )
                    written_entities += 1

                # 3. Merge relations with weight accumulation
                high_conf_rels = [r for r in relations if r.get("confidence", 1.0) >= min_confidence]
                written_rels = 0
                for rel in high_conf_rels:
                    rel_type = rel.get("type", "RELATES").upper().replace(" ", "_")
                    await s.run(
                        "MERGE (a:Entity {name: $src}) "
                        "MERGE (b:Entity {name: $tgt}) "
                        "MERGE (a)-[r:RELATES {rel_type: $rtype, session_id: $sid}]->(b) "
                        "SET r.weight=coalesce(r.weight,0)+1, r.confidence=$conf, r.last_seen=$ts",
                        src=rel["source"], tgt=rel["target"],
                        rtype=rel_type, sid=session_id,
                        conf=rel.get("confidence", 1.0), ts=ts,
                    )
                    written_rels += 1

        finally:
            await driver.close()

        # Feature engineering: run centrality + community detection on LTM graph
        try:
            from neo4j import GraphDatabase as _GDB

            from graphknows.feature_engineering import run_feature_engineering

            sync_driver = _GDB.driver(self._uri, auth=(self._user, self._password))
            fe_result = run_feature_engineering(sync_driver, session_id=session_id)
            log.info("Feature engineering: entity_count=%d communities=%d runtime_ms=%.0f",
                     fe_result.get("entity_count", 0), fe_result.get("community_count", 0),
                     0)
            sync_driver.close()
        except Exception:
            log.exception("Feature engineering failed — non-fatal")

        # Topic summarization: generate LLM summaries per Leiden community
        try:
            import os
            import asyncio as _asyncio
            from neo4j import GraphDatabase as _GDB2
            from graphknows.topic_summarizer import TopicSummarizer

            async def _run_topics():
                try:
                    ts = TopicSummarizer(
                        neo4j_driver=_GDB2.driver(self._uri, auth=(self._user, self._password)),
                        pg_conn_string=os.environ["PGVECTOR_DSN"],
                        openrouter_api_key=os.environ["OPENROUTER_API_KEY"],
                    )
                    res = await ts.run()
                    log.info("Topic summarization result: %s", res)
                except Exception:
                    log.exception("Topic summarization failed — non-fatal")

            try:
                loop = _asyncio.get_event_loop()
                loop.create_task(_run_topics())
            except RuntimeError:
                _asyncio.run(_run_topics())
        except Exception:
            log.exception("Topic summarization init failed — non-fatal")

        log.info(
            "WriteBack session=%s entities=%d relations=%d",
            session_id, written_entities, written_rels,
        )
        return {
            "session_id": session_id,
            "written_entities": written_entities,
            "written_relations": written_rels,
            "segment_ts": ts,
        }

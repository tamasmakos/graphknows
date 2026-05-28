# LTM-1 Brief: Neo4j Schema + WriteBack v2 — Segment data model & session_end wiring

## Working Directory
/root/graphknows

## Your Tools
You have native DB access via .opencode/tools/:
- neo4j-query.ts  → run Cypher against Neo4j at bolt://172.18.0.1:7687 (neo4j/[from .env NEO4J_PASSWORD])
- memgraph-query.ts → run Cypher against Memgraph at bolt://172.18.0.1:17687 (no auth)
- pgvector-query.ts → run SQL against PostgreSQL at 172.18.0.1:5432

Use these tools to inspect current state before writing any code.

## Model
claude-sonnet-4.6 via GitHub Copilot

## Context: What Exists

### services/agent/session/writeback.py (58 lines — current state)
Minimal WriteBackService. Only does:
- MERGE (e:Entity {name, label}) — no Segment node, no temporal tracking, no confidence filter on relations

### services/agent/session/end.py (410 lines)
Has merge_session_to_neo4j() which handles node/edge promotion but does NOT create Segment nodes
and does NOT call WriteBackService. These two paths are parallel and need to be unified.

### Environment
- NEO4J_URI=bolt://172.18.0.1:7687, NEO4J_USER=neo4j, password in .env NEO4J_PASSWORD
- MEMGRAPH: bolt://172.18.0.1:17687, no auth
- Python 3.11 venv: /root/graphknows/services/agent/.venv

## What to Build

### Step 1: Inspect current Neo4j schema
Use neo4j-query.ts to run:
- CALL db.schema.visualization()
- MATCH (n) RETURN labels(n), count(n) LIMIT 20
- CALL db.constraints()
- CALL db.indexes()
Record what exists before making changes.

### Step 2: Inspect current Memgraph for a sample session
Use memgraph-query.ts to run:
- MATCH (n) RETURN labels(n), n.name, n.session_id LIMIT 20
- MATCH ()-[r]->() RETURN type(r), count(r) LIMIT 10
This tells you what fields are available to extract.

### Step 3: Create Neo4j schema (constraints + indexes)
In services/agent/session/ create neo4j_schema.py with an ensure_ltm_schema() function that runs:

```cypher
CREATE CONSTRAINT entity_name_unique IF NOT EXISTS FOR (e:Entity) REQUIRE e.name IS UNIQUE;
CREATE CONSTRAINT segment_id_unique IF NOT EXISTS FOR (s:Segment) REQUIRE s.session_id IS UNIQUE;
CREATE CONSTRAINT topic_community_unique IF NOT EXISTS FOR (t:Topic) REQUIRE t.community_id IS UNIQUE;
CREATE INDEX entity_type_idx IF NOT EXISTS FOR (e:Entity) ON (e.type);
CREATE INDEX entity_community_idx IF NOT EXISTS FOR (e:Entity) ON (e.community_id);
CREATE INDEX segment_timestamp_idx IF NOT EXISTS FOR (s:Segment) ON (s.timestamp);
```

Verify each constraint/index was created via neo4j-query.ts after running.

### Step 4: Rewrite WriteBackService (services/agent/session/writeback.py)

Replace the existing minimal implementation with:

```python
"""WriteBackService v2 — temporal LTM write-back with Segment nodes."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any

from neo4j import AsyncGraphDatabase

log = logging.getLogger("agent.session.writeback")


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
        ts = datetime.now(timezone.utc).isoformat()
        driver = AsyncGraphDatabase.driver(self._uri, auth=(self._user, self._password))
        try:
            async with driver.session() as s:
                # 1. Create/merge Segment node
                await s.run(
                    "MERGE (seg:Segment {session_id: $sid}) "
                    "SET seg.timestamp = $ts, seg.query = $query",
                    sid=session_id, ts=ts, query=query,
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
                        name=e["name"], type=e.get("type","Entity"),
                        ts=ts, conf=e.get("confidence",1.0), sid=session_id,
                    )
                    written_entities += 1

                # 3. Merge relations with weight accumulation
                high_conf_rels = [r for r in relations if r.get("confidence", 1.0) >= min_confidence]
                written_rels = 0
                for rel in high_conf_rels:
                    rel_type = rel.get("type","RELATES").upper().replace(" ","_")
                    await s.run(
                        "MERGE (a:Entity {name: $src}) "
                        "MERGE (b:Entity {name: $tgt}) "
                        "MERGE (a)-[r:RELATES {rel_type: $rtype, session_id: $sid}]->(b) "
                        "SET r.weight=coalesce(r.weight,0)+1, r.confidence=$conf, r.last_seen=$ts",
                        src=rel["source"], tgt=rel["target"],
                        rtype=rel_type, sid=session_id,
                        conf=rel.get("confidence",1.0), ts=ts,
                    )
                    written_rels += 1

        finally:
            await driver.close()

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
```

### Step 5: Wire into session/end.py

In merge_session_to_neo4j() (around line 80 of end.py), after the existing Neo4j merge completes,
add an async fire-and-forget call to WriteBackService.write_back():

```python
import asyncio
from agent.session.writeback import WriteBackService
from agent.config import settings  # or however config is accessed

# After existing merge logic completes:
async def _async_writeback():
    try:
        wb = WriteBackService(
            neo4j_uri=settings.NEO4J_URI,
            neo4j_user=settings.NEO4J_USER,
            neo4j_password=settings.NEO4J_PASSWORD,
        )
        await wb.write_back(
            session_id=session_id,
            entities=[{"name": n["name"], "type": n.get("type","Entity"), "confidence": 1.0}
                      for n in nodes],
            relations=[{"source": e["source"], "target": e["target"], "type": e.get("type","RELATES"),
                        "confidence": 1.0} for e in edges],
            query=getattr(sgm, "_session_query", {}).get(session_id, ""),
        )
    except Exception:
        log.exception("LTM writeback failed for session %s — non-fatal", session_id)

# Fire and forget (don't block session close):
try:
    loop = asyncio.get_event_loop()
    loop.create_task(_async_writeback())
except RuntimeError:
    asyncio.run(_async_writeback())
```

### Step 6: Verify end-to-end

Run these verifications after your changes:

```bash
# 1. Schema check
# (use neo4j-query.ts) CALL db.constraints()

# 2. Syntax check
cd /root/graphknows
python3 -B -c "from agent.session.writeback import WriteBackService; print('import OK')"     || PYTHONPATH=/root/graphknows/services python3 -B -c "from agent.session.writeback import WriteBackService; print('import OK')"

# 3. Unit import check for session/end.py
PYTHONPATH=/root/graphknows/services python3 -B -c "import agent.session.end; print('end.py OK')"
```

## Acceptance Criteria
- Neo4j has Segment, Entity, Topic constraints (verify with neo4j-query.ts)
- WriteBackService.write_back() accepts entities + relations + query params
- session/end.py calls write_back asynchronously (fire-and-forget, non-blocking)
- All imports succeed (no missing modules)
- Do NOT commit — leave changes staged for principal review

## Pitfalls
- Do NOT use asyncio.run() inside an already-running event loop — use loop.create_task()
- MERGE on Entity must use only {name:} as the key — not {name, type} (type is a property, not constraint key)
- neo4j AsyncGraphDatabase.driver requires async context — use async with or close() in finally
- session/end.py imports via `from agent.session.manager import ...` — same pattern for writeback import

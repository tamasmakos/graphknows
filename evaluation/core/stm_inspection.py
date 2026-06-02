"""STM inspection helpers for production evaluation runs."""

from __future__ import annotations

import asyncio
import json
import shlex
from typing import Any


_INSPECTOR_SCRIPT = r"""
import asyncio
import json
import sys

from graphknows.config import GraphKnowsSettings
from graphknows.config.builders.stores import build_memgraph_store, build_pgvector_store


async def main(session_id: str) -> None:
    settings = GraphKnowsSettings()
    mg = build_memgraph_store(settings)
    pg = build_pgvector_store(settings)
    await pg.connect()
    try:
        subgraph = await mg.read_session_subgraph(session_id)
        with mg._driver.session() as s:
            rel_rows = list(
                s.run(
                    '''
                    MATCH (a:ENTITY {session_id: $sid})-[r:REL {session_id: $sid}]->(b:ENTITY {session_id: $sid})
                    RETURN a.name AS subject, r.name AS relation, b.name AS object,
                           coalesce(r.confidence, 0.0) AS confidence,
                           coalesce(r.is_current, true) AS is_current
                    ORDER BY confidence DESC, subject, relation, object
                    LIMIT 25
                    ''',
                    sid=session_id,
                )
            )
            label_counts = {
                "SESSION": int(list(s.run("MATCH (:SESSION {id: $sid}) RETURN count(*) AS c", sid=session_id))[0]["c"]),
                "FILE": int(list(s.run("MATCH (:FILE {session_id: $sid}) RETURN count(*) AS c", sid=session_id))[0]["c"]),
                "CHUNK": int(list(s.run("MATCH (:CHUNK {session_id: $sid}) RETURN count(*) AS c", sid=session_id))[0]["c"]),
                "ENTITY": int(list(s.run("MATCH (:ENTITY {session_id: $sid}) RETURN count(*) AS c", sid=session_id))[0]["c"]),
                "DIALOG": int(list(s.run("MATCH (:DIALOG {session_id: $sid}) RETURN count(*) AS c", sid=session_id))[0]["c"]),
                "TURN": int(list(s.run("MATCH (:TURN {session_id: $sid}) RETURN count(*) AS c", sid=session_id))[0]["c"]),
            }
            rel_total = int(
                list(
                    s.run(
                        "MATCH (:ENTITY {session_id: $sid})-[r:REL {session_id: $sid}]->(:ENTITY {session_id: $sid}) RETURN count(r) AS c",
                        sid=session_id,
                    )
                )[0]["c"]
            )
        async with pg._pool.acquire() as conn:
            stm_chunks = int(await conn.fetchval("SELECT count(*) FROM stm_chunks WHERE session_id = $1", session_id))
            stm_turns = int(await conn.fetchval("SELECT count(*) FROM stm_turns WHERE session_id = $1", session_id))
        entities = sorted({(e["name"], e.get("type", "ENTITY")) for e in subgraph.get("entities", [])})
        payload = {
            "label_counts": label_counts,
            "stm_chunks_pgvector": stm_chunks,
            "stm_turns_pgvector": stm_turns,
            "chunk_entity_edges": len(subgraph.get("chunk_entity_edges", [])),
            "turn_entity_edges": len(subgraph.get("turn_entity_edges", [])),
            "relation_edges_total": rel_total,
            "distinct_entity_nodes": len(entities),
            "total_stm_nodes": sum(label_counts.values()),
            "entity_sample": [{"name": name, "type": etype} for name, etype in entities[:25]],
            "relation_sample": [
                {
                    "subject": row["subject"],
                    "relation": row["relation"],
                    "object": row["object"],
                    "confidence": round(float(row["confidence"]), 3),
                    "is_current": bool(row["is_current"]),
                }
                for row in rel_rows
            ],
        }
        print(json.dumps(payload, ensure_ascii=False))
    finally:
        mg.close()
        await pg.close()


asyncio.run(main(sys.argv[1]))
"""


def derive_inspect_command(command: str) -> str | None:
    """Best-effort derive a python-capable command from an MCP stdio command."""
    tokens = shlex.split(command)
    if not tokens:
        return None
    if tokens[-1] == "graphknows-mcp":
        return " ".join(tokens[:-1] + ["python", "-"])
    return None


async def inspect_session(inspect_command: str, session_id: str) -> dict[str, Any]:
    """Run the container-side STM inspector and return the parsed snapshot."""
    proc = await asyncio.create_subprocess_exec(
        *shlex.split(inspect_command),
        session_id,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    assert proc.stdin is not None
    stdout, stderr = await proc.communicate(_INSPECTOR_SCRIPT.encode("utf-8"))
    if proc.returncode != 0:
        raise RuntimeError(stderr.decode("utf-8", errors="replace").strip() or "STM inspection failed")
    output = stdout.decode("utf-8", errors="replace").strip().splitlines()
    if not output:
        raise RuntimeError("STM inspection returned no output")
    return json.loads(output[-1])

"""Backend connectivity health checks."""

from __future__ import annotations

import subprocess  # nosec

from graphknows.config import GraphKnowsSettings


def get_version() -> str:
    """Return current git commit SHA (short)."""
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()[:12]  # nosec
    except Exception:
        return "unknown"


async def check_backends(settings: GraphKnowsSettings) -> dict[str, object]:
    """Check connectivity to all configured backends.

    Returns dict with keys: version, neo4j, memgraph, pgvector.
    Values are True (ok), False (fail), or error string.
    Does NOT propagate exceptions.
    """
    result: dict[str, object] = {"version": get_version()}

    try:
        from neo4j import GraphDatabase as _GDB
    except ImportError:
        result["neo4j"] = False
        result["memgraph"] = False
        _GDB = None  # type: ignore[assignment]

    if _GDB is not None:
        # neo4j
        try:
            driver = _GDB.driver(
                settings.neo4j_uri,
                auth=(settings.neo4j_user, settings.neo4j_password),
            )
            driver.verify_connectivity()
            driver.close()
            result["neo4j"] = True
        except Exception:
            result["neo4j"] = False

        # memgraph (bolt, no auth by default)
        try:
            driver = _GDB.driver(settings.memgraph_uri)
            driver.verify_connectivity()
            driver.close()
            result["memgraph"] = True
        except Exception:
            result["memgraph"] = False

    # pgvector
    try:
        import asyncpg

        conn = await asyncpg.connect(settings.pgvector_dsn)
        await conn.close()
        result["pgvector"] = True
    except Exception:
        result["pgvector"] = False

    return result


async def get_memory_stats(settings: GraphKnowsSettings, session_id: str = "") -> dict[str, object]:
    """Return STM and LTM memory counts.

    Args:
        settings: Runtime configuration.
        session_id: Optional session filter. Empty string returns aggregate counts.

    Returns:
        Dict with ``stm`` and ``ltm`` sub-dicts containing row counts.
    """
    import asyncpg
    from neo4j import GraphDatabase

    async def _pg_counts() -> dict[str, int]:
        conn = await asyncpg.connect(settings.pgvector_dsn)
        try:
            if session_id:
                sc = await conn.fetchval(
                    "SELECT count(*) FROM stm_chunks WHERE session_id = $1", session_id
                )
                st = await conn.fetchval(
                    "SELECT count(*) FROM stm_turns WHERE session_id = $1", session_id
                )
                lc = await conn.fetchval(
                    "SELECT count(*) FROM ltm_chunks WHERE session_id = $1", session_id
                )
                lt = await conn.fetchval(
                    "SELECT count(*) FROM ltm_turns WHERE session_id = $1", session_id
                )
                top = await conn.fetchval(
                    """
                    SELECT count(*)
                    FROM ltm_topics
                    WHERE $1 = ANY(COALESCE(session_ids, ARRAY[]::text[]))
                    """,
                    session_id,
                )
            else:
                sc = await conn.fetchval("SELECT count(*) FROM stm_chunks")
                st = await conn.fetchval("SELECT count(*) FROM stm_turns")
                lc = await conn.fetchval("SELECT count(*) FROM ltm_chunks")
                lt = await conn.fetchval("SELECT count(*) FROM ltm_turns")
                top = await conn.fetchval("SELECT count(*) FROM ltm_topics")
            return {
                "stm_chunks": int(sc),
                "stm_turns": int(st),
                "ltm_chunks": int(lc),
                "ltm_turns": int(lt),
                "ltm_topics": int(top),
            }
        finally:
            await conn.close()

    pg = await _pg_counts()

    drv = GraphDatabase.driver(  # nosec
        settings.neo4j_uri, auth=(settings.neo4j_user, settings.neo4j_password)
    )
    with drv.session() as s:
        if session_id:
            row_e = s.run(
                """
                MATCH (sess:SESSION {id: $sid})-[:IN_SESSION]->()-[*0..]->(e:ENTITY)
                RETURN count(DISTINCT e) as c
                """,
                sid=session_id,
            ).single()
            row_t = s.run(
                """
                MATCH (sess:SESSION {id: $sid})-[:IN_SESSION]->()-[*0..]->(e:ENTITY)
                OPTIONAL MATCH (e)-[r]->(t)
                WHERE t IS NOT NULL
                  AND 'TOPIC' IN labels(t)
                  AND type(r) IN ['IN_TOPIC', 'IN_COMMUNITY']
                RETURN count(DISTINCT t) as c
                """,
                sid=session_id,
            ).single()
            row_s = s.run(
                "MATCH (s:SESSION {id: $sid}) RETURN count(s) as c",
                sid=session_id,
            ).single()
        else:
            row_e = s.run("MATCH (e:ENTITY) RETURN count(e) as c").single()
            row_t = s.run("MATCH (t:TOPIC) RETURN count(t) as c").single()
            row_s = s.run("MATCH (s:SESSION) RETURN count(s) as c").single()
        ec = row_e["c"] if row_e else 0
        tc = row_t["c"] if row_t else 0
        sess_c = row_s["c"] if row_s else 0
    drv.close()

    return {
        "stm": {
            "chunks": pg["stm_chunks"],
            "turns": pg["stm_turns"],
        },
        "ltm": {
            "sessions": int(sess_c),
            "entities": int(ec),
            "topics": int(tc),
            "chunks": pg["ltm_chunks"],
            "turns": pg["ltm_turns"],
        },
    }

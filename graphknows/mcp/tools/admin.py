"""MCP tools: admin — memory_doctor, memory_stats, memory_purge."""

from __future__ import annotations

import asyncio
import os
from typing import Any

from graphknows.mcp._app import app
from graphknows.mcp._state import get_mg_store, get_neo_store, get_pg_store, get_settings


@app.tool()
async def memory_doctor() -> dict[str, Any]:
    """Check graphknows backend connectivity (Memgraph, Neo4j, pgvector).

    Returns a dict with boolean values for each backend, plus version and pid.
    """
    from graphknows.config.health import check_backends

    result = await check_backends(get_settings())
    result["pid"] = os.getpid()
    return result


@app.tool()
async def memory_stats(session_id: str = "") -> dict[str, Any]:
    """Return graphknows memory statistics (STM and LTM counts).

    Args:
        session_id: Optional session to filter STM stats by.  Empty string
            returns aggregate counts.

    Returns:
        Dict with ``stm`` and ``ltm`` sub-dicts containing row counts.
    """
    from graphknows.config.health import get_memory_stats

    return await get_memory_stats(get_settings(), session_id=session_id)


async def _purge_stm(session_id: str, errors: list[str]) -> None:
    """Delete STM data from Memgraph and pgvector."""
    try:
        mg_store = get_mg_store()
        if session_id:
            await mg_store.delete_session(session_id)
        else:

            def _mg_wipe() -> None:
                with mg_store._driver.session() as s:
                    s.run("MATCH (n) DETACH DELETE n")

            await asyncio.to_thread(_mg_wipe)
    except Exception as exc:
        errors.append(f"memgraph stm: {exc}")

    try:
        pg_store = await get_pg_store()
        if session_id:
            await pg_store.clear_stm(session_id)
        else:
            pool = pg_store._pool_or_raise()
            async with pool.acquire() as conn:
                await conn.execute("TRUNCATE TABLE stm_chunks, stm_turns")
    except Exception as exc:
        errors.append(f"pgvector stm: {exc}")


async def _purge_ltm(session_id: str, errors: list[str]) -> None:
    """Delete LTM data from Neo4j and pgvector."""
    try:
        neo_store = get_neo_store()
        if session_id:

            def _neo_delete() -> None:
                with neo_store._driver.session(database=neo_store._db) as s:
                    s.run(
                        """
                        MATCH (sess:SESSION {id: $sid})-[:IN_SESSION]->(f:FILE)
                        OPTIONAL MATCH (f)-[:HAS_CHUNK]->(c:CHUNK)
                        WITH f, collect(DISTINCT c) AS chunks
                        FOREACH (chunk IN chunks | DETACH DELETE chunk)
                        DETACH DELETE f
                        """,
                        sid=session_id,
                    )
                    s.run(
                        """
                        MATCH (sess:SESSION {id: $sid})-[:IN_SESSION]->(d:DIALOG)
                        OPTIONAL MATCH (d)-[:HAS_TURN]->(t:TURN)
                        WITH d, collect(DISTINCT t) AS turns
                        FOREACH (turn IN turns | DETACH DELETE turn)
                        DETACH DELETE d
                        """,
                        sid=session_id,
                    )
                    s.run(
                        """
                        MATCH (sess:SESSION {id: $sid})
                        DETACH DELETE sess
                        """,
                        sid=session_id,
                    )
                    s.run(
                        "MATCH (n:SessionEntity {session_id: $sid}) DETACH DELETE n",
                        sid=session_id,
                    )
                    s.run(
                        "MATCH ()-[r {session_id: $sid}]-() DELETE r",
                        sid=session_id,
                    )
                    s.run(
                        """
                        MATCH (e:ENTITY)
                        WHERE NOT (e)--()
                        DETACH DELETE e
                        """
                    )
                    s.run(
                        """
                        MATCH (t:TOPIC)
                        WHERE NOT (t)<-[:IN_TOPIC]-(:ENTITY)
                          AND NOT (t)<-[:IN_COMMUNITY]-(:ENTITY)
                        DETACH DELETE t
                        """
                    )

            await asyncio.to_thread(_neo_delete)
        else:

            def _neo_wipe() -> None:
                with neo_store._driver.session(database=neo_store._db) as s:
                    s.run("MATCH (n) DETACH DELETE n")

            await asyncio.to_thread(_neo_wipe)
    except Exception as exc:
        errors.append(f"neo4j ltm: {exc}")

    try:
        pg_store = await get_pg_store()
        pool = pg_store._pool_or_raise()
        async with pool.acquire() as conn:
            if session_id:
                await conn.execute("DELETE FROM ltm_chunks WHERE session_id = $1", session_id)
                await conn.execute("DELETE FROM ltm_turns WHERE session_id = $1", session_id)
                await conn.execute(
                    "DELETE FROM ltm_topics WHERE $1 = ANY(COALESCE(session_ids, ARRAY[]::text[]))",
                    session_id,
                )
                has_legacy_chunks = await conn.fetchval(
                    "SELECT to_regclass('public.chunks') IS NOT NULL"
                )
                if has_legacy_chunks:
                    await conn.execute("DELETE FROM chunks WHERE session_id = $1", session_id)
            else:
                await conn.execute(
                    "TRUNCATE TABLE ltm_chunks, ltm_turns, ltm_topics, chunks, chunk_entities, entities"
                )
    except Exception as exc:
        errors.append(f"pgvector ltm: {exc}")


@app.tool()
async def memory_purge(
    scope: str = "both",
    session_id: str = "",
) -> dict[str, Any]:
    """Purge graphknows memory — wipe STM and/or LTM across all backends.

    WARNING: This is irreversible. All matching data is permanently deleted.

    Args:
        scope: Which tier to purge.
               ``"stm"`` deletes Memgraph nodes + pgvector stm_chunks/stm_turns.
               ``"ltm"`` deletes Neo4j graph + pgvector ltm_chunks/ltm_turns/ltm_topics/chunks/chunk_entities/entities.
               ``"both"`` (default) purges both tiers.
        session_id: When provided, restricts deletion to that session only.
                    Empty string (default) deletes ALL sessions.

    Returns:
        Dict with keys: scope, session_id, stm_deleted, ltm_deleted, errors.
    """
    errors: list[str] = []
    stm_deleted = False
    ltm_deleted = False

    if scope in ("stm", "both"):
        await _purge_stm(session_id, errors)
        stm_deleted = True

    if scope in ("ltm", "both"):
        await _purge_ltm(session_id, errors)
        ltm_deleted = True

    return {
        "scope": scope,
        "session_id": session_id or "(all)",
        "stm_deleted": stm_deleted,
        "ltm_deleted": ltm_deleted,
        "errors": errors,
    }

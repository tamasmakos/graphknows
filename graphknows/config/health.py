"""Backend connectivity health checks."""

from __future__ import annotations

import subprocess  # nosec
from typing import Any

from graphknows.config import GraphKnowsSettings


def get_version() -> str:
    """Return current git commit SHA (short)."""
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()[:12]  # nosec
    except Exception:
        return "unknown"


async def check_backends(settings: GraphKnowsSettings) -> dict[str, object]:
    """Check connectivity to ArcadeDB.

    Returns dict with keys: version, arcadedb_stm, arcadedb_ltm.
    Values are True (ok), False (fail), or error string.
    Does NOT propagate exceptions.
    """
    result: dict[str, object] = {"version": get_version()}

    try:
        from graphknows.memory.stores.arcadedb.client import ArcadeDBClient

        c = ArcadeDBClient(
            base_url=settings.arcadedb_url,
            user=settings.arcadedb_user,
            password=settings.arcadedb_password,
        )
        await c.connect()
        result["arcadedb_stm"] = await c.database_exists("stm")
        result["arcadedb_ltm"] = await c.database_exists("ltm")
        await c.close()
    except Exception as exc:
        result["arcadedb_stm"] = str(exc)
        result["arcadedb_ltm"] = str(exc)

    return result


async def get_memory_stats(settings: GraphKnowsSettings, session_id: str = "") -> dict[str, object]:
    """Return STM and LTM memory counts from ArcadeDB."""
    from graphknows.memory.stores.arcadedb.client import ArcadeDBClient

    c = ArcadeDBClient(
        base_url=settings.arcadedb_url,
        user=settings.arcadedb_user,
        password=settings.arcadedb_password,
    )
    await c.connect()
    try:
        if session_id:
            stm_chunks = await c.query(
                "stm",
                "MATCH (n:CHUNK {session_id: $sid}) RETURN count(n) AS c",
                params={"sid": session_id},
            )
            stm_turns = await c.query(
                "stm",
                "MATCH (n:TURN {session_id: $sid}) RETURN count(n) AS c",
                params={"sid": session_id},
            )
            ltm_chunks = await c.query(
                "ltm",
                "MATCH (n:CHUNK {session_id: $sid}) RETURN count(n) AS c",
                params={"sid": session_id},
            )
            ltm_turns = await c.query(
                "ltm",
                "MATCH (n:TURN {session_id: $sid}) RETURN count(n) AS c",
                params={"sid": session_id},
            )
        else:
            stm_chunks = await c.query("stm", "MATCH (n:CHUNK) RETURN count(n) AS c")
            stm_turns = await c.query("stm", "MATCH (n:TURN) RETURN count(n) AS c")
            ltm_chunks = await c.query("ltm", "MATCH (n:CHUNK) RETURN count(n) AS c")
            ltm_turns = await c.query("ltm", "MATCH (n:TURN) RETURN count(n) AS c")

        ltm_entities = await c.query("ltm", "MATCH (n:ENTITY) RETURN count(n) AS c")
        ltm_topics = await c.query("ltm", "MATCH (n:TOPIC) RETURN count(n) AS c")
        ltm_sessions = await c.query("ltm", "MATCH (n:SESSION) RETURN count(n) AS c")

        def _n(rows: list[dict[str, Any]]) -> int:
            return int(rows[0].get("c") or 0) if rows else 0

        return {
            "stm": {
                "chunks": _n(stm_chunks),
                "turns": _n(stm_turns),
            },
            "ltm": {
                "sessions": _n(ltm_sessions),
                "entities": _n(ltm_entities),
                "topics": _n(ltm_topics),
                "chunks": _n(ltm_chunks),
                "turns": _n(ltm_turns),
            },
        }
    finally:
        await c.close()

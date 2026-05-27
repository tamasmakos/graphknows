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

    # neo4j
    try:
        from neo4j import GraphDatabase

        driver = GraphDatabase.driver(
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
        from neo4j import GraphDatabase

        driver = GraphDatabase.driver(settings.memgraph_uri)
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

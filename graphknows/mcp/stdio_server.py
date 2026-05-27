r"""Standalone MCP stdio server for graphknows memory operations.

Exposes 13 tools over JSON-RPC 2.0 via stdin/stdout so any MCP-aware client
(Claude Desktop, Cursor, Zed, Hermes, OpenCode) can use graphknows memory
without requiring the full agent service.

Tools
-----
STM tools:
  memory_doctor   — connectivity check: memgraph, neo4j, pgvector
  memory_stats    — STM and LTM node/chunk counts
  memory_ingest   — ingest text into STM (Memgraph + pgvector)
  memory_recall   — hybrid RRF recall from STM, LTM, or both
  memory_flush    — promote STM session → Neo4j LTM

LTM tools:
  ltm_query             — semantic topic search (pgvector)
  ltm_entity            — named entity lookup in Neo4j LTM
  ltm_topics            — list topics by entity threshold
  recall_by_document    — document chunks by entity match
  recall_by_dialog      — conversation utterances by entity match
  recall_by_topic       — topic matching by entity relevance

Corpus:
  corpus_ingest   — batch document ingestion into LTM knowledge graph

Usage (stdio probe)::

    echo '{"jsonrpc":"2.0","id":1,"method":"tools/list","params":{}}' \
      | graphknows-mcp 2>/dev/null

All MCP protocol traffic goes to stdout. All logging goes to stderr.
"""

from __future__ import annotations

import asyncio
import logging

logging.basicConfig(level=logging.WARNING, stream=__import__("sys").stderr)

# --- Import tool modules (registers @app.tool() decorators) ---
import graphknows.mcp.tools.admin  # noqa: E402
import graphknows.mcp.tools.corpus  # noqa: E402
import graphknows.mcp.tools.ltm  # noqa: E402
import graphknows.mcp.tools.recall  # noqa: E402
import graphknows.mcp.tools.stm  # noqa: F401, E402
from graphknows.mcp._app import app  # noqa: E402

# Re-export tool functions so existing code/tests can reference them on this module
from graphknows.mcp.tools.admin import memory_doctor, memory_stats  # noqa: F401, E402
from graphknows.mcp.tools.corpus import corpus_ingest  # noqa: F401, E402
from graphknows.mcp.tools.ltm import ltm_entity, ltm_query, ltm_topics  # noqa: F401, E402
from graphknows.mcp.tools.recall import (  # noqa: F401, E402
    recall_by_dialog,
    recall_by_document,
    recall_by_topic,
)
from graphknows.mcp.tools.stm import memory_flush, memory_ingest, memory_recall  # noqa: F401, E402

# ---------------------------------------------------------------------------
# Backward-compat shims — kept for existing test patches
# ---------------------------------------------------------------------------
_mg: object = None
_pg: object = None
_pg_connected: bool = False


def _get_settings() -> object:  # type: ignore[return]
    from graphknows.mcp._state import get_settings

    return get_settings()


def _mg_uri() -> str:
    import os

    return os.environ.get("MEMGRAPH_URI", "bolt://localhost:17687")


def _pg_dsn() -> str:
    import os

    return os.environ.get("PGVECTOR_DSN", "postgresql://localhost:5432/graphknows")


def _neo4j_creds() -> tuple[str, str, str]:
    import os

    return (
        os.environ.get("NEO4J_URI", "bolt://localhost:7687"),
        os.environ.get("NEO4J_USER", "neo4j"),
        os.environ.get("NEO4J_PASSWORD", ""),
    )


def _embed_fn() -> object:
    from graphknows.retrieval.stores.embedder import embed_one

    return embed_one


def _get_mg() -> object:
    global _mg
    if _mg is None:
        from graphknows.mcp._state import get_mg_store

        _mg = get_mg_store()
    return _mg


async def _get_pg() -> object:
    global _pg, _pg_connected
    if _pg is None or not _pg_connected:
        from graphknows.mcp._state import get_pg_store

        _pg = await get_pg_store()
        _pg_connected = True
    return _pg


def _get_neo() -> object:
    from graphknows.mcp._state import get_neo_store

    return get_neo_store()


# Keep private helper accessible for tests that import it
from graphknows.mcp.tools.stm import _normalise_and_weight  # noqa: F401, E402


def main() -> None:
    """Run the graphknows MCP stdio server.

    Reads JSON-RPC 2.0 requests on stdin, writes responses on stdout.
    All logging goes to stderr.
    """
    asyncio.run(app.run_stdio_async())


if __name__ == "__main__":
    main()

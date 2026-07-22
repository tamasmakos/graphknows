r"""Standalone MCP stdio server for graphknows memory operations.

Exposes tools over JSON-RPC 2.0 via stdin/stdout so any MCP-aware client
(Claude Desktop, Cursor, Zed, Hermes, OpenCode) can use graphknows memory
without requiring the full agent service.

Tools
-----
STM tools:
  memory_ingest         — ingest text into STM (ArcadeDB)
  memory_flush          — promote STM session → ArcadeDB LTM
  memory_recall         — deprecated alias of memory_query

LTM tools:
  ltm_topics            — list topics by entity threshold
  ltm_entity            — one entity with its neighbourhood
  ltm_entities          — list entities
  ltm_query             — entity-scoped LTM lookup

Retrieval:
  memory_query          — unified retrieval; strategy=det|agentic per call

Analysis:
  memory_topics         — rebuild the topic layer at new settings (no re-ingest)

Admin:
  memory_doctor         — connectivity check: ArcadeDB STM, LTM, vector
  memory_stats          — STM and LTM node/chunk counts
  memory_purge          — purge STM and/or LTM within one namespace
  memory_drop_namespace — DESTRUCTIVE: drop a tenant's whole database pair

Corpus:
  corpus_ingest         — batch document ingestion into LTM knowledge graph

Web:
  memory_mirror         — return the local Memory Mirror browser URL

Usage (stdio probe)::

    echo '{"jsonrpc":"2.0","id":1,"method":"tools/list","params":{}}' \
      | graphknows-mcp 2>/dev/null

All MCP protocol traffic goes to stdout. All logging goes to stderr.
"""

from __future__ import annotations

import asyncio
import logging

logging.basicConfig(level=logging.WARNING, stream=__import__("sys").stderr)

from graphknows.server.mcp._app import app  # noqa: E402

# Importing the tools package registers every tool (single registration point);
# the star-import re-exports them here so existing code/tests can reference them
# on this module. The package's __all__ is the authoritative inventory.
from graphknows.server.mcp.tools import *  # noqa: F403, E402


async def _startup_warmup() -> None:
    """Pre-warm embedder and stores before first tool call."""
    log = logging.getLogger("graphknows.startup")

    # Load settings first so configuration is validated before the first tool call.
    try:
        from graphknows.server.mcp._state import get_settings

        get_settings()
        log.warning("Settings warmup OK")
    except Exception as exc:
        log.warning("Settings warmup failed (non-fatal): %s", exc)

    # 1. Probe embedder (validates API key + network reachability)
    try:
        from graphknows.storage.embedder import embed_one

        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, embed_one, "warmup")
        log.warning("Embedder warmup OK")
    except Exception as exc:
        log.warning("Embedder warmup failed (non-fatal): %s", exc)


async def _run_server() -> None:
    await _startup_warmup()
    await app.run_stdio_async()


def main() -> None:
    """Run the graphknows MCP stdio server.

    Reads JSON-RPC 2.0 requests on stdin, writes responses on stdout.
    All logging goes to stderr.
    """
    from graphknows.settings import GraphKnowsSettings

    GraphKnowsSettings().check_production_secrets()
    asyncio.run(_run_server())


if __name__ == "__main__":
    main()

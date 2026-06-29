r"""Standalone MCP stdio server for graphknows memory operations.

Exposes tools over JSON-RPC 2.0 via stdin/stdout so any MCP-aware client
(Claude Desktop, Cursor, Zed, Hermes, OpenCode) can use graphknows memory
without requiring the full agent service.

Tools
-----
STM tools:
  memory_doctor   — connectivity check: ArcadeDB STM, LTM, vector
  memory_stats    — STM and LTM node/chunk counts
  memory_ingest   — ingest text into STM (ArcadeDB)
  memory_flush    — promote STM session → ArcadeDB LTM
  memory_mirror   — return the local Memory Mirror browser URL

LTM tools:
  ltm_topics      — list topics by entity threshold

Retrieval:
  memory_query    — unified retrieval (deterministic or agentic, driven by MODE)

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
import graphknows.mcp.tools.stm  # noqa: E402
import graphknows.mcp.tools.web  # noqa: F401, E402
from graphknows.mcp._app import app  # noqa: E402

# Re-export tool functions so existing code/tests can reference them on this module
from graphknows.mcp.tools.admin import memory_doctor, memory_purge, memory_stats  # noqa: F401, E402
from graphknows.mcp.tools.corpus import corpus_ingest  # noqa: F401, E402
from graphknows.mcp.tools.ltm import (  # noqa: F401, E402
    ltm_entities,
    ltm_entity,
    ltm_query,
    ltm_topics,
)
from graphknows.mcp.tools.query import memory_query  # noqa: F401, E402
from graphknows.mcp.tools.stm import memory_flush, memory_ingest, memory_recall  # noqa: F401, E402
from graphknows.mcp.tools.web import memory_mirror  # noqa: F401, E402

# Re-export private helper so tests can import it from this module.
from graphknows.runtime import normalise_and_weight as _normalise_and_weight  # noqa: F401, E402


async def _startup_warmup() -> None:
    """Pre-warm embedder and stores before first tool call."""
    log = logging.getLogger("graphknows.startup")

    # Load settings first so repo .env values are promoted to os.environ before
    # embedder code reads OPENROUTER_API_KEY directly.
    try:
        from graphknows.mcp._state import get_settings

        get_settings()
        log.warning("Settings warmup OK")
    except Exception as exc:
        log.warning("Settings warmup failed (non-fatal): %s", exc)

    # 1. Probe embedder (validates API key + network reachability)
    try:
        from graphknows.memory.stores.embedder import embed_one

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
    asyncio.run(_run_server())


if __name__ == "__main__":
    main()

r"""Standalone MCP stdio server for graphknows memory operations.

Exposes tools over JSON-RPC 2.0 via stdin/stdout so any MCP-aware client
(Claude Desktop, Cursor, Zed, Hermes, OpenCode) can use graphknows memory
without requiring the full agent service.

``graphknows.server.mcp.tools.__all__`` is the authoritative tool inventory.

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

# The destructive tools leave __all__ when the admin gate is shut, so name them
# explicitly: the Python surface always carries them, only the MCP surface is
# gated.
from graphknows.server.mcp.tools.admin import (  # noqa: E402, F401
    ENABLED_ADMIN_TOOLS,
    memory_drop_namespace,
    memory_purge,
)


def _announce_admin_tools() -> None:
    """Name an enabled destructive surface on stderr — it is never invisible."""
    if ENABLED_ADMIN_TOOLS:
        logging.getLogger("graphknows.startup").warning(
            "admin tools ENABLED: %s", ", ".join(ENABLED_ADMIN_TOOLS)
        )


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
    from graphknows.settings import get_settings

    get_settings().check_production_secrets()
    _announce_admin_tools()
    asyncio.run(_run_server())


if __name__ == "__main__":
    main()

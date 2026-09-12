"""MCP tool package — the single registration point for every tool.

Importing this package imports every tool submodule, which runs the
``@app.tool()`` decorators that register them on the shared FastMCP app.
``__all__`` is the authoritative inventory of the registered tools.
"""

from __future__ import annotations

from graphknows.server.mcp.tools.admin import (
    ENABLED_ADMIN_TOOLS,
    memory_doctor,
    memory_stats,
)

# Explicit re-exports: the two destructive tools are bound here whether or not
# the admin gate is open, so that ``__all__`` can name them when it is.
from graphknows.server.mcp.tools.admin import (
    memory_drop_namespace as memory_drop_namespace,
)
from graphknows.server.mcp.tools.admin import (
    memory_purge as memory_purge,
)
from graphknows.server.mcp.tools.corpus import corpus_ingest
from graphknows.server.mcp.tools.ltm import ltm_entities, ltm_entity, memory_forget
from graphknows.server.mcp.tools.query import memory_query
from graphknows.server.mcp.tools.recall import memory_recall
from graphknows.server.mcp.tools.stm import memory_flush, memory_ingest

__all__ = [
    "corpus_ingest",
    "ltm_entities",
    "ltm_entity",
    "memory_doctor",
    "memory_flush",
    "memory_forget",
    "memory_ingest",
    "memory_query",
    "memory_recall",
    "memory_stats",
]

# The destructive tools join the inventory only when they were registered, so
# ``__all__`` keeps describing exactly what a client can list and call.
__all__ += list(ENABLED_ADMIN_TOOLS)

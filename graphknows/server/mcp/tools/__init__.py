"""MCP tool package — the single registration point for every tool.

Importing this package imports every tool submodule, which runs the
``@app.tool()`` decorators that register them on the shared FastMCP app.
``__all__`` is the authoritative inventory of the registered tools.
"""

from __future__ import annotations

from graphknows.server.mcp.tools.admin import (
    memory_doctor,
    memory_drop_namespace,
    memory_purge,
    memory_stats,
)
from graphknows.server.mcp.tools.corpus import corpus_ingest
from graphknows.server.mcp.tools.ltm import ltm_entities, ltm_entity, ltm_topics
from graphknows.server.mcp.tools.query import memory_query
from graphknows.server.mcp.tools.stm import memory_flush, memory_ingest
from graphknows.server.mcp.tools.topics import memory_topics

__all__ = [
    "corpus_ingest",
    "ltm_entities",
    "ltm_entity",
    "ltm_topics",
    "memory_doctor",
    "memory_drop_namespace",
    "memory_flush",
    "memory_ingest",
    "memory_purge",
    "memory_query",
    "memory_stats",
    "memory_topics",
]

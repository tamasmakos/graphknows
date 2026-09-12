"""Single FastMCP application instance shared by all tool modules."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

try:
    from mcp.server.fastmcp import FastMCP
except ModuleNotFoundError as exc:  # pragma: no cover - exercised only on broken installs
    from graphknows.exceptions import BrokenInstallError

    raise BrokenInstallError("The MCP stdio server", "mcp") from exc


@asynccontextmanager
async def _lifespan(app: FastMCP) -> AsyncIterator[None]:
    yield
    from graphknows.server.mcp._state import close

    await close()


app = FastMCP(
    "graphknows-memory",
    instructions=(
        "graphknows-memory exposes short-term and long-term knowledge-graph "
        "backed memory. Use memory_ingest to store content, memory_query to "
        "retrieve it (unified retrieval: deterministic or agentic, driven by MODE), "
        "memory_flush to promote session memory to long-term, "
        "memory_doctor to verify backend connectivity, and memory_stats for counts. "
        "Use corpus_ingest for bulk document ingestion into the LTM knowledge graph."
    ),
    lifespan=_lifespan,
)

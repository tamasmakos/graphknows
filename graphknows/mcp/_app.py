"""Single FastMCP application instance shared by all tool modules."""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

app = FastMCP(
    "graphknows-memory",
    instructions=(
        "graphknows-memory exposes short-term and long-term knowledge-graph "
        "backed memory. Use memory_ingest to store content, memory_query to "
        "retrieve it (unified retrieval: deterministic or agentic via RETRIEVAL_MODE), "
        "memory_flush to promote session memory to long-term, "
        "memory_doctor to verify backend connectivity, and memory_stats for counts. "
        "Use memory_mirror when the user asks for the Memory Mirror link or a browser "
        "URL for the local graph view. "
        "Use ltm_topics to browse topic clusters by entity threshold. "
        "Use corpus_ingest for bulk document ingestion into the LTM knowledge graph."
    ),
)

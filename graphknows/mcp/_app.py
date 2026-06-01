"""Single FastMCP application instance shared by all tool modules."""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

app = FastMCP(
    "graphknows-memory",
    instructions=(
        "graphknows-memory exposes short-term and long-term knowledge-graph "
        "backed memory. Use memory_ingest to store content, memory_recall to "
        "retrieve it, memory_flush to promote session memory to long-term, "
        "memory_doctor to verify backend connectivity, and memory_stats for counts. "
        "Use memory_mirror when the user asks for the Memory Mirror link or a browser "
        "URL for the local graph view. "
        "For LTM exploration: ltm_query searches topics semantically, "
        "ltm_entity looks up named entities, ltm_topics lists topics by threshold. "
        "recall_by_document, recall_by_dialog, and recall_by_topic provide "
        "entity-matched retrieval across documents, conversations, and topics."
    ),
)

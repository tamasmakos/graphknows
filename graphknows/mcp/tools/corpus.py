"""MCP tools: corpus — corpus_ingest."""

from __future__ import annotations

from typing import Any

from graphknows.mcp._app import app
from graphknows.mcp._state import get_settings


@app.tool()
async def corpus_ingest(
    documents: list[dict[str, Any]],
    session_id: str | None = None,
) -> dict[str, Any]:
    r"""Ingest a corpus of documents into the long-term knowledge graph.

    Runs the STM-first corpus pipeline via `CorpusIngestWorkflow`: document ingest
    into short-term memory, then promotion into the long-term graph. Topic
    summarization and GDS enrichment always run so LTM topics stay available.

    Args:
        documents: List of document dicts. Each must have:
            - text (str): document content
            - title (str): human-readable title
            - id (str): unique document identifier
            - source (str, optional): provenance URL or label
        session_id: Optional session identifier for idempotent retries and cleanup.

    Returns:
        Dict with pipeline statistics: documents_processed,
        entities_extracted, chunks_written, nodes_written,
        edges_written, elapsed_s, errors (list).
    """
    if not documents:
        return {"documents_processed": 0, "error": "no documents provided"}
    try:
        from graphknows.config.builders.stores import build_corpus_workflow
        from graphknows.memory.workflows.corpus_ingest import CorpusIngestRequest

        s = get_settings()
        workflow = build_corpus_workflow(s)
        request = CorpusIngestRequest(
            documents=documents,
            session_id=session_id,
            summarize_topics=True,
            run_gds=True,
        )
        result = await workflow.run(request)
    except Exception as exc:
        return {"error": str(exc), "documents_processed": 0}
    else:
        return {
            "session_id": result.session_id,
            "documents_processed": result.documents_processed,
            "entities_extracted": result.entities_extracted,
            "chunks_written": result.chunks_written,
            "nodes_written": result.nodes_written,
            "edges_written": result.edges_written,
            "elapsed_s": result.elapsed_s,
            "errors": result.errors,
        }

"""MCP tools: corpus — corpus_ingest."""

from __future__ import annotations

from typing import Any

from graphknows.mcp._app import app
from graphknows.mcp._state import get_settings


@app.tool()
async def corpus_ingest(
    documents: list[dict[str, Any]],
    taxonomy_file: str | None = None,
) -> dict[str, Any]:
    r"""Ingest a corpus of documents into the long-term knowledge graph.

    Runs the full graphgen pipeline: NER → relation extraction → graph
    cleaning → community detection → Neo4j write-back.  Use for batch
    knowledge ingestion.  GDS enrichment (Node2Vec + Louvain) runs
    automatically after write.

    Args:
        documents: List of document dicts. Each must have:
            - text (str): document content
            - title (str): human-readable title
            - id (str): unique document identifier
            - source (str, optional): provenance URL or label
        taxonomy_file: Optional path or name of taxonomy YAML to use.
            Defaults to taxonomy.generic.yaml.  Pass
            \"taxonomy.eu-politics.yaml\" for EU political domain.

    Returns:
        Dict with pipeline statistics: documents_processed,
        entities_extracted, relations_extracted, nodes_written,
        edges_written, errors (list).
    """
    if not documents:
        return {"documents_processed": 0, "error": "no documents provided"}
    try:
        from graphknows.builders.stores import build_corpus_workflow
        from graphknows.memory.workflows.corpus_ingest import CorpusIngestRequest

        s = get_settings()
        workflow = build_corpus_workflow(s)
        request = CorpusIngestRequest(
            documents=documents,
            taxonomy_file=taxonomy_file,
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

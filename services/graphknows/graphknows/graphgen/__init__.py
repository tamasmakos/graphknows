"""graphknows.graphgen — knowledge graph generation pipeline.

Absorbs the former top-level ``graphgen`` package.  The primary public entry
points are:

- :func:`run_ingestion` — async wrapper for the full pipeline (writes documents
  to a temp dir and calls :func:`graphknows.graphgen.orchestrator.run_pipeline`
  in a thread pool).
- :class:`~graphknows.graphgen.orchestrator.KnowledgePipeline` — the pipeline
  class itself, for programmatic use without HTTP.

Example:
    from graphknows.graphgen import run_ingestion

    stats = await run_ingestion(documents=[{"text": "...", "id": "d1", "source": "web"}])
"""

from graphknows.graphgen.ingest import run_ingestion

__all__ = ["run_ingestion"]

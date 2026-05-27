"""Thin async wrapper around the graphgen KnowledgePipeline for use by graphknows.

Exposes :func:`run_ingestion` — write documents to a temp dir and run the full
NLP + graph extraction + Neo4j upload pipeline in a background thread.

Example:
    from graphknows.graphgen import run_ingestion

    stats = await run_ingestion(documents=[{"text": "...", "title": "...", "id": "doc1", "source": "web"}])
"""

from __future__ import annotations

import asyncio
import logging
import tempfile
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


async def run_ingestion(
    documents: list[dict[str, Any]],
    config_override: dict[str, Any] | None = None,
    output_dir: str | None = None,
) -> dict[str, Any]:
    """Write *documents* to a temp dir and run the graphgen KnowledgePipeline.

    Runs in a dedicated thread pool executor so the FastAPI event loop is never
    blocked.  Forces ``infra.clean_start=False`` by default to prevent wiping
    a shared Neo4j corpus; callers may override this explicitly.

    Args:
        documents: List of dicts with at minimum a ``text`` key.
            Optional keys: ``id``, ``source``, ``title``.
        config_override: Flat or nested dict overriding PipelineSettings fields.
        output_dir: Pipeline artefact output dir; uses graphgen default when
            ``None``.

    Returns:
        Pipeline stats dict collected across all stages.

    Raises:
        RuntimeError: If ``graphknows.graphgen.orchestrator`` is not importable.
        ValueError: If ``documents`` is empty.
    """
    import warnings

    warnings.warn(
        "graphknows.graphgen.ingest.run_ingestion is deprecated and will be removed in Wave 4. "
        "Use graphknows.memory.workflows.corpus_ingest.CorpusIngestWorkflow instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    from graphknows.graphgen.orchestrator import run_pipeline

    if not documents:
        logger.warning("run_ingestion called with empty document list — skipping")
        return {"status": "skipped", "reason": "no documents"}

    def _write_and_run() -> dict[str, Any]:
        input_files: list[str] = []
        with tempfile.TemporaryDirectory(prefix="graphknows_ingest_") as tmpdir:
            for i, doc in enumerate(documents):
                doc_id = doc.get("id") or f"doc_{i:04d}"
                fname = Path(tmpdir) / f"{doc_id}.txt"
                title = doc.get("title", "")
                body = doc.get("text", "")
                fname.write_text(
                    f"{title}\n\n{body}" if title else body,
                    encoding="utf-8",
                )
                input_files.append(str(fname))

            logger.info("Starting pipeline for %d document(s)", len(input_files))

            safe_override: dict[str, Any] = dict(config_override or {})
            infra_section = dict(safe_override.get("infra", {}))
            infra_section.setdefault("clean_start", False)
            safe_override["infra"] = infra_section

            stats = asyncio.run(
                run_pipeline(
                    input_files=input_files,
                    config_override=safe_override,
                    output_dir=output_dir,
                )
            )
            logger.info("Pipeline completed. Stats: %s", stats)
            return stats or {}

    return await asyncio.to_thread(_write_and_run)

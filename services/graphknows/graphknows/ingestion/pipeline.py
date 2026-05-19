"""graphknows ingestion module — thin async wrapper around graphgen's KnowledgePipeline.

Exposes a single coroutine, run_ingestion(), that:
1. Writes document text to a temporary directory.
2. Calls graphgen's run_pipeline() programmatic entry point.
3. Returns the stats dict from the pipeline.

Graphgen's pipeline handles all NLP / embedding / Neo4j upload steps.
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

    Args:
        documents: List of dicts with at minimum a ``text`` key.
                   Optional keys: ``id``, ``source``, ``title``.
        config_override: Flat or nested dict overriding PipelineSettings fields.
        output_dir: Pipeline artefact output dir (uses graphgen default if None).

    Returns:
        Pipeline stats dict collected across all stages.
    """
    try:
        from graphknows.graphgen.orchestrator import run_pipeline
    except ImportError as exc:
        raise RuntimeError(
            "graphknows.graphgen is not importable — check your installation"
        ) from exc

    def _write_and_run() -> dict[str, Any]:
        """Synchronous portion: write files and call run_pipeline via asyncio.run."""
        input_files: list[str] = []

        with tempfile.TemporaryDirectory(prefix="graphknows_ingest_") as tmpdir:
            for i, doc in enumerate(documents):
                doc_id = doc.get("id") or f"doc_{i:04d}"
                fname = Path(tmpdir) / f"{doc_id}.txt"
                # Prepend title if present so the lexical graph captures it
                title = doc.get("title", "")
                body = doc.get("text", "")
                fname.write_text(f"{title}\n\n{body}" if title else body, encoding="utf-8")
                input_files.append(str(fname))

            if not input_files:
                logger.warning("run_ingestion called with empty document list")
                return {"status": "skipped", "reason": "no documents"}

            logger.info("Starting graphgen pipeline for %d document(s)", len(input_files))
            # Force clean_start=False so graphknows never wipes the shared Neo4j
            # database.  Individual callers may still request a clean start by
            # passing {"infra": {"clean_start": True}} in config_override, but
            # it must be explicit.
            safe_override: dict = dict(config_override or {})
            infra_section = dict(safe_override.get("infra", {}))
            infra_section.setdefault("clean_start", False)
            safe_override["infra"] = infra_section

            # run_pipeline is async — we are already in a thread, so use asyncio.run
            stats = asyncio.run(
                run_pipeline(
                    input_files=input_files,
                    config_override=safe_override,
                    output_dir=output_dir,
                )
            )
            logger.info("Pipeline completed. Stats: %s", stats)
            return stats or {}

    # Offload the blocking I/O + CPU-bound pipeline to a thread pool so the
    # FastAPI event loop is not blocked during ingestion.
    return await asyncio.to_thread(_write_and_run)

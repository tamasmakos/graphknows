"""Ingestion stage: ingest a dialogue's sessions, optionally flush to LTM.

Wraps the ``memory_ingest`` / ``memory_flush`` / ``ltm_entities`` MCP tools and
returns a typed :class:`IngestResult` aggregating chunk/entity/graph counts.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

from graphknows.client import GraphKnowsMCPClient
from evaluation.config import EvalConfig
from evaluation.datamodels import EvalDocument, IngestResult

log = logging.getLogger(__name__)


class Ingestor:
    """Ingests a dialogue's documents into one memory session."""

    def __init__(self, client: GraphKnowsMCPClient, config: EvalConfig) -> None:
        self._client = client
        self._config = config

    async def ingest(self, session_id: str, documents: list[EvalDocument]) -> IngestResult:
        """Ingest all documents (concurrency opt-in), then optionally flush to LTM.

        Default 1 (sequential): concurrent docs write the same session graph, and
        the ENTITY(name) UNIQUE index makes parallel MERGE of a shared entity name
        race — measured to degrade retrieval F1. Raise EVAL_INGEST_CONCURRENCY only
        once the STM write path serializes per-entity MERGEs.
        """
        cap = int(os.environ.get("EVAL_INGEST_CONCURRENCY", "1"))
        sem = asyncio.Semaphore(cap)

        async def _ingest_one(index: int, doc: EvalDocument) -> dict[str, Any]:
            async with sem:
                result = await self._client.call_tool(
                    "memory_ingest",
                    {
                        "text": doc.text,
                        "session_id": session_id,
                        "title": doc.title or f"{session_id}-doc-{index}",
                        "metadata": {"anchor_date": doc.anchor_date},
                    },
                )
            return result if isinstance(result, dict) else {"result": result}

        outputs: list[dict[str, Any]] = await asyncio.gather(
            *[_ingest_one(i, d) for i, d in enumerate(documents)]
        )

        ingest = IngestResult(
            documents=len(outputs),
            chunks=sum(int(o.get("chunks", 0)) for o in outputs),
            entities=sum(int(o.get("entities", 0)) for o in outputs),
            ingest_elapsed_s=round(sum(float(o.get("elapsed_s", 0)) for o in outputs), 3),
            ingest_error_count=sum(1 for o in outputs if "chunks" not in o),
        )

        if self._config.promote_to_ltm:
            await self._flush(session_id, ingest)
        return ingest

    async def _flush(self, session_id: str, ingest: IngestResult) -> None:
        """Flush the session to LTM and capture graph + entity counts in-place."""
        try:
            result = await self._client.call_tool("memory_flush", {"session_id": session_id})
            flush = result if isinstance(result, dict) else {"result": result}
        except Exception as exc:
            log.warning("flush failed for %s: %s", session_id, exc)
            ingest.flush_error_count = 1
            return

        ingest.flush_nodes = int(flush.get("nodes", 0))
        ingest.flush_edges = int(flush.get("edges", 0))
        ingest.flush_chunks = int(flush.get("chunks", 0))
        ingest.flush_topics = int(flush.get("topics_created", 0))
        ingest.flush_elapsed_s = round(float(flush.get("elapsed_s", 0)), 3)
        ingest.flush_error_count = len(flush.get("errors", []))

        try:
            entity_result = await self._client.call_tool(
                "ltm_entities", {"session_id": session_id, "limit": 500}
            )
            if isinstance(entity_result, dict):
                ingest.ltm_entity_count = len(entity_result.get("entities", []))
        except Exception:
            pass

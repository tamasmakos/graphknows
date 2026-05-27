"""CorpusIngestWorkflow: batch document → STM → LTM promotion pipeline."""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger("graphknows.memory.workflows.corpus_ingest")


@dataclass
class CorpusIngestRequest:
    """Request parameters for batch corpus ingestion."""

    documents: list[dict[str, Any]]
    taxonomy_file: str | None = None  # reserved for Wave 4; silently ignored
    session_id: str | None = None  # auto-generated if None; supply for idempotent retry
    run_gds: bool = False
    clear_stm: bool = True
    summarize_topics: bool = True


@dataclass
class CorpusIngestResult:
    """Result of a batch corpus ingestion run."""

    session_id: str
    documents_processed: int = 0
    entities_extracted: int = 0
    chunks_written: int = 0
    nodes_written: int = 0
    edges_written: int = 0
    elapsed_s: float = 0.0
    errors: list[str] = field(default_factory=list)


class CorpusIngestWorkflow:
    """Batch corpus ingestion via STM-first pipeline.

    Phase 1: Per-document STMService.ingest() into a single corpus session_id.
    Phase 2: SessionFlusher.flush() → Neo4j LTM promotion.
    """

    def __init__(
        self,
        mg_store: Any,
        pg_store: Any,
        neo_store: Any,
        extractor: Any,
        embed_fn: Any,
        openrouter_key: str = "",
        rel_extractor: Any = None,
    ) -> None:
        self._mg = mg_store
        self._pg = pg_store
        self._neo = neo_store
        self._extractor = extractor
        self._embed_fn = embed_fn
        self._openrouter_key = openrouter_key
        self._rel_extractor = rel_extractor

    async def run(self, request: CorpusIngestRequest) -> CorpusIngestResult:
        """Execute the STM-first corpus ingestion pipeline.

        Args:
            request: Batch ingestion parameters.

        Returns:
            CorpusIngestResult with aggregated counts and any errors.
        """
        # Ensure asyncpg pool is created on THIS event loop (not _BG_LOOP)
        if not getattr(self._pg, "_pool", None):
            await self._pg.connect()

        from graphknows.memory.flush import FlushOptions, SessionFlusher
        from graphknows.memory.stm.service import STMService

        t0 = time.monotonic()
        session_id = request.session_id or f"corpus-{uuid.uuid4().hex}"
        result = CorpusIngestResult(session_id=session_id)

        stm = STMService(
            mg_store=self._mg,
            pg_store=self._pg,
            extractor=self._extractor,
            embed_fn=self._embed_fn,
            rel_extractor=self._rel_extractor,
        )

        # Phase 1: ingest each document into STM
        for doc in request.documents:
            text = doc.get("text", "")
            if not text:
                result.errors.append(
                    f"Skipped document missing 'text': id={doc.get('id', '<unknown>')}"
                )
                log.warning("Skipping document without text: %s", doc.get("id", "<unknown>"))
                continue

            title = doc.get("title", "")
            source_hint = doc.get("source", "")

            try:
                ingest_result = stm.ingest(
                    text=text,
                    session_id=session_id,
                    title=title,
                    source_hint=source_hint,
                )
                result.documents_processed += 1
                result.entities_extracted += getattr(ingest_result, "entities", 0)
                result.chunks_written += getattr(ingest_result, "chunks", 0)
                log.debug(
                    "Ingested doc id=%s into session=%s", doc.get("id", "<unknown>"), session_id
                )
            except Exception as exc:
                err = f"STM ingest failed for doc id={doc.get('id', '<unknown>')}: {exc}"
                result.errors.append(err)
                log.warning(err)

        # Phase 2: flush STM → LTM
        flusher = SessionFlusher(
            mg_store=self._mg,
            pg_store=self._pg,
            neo_store=self._neo,
            openrouter_key=self._openrouter_key,
        )
        flush_opts = FlushOptions(
            summarize_topics=request.summarize_topics,
            run_gds=request.run_gds,
            clear_stm=request.clear_stm,
        )
        try:
            flush_result = await flusher.flush(session_id, options=flush_opts)
            result.nodes_written = flush_result.nodes_written
            result.edges_written = flush_result.edges_written
            result.errors.extend(flush_result.errors)
        except Exception as exc:
            err = f"SessionFlusher.flush failed: {exc}"
            result.errors.append(err)
            log.exception(err)

        result.elapsed_s = time.monotonic() - t0
        log.info(
            "CorpusIngestWorkflow session=%s docs=%d entities=%d chunks=%d "
            "nodes=%d edges=%d errors=%d elapsed=%.1fs",
            session_id,
            result.documents_processed,
            result.entities_extracted,
            result.chunks_written,
            result.nodes_written,
            result.edges_written,
            len(result.errors),
            result.elapsed_s,
        )
        return result

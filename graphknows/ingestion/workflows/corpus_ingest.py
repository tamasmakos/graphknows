"""CorpusIngestWorkflow: batch document → STM → LTM promotion pipeline."""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger("graphknows.ingestion.workflows.corpus_ingest")


@dataclass
class CorpusIngestRequest:
    """Request parameters for batch corpus ingestion."""

    documents: list[dict[str, Any]]
    session_id: str | None = None  # auto-generated if None; supply for idempotent retry
    run_gds: bool = True
    clear_stm: bool = True


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
    Phase 2: SessionFlusher.flush() → ArcadeDB LTM promotion.
    """

    def __init__(
        self,
        stm_store: Any,
        vec_store: Any,
        ltm_store: Any,
        extractor: Any,
        embed_fn: Any,
        llm_api_key: str = "",
        llm_api_base: str = "",
        rel_extractor: Any = None,
        extraction_uses_llm: bool = False,
        flush_vec_store: Any | None = None,
        channels: list[Any] | None = None,
    ) -> None:
        self._stm = stm_store
        self._vec = vec_store
        self._ltm = ltm_store
        self._extractor = extractor
        self._embed_fn = embed_fn
        self._llm_api_key = llm_api_key
        self._llm_api_base = llm_api_base
        self._rel_extractor = rel_extractor
        self._extraction_uses_llm = extraction_uses_llm
        self._flush_vec_store = flush_vec_store
        self._channels = channels or []

    async def run(self, request: CorpusIngestRequest) -> CorpusIngestResult:
        """Execute the STM-first corpus ingestion pipeline.

        Args:
            request: Batch ingestion parameters.

        Returns:
            CorpusIngestResult with aggregated counts and any errors.
        """
        from graphknows.ingestion.promotion.pipeline import LTMPromotionPipeline
        from graphknows.ingestion.models.promotion import PromotionRequest
        from graphknows.ingestion.stm.service import STMService

        t0 = time.monotonic()
        session_id = request.session_id or f"corpus-{uuid.uuid4().hex}"
        result = CorpusIngestResult(session_id=session_id)

        resolved_extractor = self._extractor
        resolved_rel_extractor = self._rel_extractor
        stm = STMService(
            stm_store=self._stm,
            vec_store=self._vec,
            extractor=resolved_extractor,
            embed_fn=self._embed_fn,
            rel_extractor=resolved_rel_extractor,
        )
        # Connect on the caller's event loop so the store client is bound to it.
        await stm.connect()

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
                ingest_result = await stm.ingest(
                    text=text,
                    session_id=session_id,
                    title=title,
                    source_hint=source_hint,
                )
                result.documents_processed += 1
                result.entities_extracted += getattr(ingest_result, "entity_count", 0)
                result.chunks_written += getattr(ingest_result, "chunk_count", 0)
                log.debug(
                    "Ingested doc id=%s into session=%s", doc.get("id", "<unknown>"), session_id
                )
            except Exception as exc:
                err = f"STM ingest failed for doc id={doc.get('id', '<unknown>')}: {exc}"
                result.errors.append(err)
                log.warning(err)

        # Phase 2: flush STM → LTM. All stores now share the caller's event loop,
        # so the flusher can reuse the ingest vector store; an explicit
        # flush_vec_store is still honoured when supplied.
        flush_vec = self._flush_vec_store or self._vec
        pipeline = LTMPromotionPipeline(
            stm_store=self._stm,
            vec_store=flush_vec,
            ltm_store=self._ltm,
            llm_api_key=self._llm_api_key,
            llm_api_base=self._llm_api_base,
            extraction_uses_llm=self._extraction_uses_llm,
            channels=self._channels,
        )
        try:
            flush_result = await pipeline.promote(
                PromotionRequest(
                    session_id=session_id,
                    run_gds=request.run_gds,
                    clear_stm=request.clear_stm,
                )
            )
            result.nodes_written = flush_result.nodes_written
            result.edges_written = flush_result.edges_written
            result.errors.extend(flush_result.errors)
        except Exception as exc:
            err = f"LTMPromotionPipeline.promote failed: {exc}"
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

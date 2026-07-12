"""_STMIngestHandler: STM ingestion logic (moved from memory/ingestor.py)."""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import re
import time
from typing import Any

from graphknows.ingestion.parsers.base import ParsedFile
from graphknows.ingestion.parsers.registry import get_parser
from graphknows.memory.extractor import ExtractionResult
from graphknows.memory.models.ingest import IngestResult
from graphknows.memory.stm._entity_writer import (
    graph_entities,
    write_chunk_entities_and_edges_bulk,
)

log = logging.getLogger("graphknows.memory.stm.ingest")


class _STMIngestHandler:
    """Ingest parsed text into STM (ArcadeDB evidence graph + vector store).

    Attributes:
        _stm: ArcadeDBSTMStore for session evidence graph writes.
        _vec: ArcadeDBVectorStore for stm_chunks writes.
        _extractor: Entity extractor exposing extract(text) -> ExtractionResult.
        _embed_fn: Callable that takes text and returns list[float] embedding.
        _rel_extractor: Optional relation extractor.
    """

    def __init__(
        self,
        stm_store: Any,
        vec_store: Any,
        extractor: Any,
        embed_fn: Any | None = None,
        rel_extractor: Any | None = None,
    ) -> None:
        self._stm = stm_store
        self._vec = vec_store
        self._extractor = extractor
        self._rel_extractor = rel_extractor
        if embed_fn is not None:
            self._embed_fn = embed_fn
        else:
            from graphknows.memory.stores.embedder import embed_one

            self._embed_fn = embed_one

        # Ontology curation layer — applied AFTER extraction to drop relations
        # whose label is not part of the configured taxonomy. The relation model
        # now emits the taxonomy's canonical predicates, so the allow-list is the
        # taxonomy vocabulary (not the unrelated schema-org property set, which
        # silently dropped every conversational predicate).
        from graphknows.extraction.ontology.relation_filter import RelationOntologyFilter
        from graphknows.ingestion.domain.taxonomy import get_default_taxonomy

        self._relation_filter = RelationOntologyFilter(get_default_taxonomy().relation_types)

    async def ingest_text(
        self,
        text: str,
        session_id: str,
        title: str = "",
        source_hint: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> IngestResult:
        """Ingest raw text into STM.

        Args:
            text: Raw source text.
            session_id: Session namespace.
            title: Optional human title.
            source_hint: Original path/URL for provenance.
            metadata: Extra key/value pairs stored on the FILE node.

        Returns:
            IngestResult with counts and timing.
        """
        parser = get_parser(source_hint or ".txt")
        parsed = parser.parse(text, source_hint=source_hint, title=title)
        return await self._ingest_parsed(parsed, session_id, metadata or {})

    async def ingest_parsed(
        self,
        parsed: ParsedFile,
        session_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> IngestResult:
        """Ingest an already-parsed ParsedFile into STM.

        Args:
            parsed: Pre-parsed file output from any BaseParser.
            session_id: Session namespace.
            metadata: Extra key/value pairs stored on the FILE node.

        Returns:
            IngestResult with counts and timing.
        """
        return await self._ingest_parsed(parsed, session_id, metadata or {})

    async def _ingest_parsed(
        self, parsed: ParsedFile, session_id: str, metadata: dict[str, Any]
    ) -> IngestResult:
        t0 = time.monotonic()

        await self._ensure_session_and_file(session_id, parsed, metadata, t0)

        # Pre-compute cleaned texts so they are identical for both embedding and
        # writing (no double-clean).
        cleaned_texts = [self._clean_text(seg.text) for seg in parsed.segments]

        # Batch all embeddings in one HTTP round-trip when using the default
        # embed_one function.  If embed_fn has been overridden (e.g. in tests),
        # fall back to per-chunk calls to preserve the injection contract.
        use_batch = False
        batch_embeddings: list[list[float]] = []
        if cleaned_texts:
            from graphknows.memory.stores.embedder import embed as embed_batch
            from graphknows.memory.stores.embedder import embed_one

            if self._embed_fn is embed_one:
                use_batch = True
                import numpy as np  # embed_batch returns np.ndarray

                arr: np.ndarray = embed_batch(cleaned_texts)
                batch_embeddings = [arr[i].tolist() for i in range(len(cleaned_texts))]

        # Bounded chunk-level concurrency.  Each chunk coroutine runs extraction
        # (CPU/model I/O) and DB writes concurrently; a semaphore caps parallelism.
        # All chunks run on the caller's event loop (the same loop the store client
        # was connected on) via a single asyncio.gather.
        # Default 1 (sequential): the ENTITY(name) index is UNIQUE, so concurrent
        # MERGEs of a shared entity name across chunks can race on the unique key.
        # Opt into parallelism (e.g. for the LLM-heavy llm_assisted ingest) by
        # setting INGEST_CHUNK_CONCURRENCY>1 after validating it for your store.
        cap = int(os.environ.get("INGEST_CHUNK_CONCURRENCY", "1"))

        # Batch NER for ALL chunks in one GLiNER2 forward pass when the extractor
        # supports it — one padded batch beats N per-chunk calls on the CPU-bound
        # ingest path. Falls back to per-chunk extraction otherwise (e.g. tests).
        # Check the class (not the instance) so a Mock's auto-created attributes
        # don't spoof support — only a real extractor exposing extract_batch batches.
        batch_extractions: list[ExtractionResult] | None = None
        if (
            self._extractor is not None
            and hasattr(type(self._extractor), "extract_batch")
            and cleaned_texts
        ):
            batch_extractions = await asyncio.to_thread(
                self._extractor.extract_batch, cleaned_texts
            )

        # Set chunk_index before building coroutines (captured by value via idx).
        for idx, seg in enumerate(parsed.segments):
            seg.metadata["chunk_index"] = idx

        async def _process_chunk(idx: int) -> int:
            """Run extraction + graph + vector writes for chunk at *idx*.

            Returns the entity count for this chunk.
            """
            seg = parsed.segments[idx]
            clean = cleaned_texts[idx]
            embedding: list[float] = batch_embeddings[idx] if use_batch else self._embed_fn(clean)
            # Use the pre-batched extraction when available; else extract this chunk
            # in a thread so its model I/O overlaps DB writes.
            if batch_extractions is not None:
                extraction = batch_extractions[idx]
            else:
                extraction = await asyncio.to_thread(self._extract, clean)
            temporal = self._resolve_temporal(extraction.entities, seg, metadata)
            chunk_id = self._make_chunk_id(session_id, parsed.source_id, seg)

            # Await store methods directly — no nested _run; we are already
            # executing inside the background loop.
            ec = await self._write_chunk_to_mg_async(
                session_id, parsed.source_id, chunk_id, clean, seg, extraction, temporal
            )
            await self._write_chunk_to_pg_async(
                session_id,
                parsed.source_id,
                chunk_id,
                clean,
                seg,
                embedding,
                graph_entities(extraction.entities),
                {**metadata, "temporal": temporal} if temporal else metadata,
            )
            return ec

        async def _run_all() -> list[int]:
            sem = asyncio.Semaphore(cap)

            async def _with_sem(idx: int) -> int:
                async with sem:
                    return await _process_chunk(idx)

            return list(await asyncio.gather(*[_with_sem(i) for i in range(len(parsed.segments))]))

        counts: list[int] = await _run_all()
        entity_count = sum(counts)

        elapsed = time.monotonic() - t0
        log.info(
            "ingest session=%s file=%s chunks=%d entities=%d elapsed=%.1fs",
            session_id,
            parsed.source_id,
            len(parsed.segments),
            entity_count,
            elapsed,
        )
        return IngestResult(
            session_id=session_id,
            file_id=parsed.source_id,
            title=parsed.title,
            chunk_count=len(parsed.segments),
            entity_count=entity_count,
            elapsed_s=elapsed,
            errors=[],
        )

    # ------------------------------------------------------------------
    # Private step methods
    # ------------------------------------------------------------------

    async def _ensure_session_and_file(
        self,
        session_id: str,
        parsed: ParsedFile,
        metadata: dict[str, Any],
        t0: float,
    ) -> None:
        await self._stm.write_session_metadata(session_id, {"started_at": t0})
        await self._stm.write_file_node(
            session_id=session_id,
            file_id=parsed.source_id,
            title=parsed.title,
            parser_name=parsed.parser_name,
            source_hint=metadata.get("source_hint", ""),
        )

    def _resolve_temporal(
        self, entities: list[dict[str, Any]], seg: Any, metadata: dict[str, Any]
    ) -> list[dict[str, str]]:
        """Resolve DATE/TIME spans to ISO dates, anchored on any supplied date.

        The anchor for relative expressions ("last Saturday") is taken from the
        segment metadata, then the file metadata; absent an anchor only absolute
        dates resolve.
        """
        from graphknows.memory.temporal import resolve_temporal

        seg_meta = getattr(seg, "metadata", {}) or {}
        anchor = (
            seg_meta.get("timestamp")
            or seg_meta.get("anchor_date")
            or metadata.get("anchor_date")
            or metadata.get("timestamp")
        )
        return resolve_temporal(entities, anchor)

    def _make_chunk_id(self, session_id: str, source_id: str, seg: Any) -> str:
        # Include the chunk text in the key: char_offset alone collides when a
        # section yields several chunks that resolve to the same offset (e.g.
        # sentence-packed dialogue), which would overwrite all but one chunk.
        key = f"{session_id}:{source_id}::{seg.char_offset}::{seg.text}"
        return hashlib.sha256(key.encode()).hexdigest()[:16]

    def _clean_text(self, text: str) -> str:
        # Strip the '## Content\n\n' heading artifact inserted by PlainTextParser.
        return re.sub(r"^##\s*Content\s*\n+", "", text, flags=re.IGNORECASE).strip()

    def _extract(self, clean_text: str) -> ExtractionResult:
        # One GLiNER2 call returns both entities and relations — capture both.
        if self._extractor is not None:
            from typing import cast

            return cast(ExtractionResult, self._extractor.extract(clean_text))
        return ExtractionResult()

    async def _write_chunk_to_mg_async(
        self,
        session_id: str,
        source_id: str,
        chunk_id: str,
        clean_text: str,
        seg: Any,
        extraction: ExtractionResult,
        temporal: list[dict[str, str]] | None = None,
    ) -> int:
        """Write a chunk's CHUNK node, entity nodes/edges, and relation edges.

        Awaits store methods directly on the caller's event loop (the same loop
        the store client was connected on). Returns the entity count for this chunk.
        """
        chunk_metadata = {**seg.metadata, "temporal": temporal} if temporal else seg.metadata
        await self._stm.write_chunk_node(
            session_id=session_id,
            file_id=source_id,
            chunk_id=chunk_id,
            text=clean_text,
            heading_path=seg.heading_path,
            metadata=chunk_metadata,
        )
        # Bulk entity write: CHUNK node must exist before edges (written above).
        entity_count = await write_chunk_entities_and_edges_bulk(
            stm=self._stm,
            session_id=session_id,
            chunk_id=chunk_id,
            entities=extraction.entities,
        )
        # GLiNER2 relations — ontology-filtered, then written sequentially.
        # Relation writes use transactions with tombstoning logic; leave sequential.
        gliner_relations = self._relation_filter.filter(extraction.relations, label_key="relation")
        for rel in gliner_relations:
            await self._stm.write_relation_edge(
                session_id=session_id,
                subject_name=rel["head"],
                relation_name=rel["relation"],
                object_name=rel["tail"],
                fact=rel.get("evidence", "") or clean_text,
                confidence=rel.get("score", 0.5),
                chunk_id=chunk_id,
            )

        # DSPy ontology-guided relation extraction (llm_assisted mode only).
        if self._rel_extractor is not None and len(extraction.entities) >= 2:
            raw = await self._rel_extractor.extract_relations(text=clean_text)
            dspy_triplets: list[tuple[Any, ...]] = raw[0] if isinstance(raw, tuple) else []
            for triplet in dspy_triplets:
                subj, rel_name, obj = triplet[0], triplet[1], triplet[2]
                props = triplet[3] if len(triplet) > 3 else {}
                confidence = props.get("confidence", 0.5) if isinstance(props, dict) else 0.5
                evidence = props.get("evidence", "") if isinstance(props, dict) else ""
                await self._stm.write_relation_edge(
                    session_id=session_id,
                    subject_name=subj,
                    relation_name=rel_name,
                    object_name=obj,
                    fact=evidence or clean_text,
                    confidence=confidence,
                    chunk_id=chunk_id,
                )
            log.debug(
                "DSPy extraction session=%s chunk=%s: %d triples",
                session_id,
                chunk_id[:8],
                len(dspy_triplets),
            )
        return entity_count

    async def _write_chunk_to_pg_async(
        self,
        session_id: str,
        source_id: str,
        chunk_id: str,
        clean_text: str,
        seg: Any,
        embedding: list[float],
        entities: list[dict[str, Any]],
        metadata: dict[str, Any],
    ) -> None:
        """Async version of _write_chunk_to_pg for the concurrent chunk path."""
        await self._vec.write_stm_chunk(
            {
                "id": chunk_id,
                "session_id": session_id,
                "file_id": source_id,
                "text": clean_text,
                "heading_path": seg.heading_path,
                "embedding": embedding,
                "metadata": {
                    **seg.metadata,
                    **metadata,
                    "atom_type": seg.metadata.get("atom_type", "text"),
                    "memory_class": seg.metadata.get("memory_class", "raw_evidence"),
                    "entity_names": [e["name"] for e in entities],
                },
            }
        )

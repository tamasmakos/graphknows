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
from graphknows.memory.stm._async_bridge import _run
from graphknows.memory.stm._entity_writer import (
    graph_entities,
    write_chunk_entities_and_edges_bulk,
    write_entity_nodes_and_edges,
)

log = logging.getLogger("graphknows.memory.stm.ingest")

# Relation endpoints that are dates/times/bare numbers are noise — and writing a
# relation edge re-creates its endpoints as ENTITY nodes, which would otherwise
# smuggle the numeric/temporal entities (dropped from the graph) back in.
_NOISE_REL_LABELS = frozenset(
    {"TIME", "DATE", "CARDINAL", "ORDINAL", "PERCENT", "MONEY", "QUANTITY"}
)
_BARE_NUM_RE = re.compile(r"^[\W\d]+$")


def _noisy_endpoint(text: str, label: str) -> bool:
    """True if a relation endpoint is a date/time/number rather than a real entity."""
    if (label or "").upper() in _NOISE_REL_LABELS:
        return True
    t = (text or "").strip()
    return not t or bool(_BARE_NUM_RE.match(t))


class _STMIngestHandler:
    """Ingest parsed text into STM (ArcadeDB evidence graph + vector store).

    Attributes:
        _mg: ArcadeDBSTMStore for session evidence graph writes.
        _pg: ArcadeDBVectorStore for stm_chunks writes.
        _extractor: Entity extractor exposing extract(text) -> ExtractionResult.
        _embed_fn: Callable that takes text and returns list[float] embedding.
        _rel_extractor: Optional relation extractor.
    """

    def __init__(
        self,
        mg_store: Any,
        pg_store: Any,
        extractor: Any,
        embed_fn: Any | None = None,
        rel_extractor: Any | None = None,
        spacy_relations: bool = True,
    ) -> None:
        self._mg = mg_store
        self._pg = pg_store
        self._extractor = extractor
        self._rel_extractor = rel_extractor
        self._spacy_relations = spacy_relations
        if embed_fn is not None:
            self._embed_fn = embed_fn
        else:
            from graphknows.memory.stores.embedder import embed_one

            self._embed_fn = embed_one

        # Ontology curation layer — applied AFTER extraction to drop relations
        # whose label is not part of the configured ontology.
        from graphknows.extraction.ontology.relation_filter import build_relation_filter

        self._relation_filter = build_relation_filter()

    def ingest_text(
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
        return self._ingest_parsed(parsed, session_id, metadata or {})

    def ingest_file(
        self,
        path: str,
        session_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> IngestResult:
        """Read a file from disk and ingest into STM.

        Args:
            path: Absolute or relative path to file.
            session_id: Session namespace.
            metadata: Extra key/value pairs stored on the FILE node.

        Returns:
            IngestResult with counts and timing.
        """
        with open(path, encoding="utf-8", errors="replace") as f:
            text = f.read()
        parser = get_parser(path)
        parsed = parser.parse(text, source_hint=path, title=os.path.basename(path))
        return self._ingest_parsed(parsed, session_id, metadata or {})

    def ingest_parsed(
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
        return self._ingest_parsed(parsed, session_id, metadata or {})

    def _ingest_parsed(
        self, parsed: ParsedFile, session_id: str, metadata: dict[str, Any]
    ) -> IngestResult:
        t0 = time.monotonic()

        self._ensure_session_and_file(session_id, parsed, metadata, t0)

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
        # All chunks are dispatched via a SINGLE _run(asyncio.gather(...)) so that
        # every await inside the coroutines lands on the SAME background event loop
        # that _run manages — no nested _run calls, no cross-loop executor issues.
        # Default 1 (sequential): the ENTITY(name) index is UNIQUE, so concurrent
        # MERGEs of a shared entity name across chunks can race on the unique key.
        # Opt into parallelism (e.g. for the LLM-heavy llm_assisted ingest) by
        # setting INGEST_CHUNK_CONCURRENCY>1 after validating it for your store.
        cap = int(os.environ.get("INGEST_CHUNK_CONCURRENCY", "1"))

        # Set chunk_index before building coroutines (captured by value via idx).
        for idx, seg in enumerate(parsed.segments):
            seg.metadata["chunk_index"] = idx

        async def _process_chunk(idx: int) -> int:
            """Run extraction + graph + vector writes for chunk at *idx*.

            Returns the entity count for this chunk.
            """
            seg = parsed.segments[idx]
            clean = cleaned_texts[idx]
            embedding: list[float] = (
                batch_embeddings[idx] if use_batch else self._embed_fn(clean)
            )
            # Wrap synchronous GLiNER2 extraction in a thread so concurrent
            # chunks overlap their model/network I/O with DB writes.
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

            return list(
                await asyncio.gather(*[_with_sem(i) for i in range(len(parsed.segments))])
            )

        counts: list[int] = _run(_run_all())
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

    def _ensure_session_and_file(
        self,
        session_id: str,
        parsed: ParsedFile,
        metadata: dict[str, Any],
        t0: float,
    ) -> None:
        _run(self._mg.write_session_metadata(session_id, {"started_at": t0}))
        _run(
            self._mg.write_file_node(
                session_id=session_id,
                file_id=parsed.source_id,
                title=parsed.title,
                parser_name=parsed.parser_name,
                source_hint=metadata.get("source_hint", ""),
            )
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

    def _write_chunk_to_mg(
        self,
        session_id: str,
        source_id: str,
        chunk_id: str,
        clean_text: str,
        seg: Any,
        extraction: ExtractionResult,
        temporal: list[dict[str, str]] | None = None,
    ) -> int:
        """Write CHUNK node, entity nodes/edges, and relation edges to ArcadeDB STM.

        Sync bridge version — used by callers outside the concurrent chunk path
        (retained for any future callers; not used by _ingest_parsed).
        Returns the entity count for this chunk.
        """
        chunk_metadata = {**seg.metadata, "temporal": temporal} if temporal else seg.metadata
        _run(
            self._mg.write_chunk_node(
                session_id=session_id,
                file_id=source_id,
                chunk_id=chunk_id,
                text=clean_text,
                heading_path=seg.heading_path,
                metadata=chunk_metadata,
            )
        )
        entity_count = write_entity_nodes_and_edges(
            mg=self._mg,
            session_id=session_id,
            entities=extraction.entities,
            edge_writer=lambda ent: _run(
                self._mg.write_chunk_entity_edge(
                    chunk_id=chunk_id,
                    entity_name=ent["name"],
                    session_id=session_id,
                )
            ),
            _run=_run,
        )
        # GLiNER2 relations — ontology-filtered, then written.
        gliner_relations = self._relation_filter.filter(extraction.relations, label_key="relation")
        for rel in gliner_relations:
            _run(
                self._mg.write_relation_edge(
                    session_id=session_id,
                    subject_name=rel["head"],
                    relation_name=rel["relation"],
                    object_name=rel["tail"],
                    fact=rel.get("evidence", "") or clean_text,
                    confidence=rel.get("score", 0.5),
                    chunk_id=chunk_id,
                )
            )

        # SpaCy dep-pattern relations (intentionality strata G/H/I) — toggleable, soft-fail.
        if self._spacy_relations:
            self._write_spacy_dep_relations(
                session_id, clean_text, extraction.entities, chunk_id=chunk_id
            )

        # DSPy ontology-guided relation extraction (llm_assisted mode only).
        # extract_relations returns (triplets, nodes, diagnostics).
        # Each triplet is (subject, relation, object, props_dict).
        if self._rel_extractor is not None and len(extraction.entities) >= 2:
            try:
                raw = _run(self._rel_extractor.extract_relations(text=clean_text))
            except Exception as exc:
                # Never let a relation-extractor failure abort ingestion of the
                # whole document — the chunk is still embedded and graph-written.
                log.warning("DSPy relation extraction skipped for chunk %s: %s", chunk_id[:8], exc)
                raw = ([], [], {})
            dspy_triplets: list[tuple[Any, ...]] = raw[0] if isinstance(raw, tuple) else []
            for triplet in dspy_triplets:
                subj, rel_name, obj = triplet[0], triplet[1], triplet[2]
                props = triplet[3] if len(triplet) > 3 else {}
                confidence = props.get("confidence", 0.5) if isinstance(props, dict) else 0.5
                evidence = props.get("evidence", "") if isinstance(props, dict) else ""
                _run(
                    self._mg.write_relation_edge(
                        session_id=session_id,
                        subject_name=subj,
                        relation_name=rel_name,
                        object_name=obj,
                        fact=evidence or clean_text,
                        confidence=confidence,
                        chunk_id=chunk_id,
                    )
                )
            log.debug(
                "DSPy extraction session=%s chunk=%s: %d triples",
                session_id,
                chunk_id[:8],
                len(dspy_triplets),
            )
        return entity_count

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
        """Async version of _write_chunk_to_mg for the concurrent chunk path.

        Awaits store methods directly — must only be called from within a
        coroutine already running on the background event loop (i.e. inside
        the gather dispatched by _run(_run_all())).  DO NOT call via _run()
        as that would nest _run calls and deadlock the single background loop.

        Returns the entity count for this chunk.
        """
        chunk_metadata = {**seg.metadata, "temporal": temporal} if temporal else seg.metadata
        await self._mg.write_chunk_node(
            session_id=session_id,
            file_id=source_id,
            chunk_id=chunk_id,
            text=clean_text,
            heading_path=seg.heading_path,
            metadata=chunk_metadata,
        )
        # Bulk entity write: CHUNK node must exist before edges (written above).
        entity_count = await write_chunk_entities_and_edges_bulk(
            mg=self._mg,
            session_id=session_id,
            chunk_id=chunk_id,
            entities=extraction.entities,
        )
        # GLiNER2 relations — ontology-filtered, then written sequentially.
        # Relation writes use transactions with tombstoning logic; leave sequential.
        gliner_relations = self._relation_filter.filter(extraction.relations, label_key="relation")
        for rel in gliner_relations:
            await self._mg.write_relation_edge(
                session_id=session_id,
                subject_name=rel["head"],
                relation_name=rel["relation"],
                object_name=rel["tail"],
                fact=rel.get("evidence", "") or clean_text,
                confidence=rel.get("score", 0.5),
                chunk_id=chunk_id,
            )

        # SpaCy dep-pattern relations — soft-fail, written sequentially.
        if self._spacy_relations:
            await self._write_spacy_dep_relations_async(
                session_id, clean_text, extraction.entities, chunk_id=chunk_id
            )

        # DSPy ontology-guided relation extraction (llm_assisted mode only).
        if self._rel_extractor is not None and len(extraction.entities) >= 2:
            try:
                raw = await self._rel_extractor.extract_relations(text=clean_text)
            except Exception as exc:
                log.warning("DSPy relation extraction skipped for chunk %s: %s", chunk_id[:8], exc)
                raw = ([], [], {})
            dspy_triplets: list[tuple[Any, ...]] = raw[0] if isinstance(raw, tuple) else []
            for triplet in dspy_triplets:
                subj, rel_name, obj = triplet[0], triplet[1], triplet[2]
                props = triplet[3] if len(triplet) > 3 else {}
                confidence = props.get("confidence", 0.5) if isinstance(props, dict) else 0.5
                evidence = props.get("evidence", "") if isinstance(props, dict) else ""
                await self._mg.write_relation_edge(
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

    async def _write_spacy_dep_relations_async(
        self,
        session_id: str,
        text: str,
        gliner_entities: list[dict[str, Any]],
        chunk_id: str = "",
    ) -> None:
        """Async version of _write_spacy_dep_relations for the concurrent chunk path.

        Semantics identical to the sync version; awaits write_relation_edge directly.
        """
        try:
            from graphknows.extraction.relations.svo import extract_svo_relations

            triples = extract_svo_relations(text, graph_entities(gliner_entities))
        except Exception as exc:
            log.debug("SVO relation extraction skipped: %s", exc)
            return

        written = 0
        for triple in triples:
            if triple.get("negated"):
                continue
            if _noisy_endpoint(
                triple["source"], triple.get("source_label", "")
            ) or _noisy_endpoint(triple["target"], triple.get("target_label", "")):
                continue
            try:
                await self._mg.write_relation_edge(
                    session_id=session_id,
                    subject_name=triple["source"],
                    relation_name=triple["relation"],
                    object_name=triple["target"],
                    fact=triple.get("evidence", "") or text,
                    confidence=triple.get("confidence", 0.7),
                    chunk_id=chunk_id,
                )
                written += 1
            except Exception as exc:
                log.debug("SVO relation write error (non-fatal): %s", exc)
        if written:
            log.debug("SVO relations: %d triples written for session=%s", written, session_id)

    def _write_spacy_dep_relations(
        self,
        session_id: str,
        text: str,
        gliner_entities: list[dict[str, Any]],
        chunk_id: str = "",
    ) -> None:
        """Extract universal SVO relations and write them to ArcadeDB STM.

        Uses the domain-agnostic SVO extractor (subject–predicate–object over the
        dependency parse, with conversational participant resolution and a compact
        relation taxonomy) rather than the narrow speech-act intentionality
        patterns, which only fire on declarative prose. The extractor handles all
        precision filtering (subject must be a real entity, noun-phrase objects,
        questions/discourse pronouns dropped).

        Soft-fails: missing spaCy model or parse errors are logged at DEBUG and do
        not interrupt ingestion.
        """
        try:
            from graphknows.extraction.relations.svo import extract_svo_relations

            triples = extract_svo_relations(text, graph_entities(gliner_entities))
        except Exception as exc:
            log.debug("SVO relation extraction skipped: %s", exc)
            return

        written = 0
        for triple in triples:
            if triple.get("negated"):
                continue
            if _noisy_endpoint(triple["source"], triple.get("source_label", "")) or _noisy_endpoint(
                triple["target"], triple.get("target_label", "")
            ):
                continue
            try:
                _run(
                    self._mg.write_relation_edge(
                        session_id=session_id,
                        subject_name=triple["source"],
                        relation_name=triple["relation"],
                        object_name=triple["target"],
                        fact=triple.get("evidence", "") or text,
                        confidence=triple.get("confidence", 0.7),
                        chunk_id=chunk_id,
                    )
                )
                written += 1
            except Exception as exc:
                log.debug("SVO relation write error (non-fatal): %s", exc)
        if written:
            log.debug("SVO relations: %d triples written for session=%s", written, session_id)

    def _write_chunk_to_pg(
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
        _run(
            self._pg.write_stm_chunk(
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
        )

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
        await self._pg.write_stm_chunk(
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

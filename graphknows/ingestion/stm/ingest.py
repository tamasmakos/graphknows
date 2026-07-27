"""_STMIngestHandler: ingest parsed text / turns into the golden GraphStore.

Writes CHUNK spine + ENTITY/MENTIONS + RELATION(evidence) + TEMPORAL against a
single per-namespace ``GraphStore`` (state defaults ``raw``). The text path
chunks a parsed file; the turn path (Phase-1 bridge) writes one CHUNK per
message with ``kind='turn'`` through the same write helper.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import time
from typing import Any

from graphknows.ingestion.extraction.entities.extractor import ExtractionResult
from graphknows.ingestion.parsers.base import ParsedFile, strip_content_heading
from graphknows.ingestion.parsers.registry import get_parser
from graphknows.ingestion.stm._entity_writer import graph_entities, is_graph_entity_name
from graphknows.models.ingest import IngestResult

log = logging.getLogger("graphknows.ingestion.stm.ingest")

# Frames matched per chunk by embedding cosine: top-k AND above a similarity floor
# (cosines run low under the remote embedder, ~0.3, so the floor is permissive).
_FRAME_TOP_K = 5
_FRAME_MIN_SIM = 0.20

# Ontology classes tagged onto a chunk as MATCHES_CLASS edges. Deliberately
# narrower than the label-injection top-k: injected labels are cheap guesses the
# extractor can ignore, whereas a persisted tag is signal retrieval will trust.
_ONT_TAG_TOP_K = 3

# Floor for mapping an entity TYPE onto an ontology class by embedding. Measured
# against personal.ttl with bge-small, correct mappings (PERSON->Person 0.66,
# LOCATION->Place 0.54, DATE->TimePoint 0.45) sit above it while wrong ones
# (OCCUPATION->Possession 0.43, ANIMAL->Emotion 0.38) fall below. Leaving an
# entity unmapped is the safe outcome, so the floor errs high.
_ONT_CLASS_MIN_SIM = 0.45


class _STMIngestHandler:
    """Ingest parsed text into the golden GraphStore (state='raw').

    Attributes:
        _store: GraphStore for the single per-namespace database.
        _extractor: Entity extractor exposing extract(text) -> ExtractionResult.
        _embed_fn: Callable that takes text and returns list[float] embedding.
        _rel_extractor: Optional relation extractor (llm_assisted).
        _ontology: Optional OntologyIndex; steers extraction labels and links
            the written entities/chunks to their ontology terms.
    """

    def __init__(
        self,
        store: Any,
        extractor: Any,
        embed_fn: Any | None = None,
        rel_extractor: Any | None = None,
        enable_frames: bool = False,
        ontology_index: Any | None = None,
    ) -> None:
        self._store = store
        self._extractor = extractor
        self._rel_extractor = rel_extractor
        self._enable_frames = enable_frames
        self._ontology = ontology_index
        # Predicate mapping uses char-n-gram TF-IDF plus a surface-alias table
        # ("is friends with" -> friend_of), which suits multi-word predicates.
        # Class mapping deliberately does NOT: on single-word type labels the
        # n-gram similarity is dominated by shared suffixes and confidently
        # returns nonsense (LOCATION -> Organization, ANIMAL -> Goal), so types
        # are matched against the class DEFINITIONS by embedding instead.
        self._property_mapper: Any = None
        self._uri_by_label: dict[str, str] = {}
        self._type_cache: dict[str, tuple[str, str, float]] = {}
        if ontology_index is not None:
            from graphknows.ingestion.extraction.relations.filter import OntologyPredicateMapper

            self._property_mapper = OntologyPredicateMapper(ontology_index.property_labels)
            self._uri_by_label = {t.label.casefold(): t.uri for t in ontology_index.classes}
        if embed_fn is not None:
            self._embed_fn = embed_fn
        else:
            from graphknows.storage.embedder import embed_one

            self._embed_fn = embed_one

    async def ingest_text(
        self,
        text: str,
        session_id: str,
        title: str = "",
        source_hint: str = "",
        metadata: dict[str, Any] | None = None,
        infer: bool = True,
    ) -> IngestResult:
        """Ingest raw text into the GraphStore."""
        parser = get_parser(source_hint or ".txt")
        parsed = parser.parse(text, source_hint=source_hint, title=title)
        return await self._ingest_parsed(parsed, session_id, metadata or {}, infer=infer)

    async def ingest_parsed(
        self,
        parsed: ParsedFile,
        session_id: str,
        metadata: dict[str, Any] | None = None,
        infer: bool = True,
    ) -> IngestResult:
        """Ingest an already-parsed ParsedFile into the GraphStore."""
        return await self._ingest_parsed(parsed, session_id, metadata or {}, infer=infer)

    async def _ingest_parsed(
        self, parsed: ParsedFile, session_id: str, metadata: dict[str, Any], infer: bool = True
    ) -> IngestResult:
        t0 = time.monotonic()

        await self._ensure_session_and_file(session_id, parsed, metadata, t0)

        # Pre-compute cleaned texts so they are identical for both embedding and
        # writing (no double-clean).
        cleaned_texts = [self._clean_text(seg.text) for seg in parsed.segments]

        use_batch, batch_embeddings = self._precompute_embeddings(cleaned_texts)

        # Bounded chunk-level concurrency. Default 1 (sequential): the
        # ENTITY(name_norm) index is UNIQUE, so concurrent MERGEs of a shared
        # entity name across chunks can race on the unique key. Opt into
        # parallelism by setting INGEST_CHUNK_CONCURRENCY>1.
        cap = int(os.environ.get("INGEST_CHUNK_CONCURRENCY", "1"))

        # Per-chunk ontology label hints, reusing the embeddings above (no
        # re-embed). Empty when no ontology is configured, which is the default.
        ont_classes = [self._match_classes(e) for e in batch_embeddings]
        entity_hints, relation_hints = self._label_hints(batch_embeddings)

        # Batch NER for ALL chunks in one GLiNER2 forward pass when supported.
        # ``infer=False`` skips extraction entirely (chunks + embeddings only) —
        # the entity/relation extraction and its per-relation graph writes are the
        # ingest cost that never touches the vector index.
        batch_extractions: list[ExtractionResult] | None = None
        if (
            infer
            and self._extractor is not None
            and hasattr(type(self._extractor), "extract_batch")
            and cleaned_texts
        ):
            batch_extractions = await asyncio.to_thread(
                self._extractor.extract_batch, cleaned_texts, entity_hints, relation_hints
            )

        for idx, seg in enumerate(parsed.segments):
            seg.metadata["chunk_index"] = idx

        async def _process_chunk(idx: int) -> int:
            seg = parsed.segments[idx]
            clean = cleaned_texts[idx]
            embedding: list[float] = batch_embeddings[idx] if use_batch else self._embed_fn(clean)
            if not infer:
                extraction = ExtractionResult()
            elif batch_extractions is not None:
                extraction = batch_extractions[idx]
            else:
                extraction = await asyncio.to_thread(self._extract, clean)
            temporal = self._resolve_temporal(extraction.entities, seg, metadata)
            chunk_id = self._make_chunk_id(session_id, parsed.source_id, seg)
            return await self._write_extracted(
                session_id=session_id,
                file_id=parsed.source_id,
                chunk_id=chunk_id,
                text=clean,
                seq=idx,
                kind="text",
                speaker="",
                ts=str(seg.metadata.get("timestamp") or metadata.get("timestamp") or ""),
                embedding=embedding,
                extraction=extraction,
                temporal=temporal,
                memory_class=seg.metadata.get("memory_class")
                or metadata.get("memory_class")
                or "raw_evidence",
                atom_type=seg.metadata.get("atom_type") or "text",
                ontology_classes=ont_classes[idx] if idx < len(ont_classes) else [],
            )

        async def _run_all() -> tuple[list[int], list[str]]:
            """Ingest every chunk, surviving individual failures.

            One bad chunk must not abandon the rest of the file: the message
            path (``STMService.ingest_messages``) already records a per-turn
            error and continues, and both paths return the same
            ``IngestResult.errors`` contract.
            """
            sem = asyncio.Semaphore(cap)

            async def _with_sem(idx: int) -> int:
                async with sem:
                    return await _process_chunk(idx)

            results = await asyncio.gather(
                *[_with_sem(i) for i in range(len(parsed.segments))],
                return_exceptions=True,
            )
            counts: list[int] = []
            failures: list[str] = []
            for idx, res in enumerate(results):
                if isinstance(res, BaseException):
                    log.warning("ingest chunk %d failed: %s", idx, res)
                    failures.append(f"chunk {idx}: {res}")
                else:
                    counts.append(res)
            return counts, failures

        counts, errors = await _run_all()
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
            errors=errors,
        )

    async def ingest_turn(
        self,
        session_id: str,
        file_id: str,
        seq: int,
        role: str,
        text: str,
        metadata: dict[str, Any] | None = None,
        infer: bool = True,
        speaker: str = "",
    ) -> int:
        """Write one conversation turn as a CHUNK(kind='turn'). Returns entity count.

        Phase-1 bridge: gives the messages path the same full extraction (ENTITY
        + MENTIONS + evidenced relations + temporal) the text/chunk path has.
        ``infer=False`` writes the embedded turn without entity/relation extraction.

        ``speaker`` is who actually said it ("Gina"); it falls back to ``role``,
        which is all this had before — so a two-human conversation recorded every
        chunk's speaker as "user".

        Whether a given turn is worth extracting from is the CALLER's decision,
        expressed through ``infer`` — this layer applies no content policy of
        its own.
        """
        metadata = metadata or {}
        clean = self._clean_text(text) or text
        if not clean.strip():
            return 0
        embedding = self._embed_fn(clean)
        # Same ontology steering as the chunk path — the turn's own embedding
        # selects the candidate labels, so no extra embedding call.
        ent_hints, rel_hints = self._label_hints([embedding])
        extraction = (
            await asyncio.to_thread(
                self._extract,
                clean,
                ent_hints[0] if ent_hints else None,
                rel_hints[0] if rel_hints else None,
            )
            if infer
            else ExtractionResult()
        )
        anchor = metadata.get("timestamp") or metadata.get("anchor_date")
        temporal = self._resolve_temporal_entities(extraction.entities, anchor)
        chunk_id = hashlib.sha256(f"{session_id}:{file_id}:{seq}:{clean}".encode()).hexdigest()[:16]
        return await self._write_extracted(
            session_id=session_id,
            file_id=file_id,
            chunk_id=chunk_id,
            text=clean,
            seq=seq,
            kind="turn",
            speaker=speaker or role,
            # The same anchor the relative dates resolve against is what dates
            # the chunk at retrieval. Until now nothing populated CHUNK.ts, so
            # the column was written, indexed and selected as "" everywhere.
            ts=str(anchor or ""),
            embedding=embedding,
            extraction=extraction,
            temporal=temporal,
            memory_class=metadata.get("memory_class") or "raw_evidence",
            atom_type="turn",
            ontology_classes=self._match_classes(embedding) if infer else [],
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
        await self._store.ensure_session(session_id, started_at=str(t0))
        await self._store.ensure_file(
            file_id=parsed.source_id,
            session_id=session_id,
            name=parsed.title,
            media_type=parsed.parser_name,
        )

    async def _write_extracted(
        self,
        *,
        session_id: str,
        file_id: str,
        chunk_id: str,
        text: str,
        seq: int,
        kind: str,
        speaker: str,
        ts: str,
        embedding: list[float],
        extraction: ExtractionResult,
        temporal: list[dict[str, str]],
        memory_class: str,
        atom_type: str,
        ontology_classes: list[tuple[str, float]] | None = None,
    ) -> int:
        """Write one chunk's CHUNK + ENTITY/MENTIONS + RELATION + TEMPORAL. Returns entity count."""
        filtered = graph_entities(extraction.entities)
        await self._store.write_chunk(
            {
                "chunk_id": chunk_id,
                "session_id": session_id,
                "file_id": file_id,
                "text": text,
                "seq": seq,
                "kind": kind,
                "speaker": speaker,
                "ts": ts,
                "memory_class": memory_class,
                "atom_type": atom_type,
                "entity_names": [e["name"] for e in filtered],
                "embedding": embedding,
            }
        )
        if filtered:
            await self._store.write_entities_bulk(session_id, chunk_id, filtered)

        # Ontology layer — tag the chunk with the classes it evokes and link each
        # entity to its class. Both are additive: an entity that maps to nothing
        # keeps its extracted type and is written exactly as before.
        await self._link_ontology(chunk_id, filtered, ontology_classes or [])

        # FRAME layer — match the chunk's embedding against FrameNet definitions
        # (top-k + min-sim floor, reusing the embedding already computed above; no
        # re-embed) so abstract/general questions can reach this chunk by the frame
        # it evokes. Each FRAME node carries the frame's summary + definition
        # embedding (copied from the catalog, not recomputed) so the query side can
        # vector-search the frames evoked in this namespace.
        if self._enable_frames:
            from graphknows.channels._frame_index import (
                frame_embedding,
                frame_summary,
                match_frames,
            )

            matches = match_frames(embedding, top_k=_FRAME_TOP_K, min_sim=_FRAME_MIN_SIM)
            if matches:
                await self._store.write_frames(
                    session_id,
                    [
                        {
                            "frame_type": name,
                            "chunk_id": chunk_id,
                            "lu": "",
                            "speaker": speaker or "",
                            "fes": [],
                            "fillers": [],
                            "confidence": sim,
                            "summary": frame_summary(name),
                            "embedding": frame_embedding(name),
                        }
                        for name, sim in matches
                    ],
                )

        # FRAME_INSTANCE layer — frame-guided semantic role labeling. The chunk's
        # top FrameNet frames (main-verb LUs, embedding-ranked) become a GLiNER2
        # core-FE schema; the role-fillability gate keeps 0-1 filled instances per
        # short text. Reuses the extractor's spaCy + GLiNER2 (no second load); a
        # second GLiNER2 schema pass, so only on fact-bearing chunks. Distinct from
        # the FRAME chunk-tag layer above.
        if self._enable_frames:
            instances = await asyncio.to_thread(
                self._frame_instances, text, chunk_id, speaker, embedding
            )
            if instances:
                await self._store.write_frame_instances(session_id, chunk_id, instances)

        await self._write_relations(extraction, chunk_id, text)

        iso_dates = [t["iso"] for t in temporal if t.get("iso")]
        if iso_dates:
            await self._store.write_temporal(chunk_id, iso_dates)
        return len(filtered)

    async def _write_relations(
        self, extraction: ExtractionResult, chunk_id: str, text: str
    ) -> None:
        """Write the chunk's relation edges — deterministic pass, then DSPy.

        The ontology ANNOTATES rather than filters: a predicate that maps to an
        ontology property records that mapping, and one that maps to nothing is
        still written under its extracted label. A filter here would reject
        every open-vocab label, which is why one was refused before.

        Temporal/numeric ENDPOINTS are a different matter: write_relation MERGEs
        its endpoints as ENTITY nodes with no filtering, which is how dates
        became ENTITY vertices with community_ids and fragmented the topic
        layer. Both loops share the same endpoint gate; the date signal is not
        lost — it lives on the chunk's MENTIONS_DATE/TEMPORAL link.
        """
        for rel in extraction.relations:
            if not is_graph_entity_name(rel["head"], rel.get("head_type", "")) or (
                not is_graph_entity_name(rel["tail"], rel.get("tail_type", ""))
            ):
                continue
            ont_property, map_score = self._map_predicate(rel["relation"])
            await self._store.write_relation(
                rel["head"],
                rel["relation"],
                rel["tail"],
                fact=rel.get("evidence", "") or text,
                chunk_id=chunk_id,
                confidence=rel.get("score", 0.5),
                source=rel.get("source", "gliner"),
                src_type=rel.get("head_type", ""),
                dst_type=rel.get("tail_type", ""),
                ontology_property=ont_property,
                map_score=map_score,
                verifier_score=rel.get("verifier_score"),
            )

        # DSPy ontology-guided relations (llm_assisted mode only).
        if self._rel_extractor is not None and len(extraction.entities) >= 2:
            raw = await self._rel_extractor.extract_relations(text=text)
            dspy_triplets: list[tuple[Any, ...]] = raw[0] if isinstance(raw, tuple) else []
            for triplet in dspy_triplets:
                subj, rel_name, obj = triplet[0], triplet[1], triplet[2]
                if not is_graph_entity_name(str(subj)) or not is_graph_entity_name(str(obj)):
                    continue
                props = triplet[3] if len(triplet) > 3 else {}
                confidence = props.get("confidence", 0.5) if isinstance(props, dict) else 0.5
                evidence = props.get("evidence", "") if isinstance(props, dict) else ""
                await self._store.write_relation(
                    subj,
                    rel_name,
                    obj,
                    fact=evidence or text,
                    chunk_id=chunk_id,
                    confidence=confidence,
                    source="dspy",
                )

    def _precompute_embeddings(self, texts: list[str]) -> tuple[bool, list[list[float]]]:
        """Embed every chunk up front when possible. Returns ``(precomputed, embeddings)``.

        Two reasons to do this before the per-chunk loop: the default embedder
        batches into one HTTP round-trip, and ontology label hints are derived
        from a chunk's embedding and must exist *before* extraction runs. An
        overridden ``embed_fn`` (tests) still gets per-chunk calls unless an
        ontology forces the precompute, preserving the injection contract.
        """
        if not texts:
            return False, []
        from graphknows.storage.embedder import embed as embed_batch
        from graphknows.storage.embedder import embed_one

        if self._embed_fn is embed_one:
            import numpy as np  # embed_batch returns np.ndarray

            arr: np.ndarray = embed_batch(texts)
            return True, [arr[i].tolist() for i in range(len(texts))]
        if self._ontology is not None:
            return True, [self._embed_fn(t) for t in texts]
        return False, []

    # ------------------------------------------------------------------
    # Ontology injection (mode-independent; inert when no ontology is set)
    # ------------------------------------------------------------------

    def _match_classes(self, embedding: list[float]) -> list[tuple[str, float]]:
        """Ontology classes this chunk evokes, as ``[(uri, sim)]`` for tagging.

        Uses the stricter TAG floor: these become persisted MATCHES_CLASS edges,
        unlike the label hints which are gated by top-k alone.
        """
        if self._ontology is None or not embedding:
            return []
        from graphknows.ontology.catalog import TAG_MIN_SIM

        matches = self._ontology.match_classes(embedding, top_k=_ONT_TAG_TOP_K, min_sim=TAG_MIN_SIM)
        return [(term.uri, sim) for term, sim in matches]

    def _label_hints(
        self, embeddings: list[list[float]]
    ) -> tuple[list[list[str]] | None, list[dict[str, str]] | None]:
        """Per-chunk ontology class labels and property label->definition maps.

        ``(None, None)`` when no ontology is configured, which keeps the
        extractor call byte-identical to the pre-ontology path.
        """
        if self._ontology is None or not embeddings:
            return None, None
        entity_hints = [[t.label for t, _ in self._ontology.match_classes(e)] for e in embeddings]
        relation_hints = [
            {t.label: t.definition or t.label for t, _ in self._ontology.match_properties(e)}
            for e in embeddings
        ]
        return entity_hints, relation_hints

    def _map_to_class(self, entity_type: str) -> tuple[str, str, float]:
        """Map a free-text entity type onto an ontology class: ``(uri, label, score)``.

        An exact label hit scores 1.0 — the common case, because the ontology's
        own labels were injected into the extractor's vocabulary. Otherwise the
        type is embedded and matched against the class definitions. Results are
        memoised per type string: types repeat across every chunk, so this costs
        one embedding per DISTINCT type in the corpus, not one per entity.

        ``("", "", 0.0)`` means unmapped, and an unmapped entity is still
        written with its extracted type: this is enrichment, not a filter.
        """
        if self._ontology is None or not entity_type:
            return "", "", 0.0
        if (hit := self._type_cache.get(entity_type)) is not None:
            return hit
        uri = self._uri_by_label.get(entity_type.casefold())
        if uri:
            # Canonical ontology casing, not the extractor's ("PERSON" -> "Person").
            label = next((t.label for t in self._ontology.classes if t.uri == uri), entity_type)
            result = (uri, label, 1.0)
        else:
            from graphknows.storage.embedder import embed_one

            matches = self._ontology.match_classes(
                embed_one(entity_type), top_k=1, min_sim=_ONT_CLASS_MIN_SIM
            )
            result = (
                (matches[0][0].uri, matches[0][0].label, matches[0][1])
                if matches
                else ("", "", 0.0)
            )
        self._type_cache[entity_type] = result
        return result

    async def _link_ontology(
        self,
        chunk_id: str,
        entities: list[dict[str, Any]],
        ontology_classes: list[tuple[str, float]],
    ) -> None:
        """Write the chunk's MATCHES_CLASS tags and its entities' INSTANCE_OF edges."""
        if self._ontology is None:
            return
        if ontology_classes:
            await self._store.write_class_matches(chunk_id, ontology_classes)
        mappings = []
        for ent in entities:
            uri, label, score = self._map_to_class(str(ent.get("type", "")))
            if uri:
                mappings.append({"name": ent["name"], "uri": uri, "label": label, "score": score})
        if mappings:
            await self._store.write_instance_of(mappings)

    def _map_predicate(self, predicate: str) -> tuple[str, float]:
        """Map a relation predicate onto an ontology property label."""
        if self._property_mapper is None or not predicate:
            return "", 0.0
        label, score = self._property_mapper.map_predicate(predicate)
        return (label or ""), score

    def _frame_instances(
        self, text: str, chunk_id: str, speaker: str, embedding: list[float] | None = None
    ) -> list[dict[str, Any]]:
        """Frame-guided role extraction for one chunk → write dicts (thread-run).

        Reuses the entity extractor's already-loaded spaCy + GLiNER2 (no second
        load) and the chunk's already-computed ``embedding`` for frame ranking (no
        re-embed). Speaker/speakers come from the chunk's turn-lines (same logic the
        SVO pass uses), falling back to the passed ``speaker``. Returns
        ``[{frame, match_sim, role_confidence, roles}]`` (0-1 items per short text,
        per the role-fillability gate; the FRAME_INSTANCE id is content-hashed at
        write time). Empty if the extractor doesn't expose reusable models.
        """
        extractor = self._extractor
        if extractor is None or not hasattr(extractor, "nlp") or not hasattr(extractor, "gliner"):
            return []
        from graphknows.ingestion.extraction.entities.extractor import _line_speakers
        from graphknows.ingestion.extraction.relations.frame_srl import (
            extract_frame_instances,
            lu_candidate_frames,
        )

        line_speakers = _line_speakers(text)
        speakers = sorted({s for _, s in line_speakers if s})
        primary = speaker or (speakers[0] if speakers else "")
        if primary and primary not in speakers:
            speakers = sorted({*speakers, primary})

        nlp = extractor.nlp
        gliner = extractor.gliner
        instances = extract_frame_instances(
            text,
            primary,
            speakers,
            gliner,
            lambda t: lu_candidate_frames(t, nlp, embedding=embedding),
        )
        return [
            {
                "frame": inst.frame,
                "match_sim": inst.match_sim,
                "role_confidence": inst.role_confidence,
                "roles": inst.roles,
            }
            for inst in instances
        ]

    def _resolve_temporal(
        self, entities: list[dict[str, Any]], seg: Any, metadata: dict[str, Any]
    ) -> list[dict[str, str]]:
        """Resolve DATE/TIME spans to ISO dates, anchored on any supplied date."""
        seg_meta = getattr(seg, "metadata", {}) or {}
        anchor = (
            seg_meta.get("timestamp")
            or seg_meta.get("anchor_date")
            or metadata.get("anchor_date")
            or metadata.get("timestamp")
        )
        return self._resolve_temporal_entities(entities, anchor)

    @staticmethod
    def _resolve_temporal_entities(
        entities: list[dict[str, Any]], anchor: Any
    ) -> list[dict[str, str]]:
        from graphknows.channels.temporal import resolve_temporal

        return resolve_temporal(entities, anchor)

    def _make_chunk_id(self, session_id: str, source_id: str, seg: Any) -> str:
        # Include the chunk text in the key: char_offset alone collides when a
        # section yields several chunks that resolve to the same offset.
        key = f"{session_id}:{source_id}::{seg.char_offset}::{seg.text}"
        return hashlib.sha256(key.encode()).hexdigest()[:16]

    def _clean_text(self, text: str) -> str:
        return strip_content_heading(text)

    def _extract(
        self,
        clean_text: str,
        extra_entity_labels: list[str] | None = None,
        extra_relation_labels: dict[str, str] | None = None,
    ) -> ExtractionResult:
        if self._extractor is not None:
            from typing import cast

            return cast(
                ExtractionResult,
                self._extractor.extract(clean_text, extra_entity_labels, extra_relation_labels),
            )
        return ExtractionResult()

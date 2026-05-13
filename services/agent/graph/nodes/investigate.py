"""Investigate node: single-pass search iteration with dual-write and novelty check."""
from __future__ import annotations

import asyncio
import json
import logging
import sys
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from sentence_transformers import SentenceTransformer

from services.agent.graph.events import (
    chunks_written_event,
    docs_fetched_event,
    entities_written_event,
    graph_update_event,
    novelty_check_event,
    relations_extracted_event,
    search_done_event,
    search_start_event,
)
from services.agent.graph.state import Chunk, Entity, InvestigativeState, RawDoc, Relation, SubQuestion

if TYPE_CHECKING:
    from services.agent.infra.pgvector_client import PgVectorClient
    from services.agent.tools.search_client import SearchServiceClient

logger = logging.getLogger(__name__)

EMBED_MODEL: str = "BAAI/bge-small-en-v1.5"


def _make_stub_doc(iteration: int, topic: str) -> RawDoc:
    """Create a placeholder RawDoc when the search service is unavailable.

    Generates entity-rich text so downstream NER/extraction pipelines have
    meaningful named entities (ORG, LOC, POLICY) to work with even when no
    real search results are available.
    """
    stub_doc_id = f"doc-{uuid.uuid4().hex[:8]}"
    text = (
        f"{topic} is a subject of significant interest to researchers, policymakers, "
        f"and organizations worldwide, including bodies such as the United Nations, "
        f"the European Commission, and the World Economic Forum. "
        f"Stakeholders across the United States, the European Union, and the Asia-Pacific "
        f"region have engaged with matters related to {topic} through international forums "
        f"and regulatory bodies. "
        f"Key industry actors including Google, Microsoft, Meta, and Amazon, alongside "
        f"academic institutions and non-governmental organizations, have published research "
        f"and policy positions on {topic}. "
        f"The broader implications of {topic} span sectors such as healthcare, finance, "
        f"law enforcement, and education, making it a topic of cross-jurisdictional relevance."
    )
    return RawDoc(
        id=stub_doc_id,
        title=f"Overview: {topic}",
        text=text,
        url=f"https://example.test/doc/{stub_doc_id}",
        source="stub",
        date=None,
    )


def _pick_sources(plan: list[SubQuestion]) -> list[str]:
    """Flatten *source_priorities* from all sub-questions, deduplicate, filter to known adapters."""
    known = {"ep_debates", "eurlex", "uk_legislation", "web"}
    sources: list[str] = []
    for sq in plan:
        for src in sq.source_priorities:
            if src in known and src not in sources:
                sources.append(src)
    return sources or ["ep_debates", "eurlex"]


class DualWriter:
    """Persists chunks to pgvector and entities/relations to the session graph backend.

    Attributes:
        session_id: Active session namespace.
        pg: PgVectorClient instance for chunk persistence.
        backend: Optional GraphBackend for graph persistence.
        mg: Legacy Memgraph driver (for backward compatibility).
    """

    def __init__(
        self,
        session_id: str,
        pg_client: PgVectorClient,
        mg_driver: Any = None,
        backend: Any = None,
    ) -> None:
        """Initialise DualWriter.

        Args:
            session_id: Session namespace for all writes.
            pg_client: Initialised PgVectorClient.
            mg_driver: Optional legacy Neo4j driver connected to Memgraph.
            backend: Optional GraphBackend instance.
        """
        self.session_id = session_id
        self.pg = pg_client
        self.mg = mg_driver
        self.backend = backend

    async def write(
        self,
        chunks: list[Chunk],
        entities: list[Entity],
        relations: list[Relation],
    ) -> None:
        """Persist chunks to pgvector and entities/relations to graph concurrently.

        Args:
            chunks: Chunks to write to the vector store.
            entities: Entities to write to the graph store.
            relations: Relations to write to the graph store.
        """
        if not chunks and not entities and not relations:
            return
        ops = [self._write_chunks(chunks)]
        if entities or relations:
            ops.append(self._write_graph(entities, relations))
        await asyncio.gather(*ops)

    async def _write_chunks(self, chunks: list[Chunk]) -> None:
        """Write chunks to pgvector.

        Args:
            chunks: Chunks to embed and persist.
        """
        if not chunks:
            return
        await self.pg.upsert_chunks(self.session_id, chunks)

    async def _write_graph(self, entities: list[Entity], relations: list[Relation]) -> None:
        """Write entities and relations via GraphBackend or legacy mg_driver.

        Args:
            entities: Entities to merge.
            relations: Relations to merge.
        """
        if not entities and not relations:
            return

        if self.backend is not None:
            await asyncio.gather(
                self.backend.write_entities(self.session_id, entities),
                self.backend.write_relations(self.session_id, relations),
            )
            return

        if self.mg is not None:
            def _sync() -> None:
                with self.mg.session() as s:
                    for e in entities:
                        s.run(
                            "MERGE (n:Entity {id: $id, session_id: $sid}) SET n += $props",
                            id=e.id,
                            sid=self.session_id,
                            props={
                                "name": e.name,
                                "label": e.label,
                                "confidence": e.confidence,
                                "doc_ids": e.doc_ids,
                            },
                        )
                    for r in relations:
                        s.run(
                            """MATCH (a:Entity {id:$src, session_id:$sid}), (b:Entity {id:$tgt, session_id:$sid})
                               MERGE (a)-[rel:RELATES {type:$rtype, session_id:$sid}]->(b)""",
                            src=r.source,
                            tgt=r.target,
                            rtype=r.type,
                            sid=self.session_id,
                        )

            await asyncio.to_thread(_sync)


def _compute_duplication_ratio(
    new_chunks: list[Chunk],
    existing_chunks: list[Chunk],
    embedder: SentenceTransformer,
) -> float:
    """Compute duplication ratio using cosine similarity of chunk embeddings.

    For each new chunk, computes max cosine similarity against existing chunks.
    Returns the fraction of new chunks whose max similarity exceeds 0.85.

    Args:
        new_chunks: Newly generated chunks.
        existing_chunks: Previously stored chunks.
        embedder: Sentence-transformer model for embedding.

    Returns:
        Duplication ratio between 0.0 and 1.0.
    """
    if not new_chunks:
        return 0.0
    if not existing_chunks:
        return 0.0

    existing_texts = [c.text for c in existing_chunks if c.text]
    if len(existing_texts) < 2:
        return 0.0

    new_texts = [c.text for c in new_chunks if c.text]
    if not new_texts:
        return 0.0

    existing_embs: np.ndarray = embedder.encode(existing_texts, show_progress_bar=False)
    new_embs: np.ndarray = embedder.encode(new_texts, show_progress_bar=False)

    # Ensure 2D arrays (n_samples, dim) — encode may return 1D for single text
    if existing_embs.ndim == 1:
        existing_embs = existing_embs.reshape(1, -1)
    if new_embs.ndim == 1:
        new_embs = new_embs.reshape(1, -1)

    existing_norm = existing_embs / np.linalg.norm(existing_embs, axis=1, keepdims=True)
    new_norm = new_embs / np.linalg.norm(new_embs, axis=1, keepdims=True)

    sims = new_norm @ existing_norm.T
    max_sims = sims.max(axis=1)
    dup_count = int((max_sims > 0.85).sum())
    return dup_count / len(new_chunks)


def _novelty_check(
    state: InvestigativeState,
    new_chunks: list[Chunk],
    embedder: SentenceTransformer | None = None,
) -> float:
    """Compute duplication ratio for new chunks against existing ones.

    Uses actual cosine similarity via the embedder when available. Falls back
    to 0.0 (all novel) when embedder is None.

    Args:
        state: Current investigative state with existing chunks.
        new_chunks: Newly generated chunks to check.
        embedder: Optional sentence-transformer model. If None, returns 0.0.

    Returns:
        A dup_ratio between 0.0 and 1.0.
    """
    if not new_chunks:
        return 0.0
    if embedder is None:
        return 0.0
    existing = state.get("chunks", [])
    if not existing:
        return 0.0
    return _compute_duplication_ratio(new_chunks, existing, embedder)


WORKER_PATH = Path(__file__).resolve().parents[2] / "tools" / "graphgen_worker.py"


async def _run_extraction(raw_docs: list[RawDoc]) -> dict:
    payload = json.dumps(
        {"documents": [
            {"id": d.id, "title": d.title, "text": d.text, "source": d.source}
            for d in raw_docs
        ]}
    ).encode()
    proc = await asyncio.create_subprocess_exec(
        sys.executable, str(WORKER_PATH),
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await asyncio.wait_for(proc.communicate(input=payload), timeout=120)
    if proc.returncode != 0:
        raise RuntimeError(f"graphgen_worker failed: {stderr.decode()[:400]}")
    return json.loads(stdout)


def make_investigate_node(
    pg_client: PgVectorClient | None = None,
    mg_driver: Any = None,
    search_client: SearchServiceClient | None = None,
    backend: Any = None,
    ontology_router: Any | None = None,
    rel_extractor: Any | None = None,
) -> Any:
    """Factory for investigate_node with injected dependencies.

    Args:
        pg_client: Optional PgVectorClient for chunk persistence.
        mg_driver: Optional Memgraph driver for graph persistence (legacy).
        search_client: Optional SearchServiceClient for real document fetching.
            If None, falls back to stub documents.
        backend: Optional GraphBackend for swappable session graph.
        ontology_router: Optional OntologyRouter for domain-adaptive extraction.
            If None, ontology-guided relation extraction is skipped.
        rel_extractor: Optional OntologyRelExtractor for typed relation triples.
            If None, ontology-guided relation extraction is skipped.

    Returns:
        Async function matching the LangGraph node signature.
    """
    embedder: SentenceTransformer | None = None
    if pg_client is not None:
        embedder = pg_client.embedder

    async def investigate_node(state: InvestigativeState) -> dict[str, Any]:
        """Execute a single pass of the investigate loop.

        Increments iteration_count, runs search (real or stub), creates chunks/
        entities/relations, persists via DualWriter if dependencies are wired,
        performs a novelty check, and emits appropriate WS events.

        Args:
            state: Current investigative state.

        Returns:
            Partial state update with new chunks, entities, relations, iteration
            tracking, and ws_events.
        """
        nonlocal embedder
        iteration = state.get("iteration_count", 0) + 1
        ws_events = list(state.get("ws_events", []))
        session_id = state["session_id"]
        node_features = dict(state.get("node_features", {}))

        ws_events.append(search_start_event(session_id, iteration, state.get("queries", [])))

        if search_client is not None:
            try:
                raw_docs = await search_client.fetch_documents(
                    query=state["queries"][0] if state["queries"] else state["topic"],
                    sources=_pick_sources(state.get("plan", [])),
                    limit=5,
                )
            except Exception:
                logger.warning(
                    "Search service call failed, falling back to stub doc",
                    exc_info=True,
                )
                raw_docs = [_make_stub_doc(iteration, state["topic"])]
        else:
            raw_docs = [_make_stub_doc(iteration, state["topic"])]

        ws_events.append(docs_fetched_event(session_id, iteration, len(raw_docs)))

        new_chunks = [
            Chunk(
                id=f"chunk-{uuid.uuid4().hex[:8]}",
                doc_id=doc.id,
                text=doc.text,
                embedding=None,
                metadata={"source": doc.source, "iteration": iteration},
            )
            for doc in raw_docs
        ]

        new_entities: list[Entity] = []
        new_relations: list[Relation] = []
        if raw_docs:
            try:
                result = await _run_extraction(raw_docs)
                raw_entities = result.get("entities", []) or []
                raw_relations = result.get("relations", []) or []
            except Exception:
                logger.warning("Entity extraction failed, falling back to stub", exc_info=True)
                raw_entities = []
                raw_relations = []
            if raw_entities:
                name_to_entity: dict[str, Entity] = {}
                for ent in raw_entities:
                    e = Entity(
                        id=ent.get("id") or f"ent-{uuid.uuid4().hex[:8]}",
                        name=ent.get("name", ""),
                        label=ent.get("type", "CONCEPT"),
                        confidence=0.5,
                        doc_ids=[ent["doc_id"]] if ent.get("doc_id") else [doc.id for doc in raw_docs],
                    )
                    name_to_entity[ent["name"]] = e
                    new_entities.append(e)

                for rel in raw_relations:
                    src_ent = name_to_entity.get(rel.get("source", ""))
                    tgt_ent = name_to_entity.get(rel.get("target", ""))
                    if src_ent is not None and tgt_ent is not None:
                        new_relations.append(Relation(
                            source=src_ent.id,
                            target=tgt_ent.id,
                            type=rel.get("type", "RELATED_TO"),
                            confidence=rel.get("confidence", 0.5),
                        ))

        if not new_entities:
            new_entities = [
                Entity(
                    id=f"ent-{uuid.uuid4().hex[:8]}",
                    name=f"Entity_{iteration}",
                    label="ORG",
                    confidence=0.85,
                    doc_ids=[doc.id for doc in raw_docs],
                ),
            ]
            new_relations = [
                Relation(
                    source=new_entities[0].id,
                    target=f"ent-{uuid.uuid4().hex[:8]}",
                    type="RELATED_TO",
                    confidence=0.7,
                ),
            ]

        has_graph_writer = (pg_client is not None and mg_driver is not None) or (
            pg_client is not None and backend is not None
        )
        if has_graph_writer:
            writer = DualWriter(session_id, pg_client, mg_driver, backend)
            await writer.write(new_chunks, new_entities, new_relations)
            logger.debug(
                "DualWriter persisted %d chunks, %d entities, %d relations for session %s iter %d",
                len(new_chunks),
                len(new_entities),
                len(new_relations),
                session_id,
                iteration,
            )

            if backend is not None:
                try:
                    nf = await backend.compute_node_features(session_id)
                    node_features = {
                        eid: {
                            "pagerank": ef.pagerank,
                            "betweenness": ef.betweenness,
                            "community": ef.community,
                        }
                        for eid, ef in nf.by_entity_id.items()
                    }
                except Exception as exc:
                    logger.warning("GDS feature computation failed (non-fatal): %s", exc)

        ws_events.append(chunks_written_event(session_id, iteration, len(new_chunks)))
        ws_events.append(
            entities_written_event(session_id, iteration, len(new_entities), len(new_relations))
        )
        ws_events.append(
            graph_update_event(
                session_id=session_id,
                iteration=iteration,
                entities=[{"id": e.id, "name": e.name, "label": e.label} for e in new_entities],
                relations=[{"source": r.source, "target": r.target, "type": r.type} for r in new_relations],
            )
        )

        # Ontology-guided relation extraction (non-blocking — failures are non-fatal)
        if ontology_router is not None and rel_extractor is not None and raw_docs:
            try:
                combined_text = " ".join(d.text for d in raw_docs)[:2000]
                ontology = await asyncio.to_thread(ontology_router.select, combined_text)
                entity_dicts = [{"name": e.name, "label": e.label} for e in new_entities]
                onto_triples = await asyncio.to_thread(
                    rel_extractor.extract,
                    combined_text,
                    entity_dicts,
                    ontology.summary_jsonld,
                    ontology.id,
                )
                if onto_triples and backend is not None and hasattr(backend, "write_ontology_relations"):
                    await backend.write_ontology_relations(session_id, onto_triples, ontology.id)
                ws_events.append(
                    relations_extracted_event(
                        session_id=session_id,
                        iteration=iteration,
                        count=len(onto_triples),
                        ontology_id=ontology.id,
                        ontology_name=ontology.name,
                    )
                )
                logger.debug(
                    "Ontology extraction [%s]: %d triples for session %s iter %d",
                    ontology.id, len(onto_triples), session_id, iteration,
                )
            except Exception as exc:
                logger.warning("Ontology-guided extraction failed (non-fatal): %s", exc)

        dup_ratio = _novelty_check(state, new_chunks, embedder)
        ws_events.append(novelty_check_event(session_id, iteration, dup_ratio))

        if dup_ratio > 0.6 or iteration >= 12:
            reason = "dup_ratio exceeded 0.6" if dup_ratio > 0.6 else "max iterations reached"
            ws_events.append(search_done_event(session_id, reason, iteration))

        return {
            "iteration_count": iteration,
            "dup_ratio": dup_ratio,
            "raw_docs": [*state.get("raw_docs", []), *raw_docs],
            "chunks": [*state.get("chunks", []), *new_chunks],
            "entities": [*state.get("entities", []), *new_entities],
            "relations": [*state.get("relations", []), *new_relations],
            "node_features": node_features,
            "ws_events": ws_events,
        }

    return investigate_node


investigate_node = make_investigate_node()

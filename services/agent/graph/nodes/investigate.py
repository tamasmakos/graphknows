"""Investigate node: single-pass search iteration with dual-write and novelty check."""
from __future__ import annotations

import asyncio
import logging
import math
import uuid
from typing import TYPE_CHECKING, Any

import numpy as np
from sentence_transformers import SentenceTransformer

from services.agent.graph.events import (
    chunks_written_event,
    docs_fetched_event,
    entities_written_event,
    novelty_check_event,
    search_done_event,
    search_start_event,
)
from services.agent.graph.state import Chunk, Entity, InvestigativeState, RawDoc, Relation

if TYPE_CHECKING:
    from services.agent.infra.pgvector_client import PgVectorClient

logger = logging.getLogger(__name__)

EMBED_MODEL: str = "BAAI/bge-small-en-v1.5"


class DualWriter:
    """Persists chunks to pgvector and entities/relations to Memgraph.

    Attributes:
        session_id: Active session namespace.
        pg: PgVectorClient instance for chunk persistence.
        mg: Memgraph (Neo4j) driver instance for graph persistence.
    """

    def __init__(
        self,
        session_id: str,
        pg_client: PgVectorClient,
        mg_driver: Any,
    ) -> None:
        """Initialise DualWriter.

        Args:
            session_id: Session namespace for all writes.
            pg_client: Initialised PgVectorClient.
            mg_driver: Neo4j driver connected to Memgraph.
        """
        self.session_id = session_id
        self.pg = pg_client
        self.mg = mg_driver

    async def write(
        self,
        chunks: list[Chunk],
        entities: list[Entity],
        relations: list[Relation],
    ) -> None:
        """Persist chunks to pgvector and entities/relations to Memgraph concurrently.

        Args:
            chunks: Chunks to write to the vector store.
            entities: Entities to write to the graph store.
            relations: Relations to write to the graph store.
        """
        if not chunks and not entities and not relations:
            return
        await asyncio.gather(
            self._write_chunks(chunks),
            self._write_graph(entities, relations),
        )

    async def _write_chunks(self, chunks: list[Chunk]) -> None:
        """Write chunks to pgvector.

        Args:
            chunks: Chunks to embed and persist.
        """
        if not chunks:
            return
        await self.pg.upsert_chunks(self.session_id, chunks)

    async def _write_graph(self, entities: list[Entity], relations: list[Relation]) -> None:
        """Write entities and relations to Memgraph via Cypher.

        Args:
            entities: Entities to merge.
            relations: Relations to merge.
        """
        if not entities and not relations:
            return

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


def make_investigate_node(
    pg_client: PgVectorClient | None = None,
    mg_driver: Any = None,
) -> Any:
    """Factory for investigate_node with injected dependencies.

    Args:
        pg_client: Optional PgVectorClient for chunk persistence.
        mg_driver: Optional Memgraph driver for graph persistence.

    Returns:
        Async function matching the LangGraph node signature.
    """
    embedder: SentenceTransformer | None = None
    if pg_client is not None:
        embedder = pg_client.embedder

    async def investigate_node(state: InvestigativeState) -> dict[str, Any]:
        """Execute a single pass of the investigate loop.

        Increments iteration_count, runs placeholder search, creates chunks/
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

        ws_events.append(search_start_event(session_id, iteration, state.get("queries", [])))

        stub_doc_id = f"doc-{uuid.uuid4().hex[:8]}"
        raw_docs = [
            RawDoc(
                id=stub_doc_id,
                title=f"Stub Document {iteration}",
                text=f"This is placeholder content for search iteration {iteration} on topic: {state['topic']}",
                url=f"https://example.test/doc/{stub_doc_id}",
                source="stub",
                date=None,
            ),
        ]
        ws_events.append(docs_fetched_event(session_id, iteration, len(raw_docs)))

        new_chunks = [
            Chunk(
                id=f"chunk-{uuid.uuid4().hex[:8]}",
                doc_id=stub_doc_id,
                text=raw_docs[0].text,
                embedding=None,
                metadata={"source": "stub", "iteration": iteration},
            ),
        ]

        new_entities = [
            Entity(
                id=f"ent-{uuid.uuid4().hex[:8]}",
                name=f"Entity_{iteration}",
                label="ORG",
                confidence=0.85,
                doc_ids=[stub_doc_id],
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

        if pg_client is not None and mg_driver is not None:
            writer = DualWriter(session_id, pg_client, mg_driver)
            await writer.write(new_chunks, new_entities, new_relations)
            logger.debug(
                "DualWriter persisted %d chunks, %d entities, %d relations for session %s iter %d",
                len(new_chunks),
                len(new_entities),
                len(new_relations),
                session_id,
                iteration,
            )

        ws_events.append(chunks_written_event(session_id, iteration, len(new_chunks)))
        ws_events.append(
            entities_written_event(session_id, iteration, len(new_entities), len(new_relations))
        )

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
            "ws_events": ws_events,
        }

    return investigate_node


investigate_node = make_investigate_node()

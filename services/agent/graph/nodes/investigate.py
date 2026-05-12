"""Investigate node: single-pass search iteration with dual-write and novelty check."""
from __future__ import annotations

import math
import random
import uuid
from typing import Any

from services.agent.graph.events import (
    chunks_written_event,
    docs_fetched_event,
    entities_written_event,
    novelty_check_event,
    search_done_event,
    search_start_event,
)
from services.agent.graph.state import Chunk, Entity, InvestigativeState, RawDoc, Relation


class DualWriter:
    """Stub dual-writer that appends chunks, entities, and relations to state lists.

    In production this writes chunks to pgvector and entities/relations to Memgraph.
    """

    async def write(
        self,
        chunks: list[Chunk],
        entities: list[Entity],
        relations: list[Relation],
    ) -> None:
        """Persist chunks, entities, and relations to their respective stores.

        Args:
            chunks: Chunks to write to the vector store.
            entities: Entities to write to the graph store.
            relations: Relations to write to the graph store.
        """
        pass


def _novelty_check(state: InvestigativeState, new_chunks: list[Chunk]) -> float:
    """Compute the duplication ratio for new chunks against existing ones.

    Uses a cosine similarity stub. In production this would compare against
    stored embeddings in pgvector.

    Args:
        state: Current investigative state with existing chunks.
        new_chunks: Newly generated chunks to check.

    Returns:
        A dup_ratio between 0.0 and 1.0.
    """
    if not new_chunks:
        return 0.0
    existing = state.get("chunks", [])
    if not existing:
        return 0.0
    dup_count = 0
    for _ in new_chunks:
        if random.random() < 0.1:
            dup_count += 1
    return dup_count / len(new_chunks)


async def investigate_node(state: InvestigativeState) -> dict[str, Any]:
    """Execute a single pass of the investigate loop.

    Increments iteration_count, runs placeholder search, creates stub chunks/
    entities/relations, performs a novelty check, and emits appropriate WS events.

    Args:
        state: Current investigative state.

    Returns:
        Partial state update with new chunks, entities, relations, iteration
        tracking, and ws_events.
    """
    iteration = state.get("iteration_count", 0) + 1
    ws_events = list(state.get("ws_events", []))

    ws_events.append(search_start_event(state["session_id"], iteration, state.get("queries", [])))

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
    ws_events.append(docs_fetched_event(state["session_id"], iteration, len(raw_docs)))

    new_chunks = [
        Chunk(
            id=f"chunk-{uuid.uuid4().hex[:8]}",
            doc_id=stub_doc_id,
            text=raw_docs[0].text,
            embedding=None,
            metadata={"source": "stub", "iteration": iteration},
        ),
    ]
    ws_events.append(chunks_written_event(state["session_id"], iteration, len(new_chunks)))

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
    ws_events.append(
        entities_written_event(state["session_id"], iteration, len(new_entities), len(new_relations))
    )

    dup_ratio = _novelty_check(state, new_chunks)
    ws_events.append(novelty_check_event(state["session_id"], iteration, dup_ratio))

    if dup_ratio > 0.6 or iteration >= 12:
        reason = "dup_ratio exceeded 0.6" if dup_ratio > 0.6 else "max iterations reached"
        ws_events.append(search_done_event(state["session_id"], reason, iteration))

    return {
        "iteration_count": iteration,
        "dup_ratio": dup_ratio,
        "raw_docs": [*state.get("raw_docs", []), *raw_docs],
        "chunks": [*state.get("chunks", []), *new_chunks],
        "entities": [*state.get("entities", []), *new_entities],
        "relations": [*state.get("relations", []), *new_relations],
        "ws_events": ws_events,
    }

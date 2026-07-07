"""DialogCapture: record agent/user conversation turns into STM interaction graph.

Migrated from graphknows.memory.dialog (DialogWriter).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from graphknows.memory.stm._entity_writer import write_entity_nodes_and_edges, write_relation_edges

log = logging.getLogger("graphknows.memory.stm.dialog")


@dataclass
class Turn:
    """A single conversation turn.

    Attributes:
        id: Unique turn identifier.
        index: Zero-based position within the dialog.
        role: One of 'user', 'agent', 'system'.
        text: Full turn text.
        timestamp: Unix timestamp float.
    """

    id: str
    index: int
    role: str
    text: str
    timestamp: float = field(default_factory=time.time)


@dataclass
class DialogResult:
    """Result of writing a dialog turn."""

    dialog_id: str
    turn_id: str
    entity_count: int
    errors: list[str] = field(default_factory=list)


class DialogCapture:
    """Write conversation turns into the STM interaction graph.

    Creates and maintains DIALOG → TURN → ENTITY nodes in ArcadeDB STM
    and embeds turn text into the vector store.

    Attributes:
        _mg: ArcadeDBSTMStore for interaction graph writes.
        _pg: ArcadeDBVectorStore for stm_turns writes.
        _extractor: Entity extractor for turn text NER (optional).
        _embed_fn: Embedding callable.
        _prev_turn_ids: Tracks the last turn ID per dialog for NEXT edge linking.
    """

    def __init__(
        self,
        mg_store: Any,
        pg_store: Any,
        extractor: Any | None = None,
        embed_fn: Any | None = None,
    ) -> None:
        self._mg = mg_store
        self._pg = pg_store
        self._extractor = extractor
        if embed_fn is not None:
            self._embed_fn = embed_fn
        else:
            from graphknows.memory.stores.embedder import embed_one

            self._embed_fn = embed_one
        self._prev_turn_ids: dict[str, str] = {}

    async def write_turn(
        self,
        session_id: str,
        dialog_id: str,
        turn: Turn,
        extra_metadata: dict[str, Any] | None = None,
    ) -> DialogResult:
        """Write a single turn into the interaction graph.

        Args:
            session_id: Parent session namespace (for entity scoping).
            dialog_id: Parent dialog identifier.
            turn: Turn data including text, role, tool calls.
            extra_metadata: Optional metadata stored with the vector store turn row.

        Returns:
            DialogResult with entity_count and any errors.
        """
        errors: list[str] = []
        entity_count = 0

        # TURN node + NEXT edge
        try:
            await self._mg.write_turn_node(
                dialog_id=dialog_id,
                session_id=session_id,
                turn_id=turn.id,
                turn_index=turn.index,
                role=turn.role,
                text=turn.text,
                timestamp=str(turn.timestamp),
            )
            prev = self._prev_turn_ids.get(dialog_id)
            if prev:
                await self._mg.write_turn_sequence(prev_turn_id=prev, next_turn_id=turn.id)
            self._prev_turn_ids[dialog_id] = turn.id
        except Exception as exc:
            errors.append(f"turn_node: {exc}")

        # Entity and relation extraction from turn text.
        # Extraction failure propagates; entity-write errors are captured; relation-write
        # errors propagate (no fallback per design).
        if self._extractor and turn.text.strip():
            extraction = self._extractor.extract(turn.text)

            async def _turn_edge_writer(ent: dict[str, Any]) -> None:
                await self._mg.write_turn_entity_edge(
                    turn_id=turn.id,
                    entity_name=ent["name"],
                    session_id=session_id,
                    confidence=ent.get("score", 1.0),
                )

            try:
                entity_count += await write_entity_nodes_and_edges(
                    mg=self._mg,
                    session_id=session_id,
                    entities=extraction.entities,
                    edge_writer=_turn_edge_writer,
                )
            except Exception as exc:
                errors.append(f"entity_write: {exc}")
            await write_relation_edges(self._mg, session_id, extraction.relations)

        # Embed turn text → stm_turns
        try:
            embedding = self._embed_fn(turn.text) if turn.text.strip() else []
            turn_record: dict[str, Any] = {
                "id": turn.id,
                "session_id": session_id,
                "dialog_id": dialog_id,
                "turn_index": turn.index,
                "role": turn.role,
                "text": turn.text,
                "embedding": embedding,
            }
            if extra_metadata:
                turn_record["metadata"] = extra_metadata
            await self._pg.write_stm_turn(turn_record)
        except Exception as exc:
            errors.append(f"pg_stm_turn: {exc}")

        if errors:
            log.warning("write_turn %s: %d errors: %s", turn.id, len(errors), errors[:2])

        return DialogResult(
            dialog_id=dialog_id,
            turn_id=turn.id,
            entity_count=entity_count,
            errors=errors,
        )

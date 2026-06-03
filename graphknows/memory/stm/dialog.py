"""DialogCapture: record agent/user conversation turns into STM interaction graph.

Migrated from graphknows.memory.dialog (DialogWriter).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from graphknows.memory.stm._async_bridge import _run

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

    Creates and maintains DIALOG → TURN → ENTITY nodes in Memgraph
    and embeds turn text into pgvector stm_turns.

    # TODO(Wave 3): replace direct store references with IGraphDb adapter.

    Attributes:
        _mg: MemgraphStore for interaction graph writes.
        _pg: PgVectorStore for stm_turns writes.
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

    def write_turn(
        self,
        session_id: str,
        dialog_id: str,
        turn: Turn,
    ) -> DialogResult:
        """Write a single turn into the interaction graph.

        Args:
            session_id: Parent session namespace (for entity scoping).
            dialog_id: Parent dialog identifier.
            turn: Turn data including text, role, tool calls.

        Returns:
            DialogResult with entity_count and any errors.
        """
        errors: list[str] = []
        entity_count = 0

        # TURN node + NEXT edge
        try:
            _run(
                self._mg.write_turn_node(
                    dialog_id=dialog_id,
                    session_id=session_id,
                    turn_id=turn.id,
                    turn_index=turn.index,
                    role=turn.role,
                    text=turn.text,
                    timestamp=str(turn.timestamp),
                )
            )
            prev = self._prev_turn_ids.get(dialog_id)
            if prev:
                _run(self._mg.write_turn_sequence(prev_turn_id=prev, next_turn_id=turn.id))
            self._prev_turn_ids[dialog_id] = turn.id
        except Exception as exc:
            errors.append(f"turn_node: {exc}")

        # Entity extraction from turn text
        if self._extractor and turn.text.strip():
            try:
                entities = self._extractor.extract(turn.text).entities
                for ent in entities:
                    _run(
                        self._mg.write_entity_node(
                            session_id=session_id,
                            entity_name=ent["name"],
                            entity_type=ent.get("type", "ENTITY"),
                            confidence=ent.get("score", 1.0),
                        )
                    )
                    _run(
                        self._mg.write_turn_entity_edge(
                            turn_id=turn.id,
                            entity_name=ent["name"],
                            session_id=session_id,
                            confidence=ent.get("score", 1.0),
                        )
                    )
                    entity_count += 1
            except Exception as exc:
                errors.append(f"entity_extract: {exc}")

        # Embed turn text → stm_turns
        try:
            embedding = self._embed_fn(turn.text) if turn.text.strip() else []
            _run(
                self._pg.write_stm_turn(
                    {
                        "id": turn.id,
                        "session_id": session_id,
                        "dialog_id": dialog_id,
                        "turn_index": turn.index,
                        "role": turn.role,
                        "text": turn.text,
                        "embedding": embedding,
                    }
                )
            )
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

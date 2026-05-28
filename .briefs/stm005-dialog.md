# STM-005: DialogWriter — interaction graph (turns, participants, tool calls) into STM

## Context
Wiki: /root/graphknows/llm-wiki/architecture/memory-layer-v2.md
Project root: /root/graphknows
Read .hermes/instructions/python-style.instructions.md first.
Read .hermes/instructions/tdd-test-driven-development.instructions.md first.
Model: github-copilot/claude-sonnet-4.6

Depends on: STM-001 (Memgraph write methods for DIALOG/TURN/PARTICIPANT/TOOL_CALL)

## Goal
DialogWriter writes agent/user conversation turns into the STM interaction graph.
Each turn text also gets entity extraction (GLiNER2) and is embedded into stm_turns.
Tool calls from the agent are recorded as TOOL_CALL nodes linked to the turn.

## Create services/graphknows/graphknows/memory/dialog.py

```python
"""DialogWriter: record agent/user conversation turns into STM interaction graph."""
from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass, field
from typing import Any

from graphknows.retrieval.stores.embedder import embed_one
from graphknows.retrieval.stores.memgraph import MemgraphStore
from graphknows.retrieval.stores.pgvector import PgVectorStore

log = logging.getLogger("graphknows.memory.dialog")


@dataclass
class ToolCall:
    """A tool invocation recorded within a turn.

    Attributes:
        name: Tool function name (e.g. "search_eu", "retrieve").
        args: Raw arguments dict — stored as hash for privacy.
        result_summary: Short summary of result (optional, for human readability).
        latency_ms: Execution latency in milliseconds.
    """
    name: str
    args: dict[str, Any] = field(default_factory=dict)
    result_summary: str = ""
    latency_ms: float = 0.0


@dataclass
class Turn:
    """A single conversation turn.

    Attributes:
        id: Unique turn identifier.
        index: Zero-based position within the dialog.
        role: One of 'user', 'agent', 'system'.
        text: Full turn text.
        timestamp: Unix timestamp float.
        tool_calls: Tool calls made during this turn (agent turns only).
    """
    id: str
    index: int
    role: str  # 'user' | 'agent' | 'system'
    text: str
    timestamp: float = field(default_factory=time.time)
    tool_calls: list[ToolCall] = field(default_factory=list)


@dataclass
class Participant:
    """A dialog participant.

    Attributes:
        id: Unique participant identifier.
        role: 'user' | 'agent' | 'system'.
        display_name: Optional human-readable name.
    """
    id: str
    role: str
    display_name: str = ""


@dataclass
class DialogResult:
    """Result of writing a dialog turn."""
    dialog_id: str
    turn_id: str
    entity_count: int
    errors: list[str] = field(default_factory=list)


class DialogWriter:
    """Write conversation turns into the STM interaction graph.

    Creates and maintains DIALOG → TURN → PARTICIPANT + ENTITY + TOOL_CALL
    nodes in Memgraph and embeds turn text into pgvector stm_turns.

    Attributes:
        _mg: MemgraphStore for interaction graph writes.
        _pg: PgVectorStore for stm_turns writes.
        _extractor: Entity extractor for turn text NER (optional).
        _prev_turn_ids: Tracks the last turn ID per dialog for NEXT edge linking.
    """

    def __init__(
        self,
        mg_store: MemgraphStore,
        pg_store: PgVectorStore,
        extractor: Any | None = None,
    ) -> None:
        self._mg = mg_store
        self._pg = pg_store
        self._extractor = extractor
        self._prev_turn_ids: dict[str, str] = {}  # dialog_id → last turn_id

    def start_dialog(
        self,
        session_id: str,
        dialog_id: str,
        participants: list[Participant],
    ) -> None:
        """Initialize a DIALOG node and register participants.

        Args:
            session_id: Parent session namespace.
            dialog_id: Unique dialog identifier.
            participants: List of participants (user, agent, system).
        """
        try:
            self._mg.write_dialog_node(
                session_id=session_id,
                dialog_id=dialog_id,
                started_at=time.time(),
            )
            for p in participants:
                self._mg.write_participant_node(
                    dialog_id=dialog_id,
                    participant_id=p.id,
                    role=p.role,
                    display_name=p.display_name,
                )
        except Exception as exc:
            log.warning("start_dialog %s: %s", dialog_id, exc)

    def write_turn(
        self,
        session_id: str,
        dialog_id: str,
        turn: Turn,
    ) -> DialogResult:
        """Write a single turn into the interaction graph.

        Performs: TURN node, NEXT edge from previous turn, SPOKEN_BY edge,
        entity extraction + MENTIONS edges, TOOL_CALL nodes, stm_turns embedding.

        Args:
            session_id: Parent session namespace (for entity scoping).
            dialog_id: Parent dialog identifier.
            turn: Turn data including text, role, tool calls.

        Returns:
            DialogResult with entity_count and any errors.
        """
        errors: list[str] = []
        entity_count = 0

        # TURN node
        try:
            self._mg.write_turn_node(
                dialog_id=dialog_id,
                turn_id=turn.id,
                turn_index=turn.index,
                role=turn.role,
                text=turn.text,
                timestamp=turn.timestamp,
            )
            # NEXT edge
            prev = self._prev_turn_ids.get(dialog_id)
            if prev:
                self._mg.write_turn_next_edge(prev_turn_id=prev, next_turn_id=turn.id)
            self._prev_turn_ids[dialog_id] = turn.id
        except Exception as exc:
            errors.append(f"turn_node: {exc}")

        # Entity extraction from turn text
        if self._extractor and turn.text.strip():
            try:
                entities = self._extractor.extract(turn.text)
                for ent in entities:
                    self._mg.write_entity_node(
                        session_id=session_id,
                        entity_name=ent["name"],
                        entity_type=ent.get("type", "ENTITY"),
                        confidence=ent.get("score", 1.0),
                    )
                    self._mg.write_turn_entity_edge(
                        turn_id=turn.id,
                        entity_name=ent["name"],
                        session_id=session_id,
                        confidence=ent.get("score", 1.0),
                    )
                    entity_count += 1
            except Exception as exc:
                errors.append(f"entity_extract: {exc}")

        # Tool calls
        for tc in turn.tool_calls:
            try:
                args_hash = hashlib.sha256(str(tc.args).encode()).hexdigest()[:8]
                self._mg.write_tool_call_node(
                    turn_id=turn.id,
                    name=tc.name,
                    args_hash=args_hash,
                    latency_ms=tc.latency_ms,
                )
            except Exception as exc:
                errors.append(f"tool_call {tc.name}: {exc}")

        # Embed turn text → stm_turns
        try:
            embedding = embed_one(turn.text) if turn.text.strip() else []
            import asyncio
            asyncio.get_event_loop().run_until_complete(
                self._pg.write_stm_turn({
                    "id": turn.id,
                    "session_id": session_id,
                    "dialog_id": dialog_id,
                    "turn_index": turn.index,
                    "role": turn.role,
                    "text": turn.text,
                    "embedding": embedding,
                })
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
```

## Add to MemgraphStore (STM-001 should have these, verify and add if missing):
- `write_participant_node(dialog_id, participant_id, role, display_name) -> None`
- `write_turn_next_edge(prev_turn_id, next_turn_id) -> None`
- `write_turn_entity_edge(turn_id, entity_name, session_id, confidence) -> None`
- Check existing memgraph.py — add only what's truly missing

## Write tests/test_dialog_writer.py

Test cases (all unit, MagicMock for mg/pg/extractor):
1. start_dialog() calls mg.write_dialog_node() and write_participant_node() for each participant
2. write_turn() calls mg.write_turn_node() with correct args
3. write_turn() calls mg.write_turn_next_edge() on second turn in same dialog
4. write_turn() does NOT call write_turn_next_edge() on first turn (no prev)
5. write_turn() calls extractor.extract(turn.text) when extractor is set
6. write_turn() calls mg.write_entity_node() for each extracted entity
7. write_turn() calls mg.write_tool_call_node() for each tool call in Turn.tool_calls
8. write_turn() calls pg.write_stm_turn() with correct embedding
9. write_turn() — extractor failure appended to errors, does NOT raise
10. write_turn() — pg failure appended to errors, does NOT raise
11. DialogResult.entity_count matches number of extracted entities
12. _prev_turn_ids updated after write_turn(); second call uses first turn ID for NEXT edge

## Acceptance criteria
- All 12 tests pass
- ruff + mypy clean
- check_imports.py clean
- No live DB or model calls in tests

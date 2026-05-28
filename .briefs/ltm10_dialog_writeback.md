# LTM-10: Dialog Writeback — SESSION→DIALOG→UTTERANCE→ENTITY paths from WS stream

## Context

GraphKnows agent emits WebSocket events per turn:
- `thought_token` — internal reasoning
- `tool_start` / `tool_result` — tool calls
- `answer_token` — streamed final answer
- `graph_update` — entity/relation updates

The session ends via `session/end.py` → `merge_session_to_neo4j()`.

LTM-8 added the schema. This task wires up the dialog path:

```
(Session/Segment)-[HAS_DIALOG]->(Dialog)
(Dialog)-[PARTICIPANT]->(Participant {role: "agent"|"user", name})
(Participant)-[SAID]->(Utterance {text, turn_index, timestamp, embedding})
(Utterance)-[MENTIONS]->(Entity)
```

A "Dialog" is the full conversation for one session. Each user message + agent response
is one turn pair, stored as two Utterance nodes (one per participant).

## What to Build

### 1. NEW: `services/agent/session/dialog.py`

A `DialogWriteBack` class that:
- Accepts a list of turn dicts (from session state)
- Writes Dialog + Participant + Utterance nodes to Neo4j
- Runs lightweight NER on each utterance text → MENTIONS edges to Entity nodes

```python
"""Dialog writeback: session turns → Neo4j Dialog/Utterance/Participant nodes."""
from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

log = logging.getLogger(__name__)


@dataclass
class TurnData:
    """A single conversation turn."""
    turn_index: int
    role: str          # "user" or "agent"
    participant_name: str  # e.g. "user", "GraphKnows-Agent"
    text: str
    timestamp: str     # ISO 8601
    embedding: list[float] = field(default_factory=list)


@dataclass
class DialogResult:
    dialog_id: str
    session_id: str
    utterance_count: int
    mention_count: int
    errors: list[str] = field(default_factory=list)


class DialogWriteBack:
    """Write session dialog turns to Neo4j as Dialog/Utterance/Entity paths."""

    def __init__(
        self,
        neo4j_driver: Any,
        embedding_model: Any | None = None,
        ner_ontology: list[str] | None = None,
    ) -> None:
        self._driver = neo4j_driver
        self._embed = embedding_model
        self._ontology = ner_ontology or [
            "PERSON", "ORGANIZATION", "LOCATION", "CONCEPT",
            "POLICY", "LAW", "INSTITUTION", "TECHNOLOGY",
        ]

    def write(self, session_id: str, turns: list[TurnData]) -> DialogResult:
        """Write dialog turns to Neo4j. Idempotent via MERGE on dialog_id + utterance_id."""
        ...
```

### Turn-to-Utterance mapping

Each `TurnData` becomes:
- `dialog_id = sha256(session_id)[:12]`
- `utterance_id = sha256(f"{session_id}:{turn.turn_index}:{turn.role}")[:16]`
- `participant_id = sha256(f"{session_id}:{turn.role}:{turn.participant_name}")[:12]`

### Cypher pattern (write_neo4j, single transaction)

```cypher
-- Dialog node (one per session)
MERGE (dia:Dialog {id: $dialog_id})
SET dia.session_id = $session_id, dia.started_at = $started_at

-- Link to Segment (session node)
MATCH (seg:Segment {session_id: $session_id})
MERGE (seg)-[:HAS_DIALOG]->(dia)

-- Per turn: Participant + Utterance
MERGE (p:Participant {id: $participant_id})
SET p.role = $role, p.name = $participant_name
MERGE (dia)-[:PARTICIPANT]->(p)

MERGE (u:Utterance {id: $utterance_id})
SET u.text = $text, u.turn_index = $turn_index,
    u.timestamp = $timestamp, u.embedding = $embedding
MERGE (p)-[:SAID]->(u)
```

### NER on utterance text → MENTIONS

After writing all utterances, run entity extraction on each utterance text.
Use the same GLiNER2/spaCy pattern from `document_ingestor.py`:

```python
from graphknows.document_ingestor import DocumentIngestor
# Reuse _extract_entities via a lightweight shim — instantiate with driver=None
_ner = DocumentIngestor(neo4j_driver=None)
mentions = _ner._extract_entities(chunk_like, self._ontology)
```

For each mention, write:
```cypher
MATCH (e:Entity {name: $entity_name})   -- only link to EXISTING entities
MERGE (u:Utterance {id: $utterance_id})-[:MENTIONS]->(e)
```

Important: only link to **existing** Entity nodes (MATCH, not MERGE). Don't create new
entity nodes from dialog — that's the extraction pipeline's job. This keeps dialog as
a reference layer, not a creation layer.

### 2. MODIFY: `services/agent/session/end.py`

In `merge_session_to_neo4j()`, after the existing entity/relation write and feature engineering,
add a dialog writeback step:

```python
# ── Dialog writeback ──────────────────────────────────────────────────────────
from agent.session.dialog import DialogWriteBack, TurnData

turns = _extract_turns_from_state(state)  # see below
if turns:
    dialog_wb = DialogWriteBack(neo4j_driver=sync_driver)
    dialog_result = dialog_wb.write(session_id=session_id, turns=turns)
    log.info(
        "dialog writeback: session=%s utterances=%d mentions=%d",
        session_id, dialog_result.utterance_count, dialog_result.mention_count,
    )
```

### 3. NEW: `_extract_turns_from_state(state: dict) -> list[TurnData]`

The agent state contains a message history. Extract turns from:
- `state.get("messages", [])` — LangGraph message list
- Each message has `.type` or `.__class__.__name__` ("HumanMessage", "AIMessage")
- Or: check for `state.get("conversation_history", [])` which may be a list of dicts
  with `{"role": "user"|"assistant", "content": "..."}`

Handle both formats. Return empty list if neither exists — non-fatal.

```python
def _extract_turns_from_state(state: dict) -> list[TurnData]:
    turns: list[TurnData] = []
    now = datetime.now(timezone.utc).isoformat()
    
    # Format 1: LangGraph messages list
    messages = state.get("messages", [])
    for i, msg in enumerate(messages):
        role = getattr(msg, "type", None) or (
            "user" if getattr(msg, "__class__", None) and "Human" in msg.__class__.__name__
            else "agent"
        )
        content = getattr(msg, "content", str(msg))
        if not isinstance(content, str):
            content = str(content)
        turns.append(TurnData(
            turn_index=i,
            role="user" if role in ("human", "user") else "agent",
            participant_name="user" if role in ("human", "user") else "GraphKnows-Agent",
            text=content[:2000],  # cap at 2000 chars
            timestamp=now,
        ))
    
    if not turns:
        # Format 2: conversation_history list of dicts
        for i, entry in enumerate(state.get("conversation_history", [])):
            role = entry.get("role", "agent")
            turns.append(TurnData(
                turn_index=i,
                role=role,
                participant_name="user" if role == "user" else "GraphKnows-Agent",
                text=str(entry.get("content", ""))[:2000],
                timestamp=entry.get("timestamp", now),
            ))
    
    return turns
```

### 4. MODIFY: `services/agent/session/end.py` sync driver

The dialog writeback needs a **sync** Neo4j driver (same pattern as `ensure_schema`).
Reuse the existing helper — extract it into a shared `_get_sync_driver(uri, user, pw)` function
in `end.py` so both `ensure_schema` and `DialogWriteBack` share the same creation pattern.

## Test to Add

### `test_dialog_writeback.py` (NEW at repo root)

```python
"""Test DialogWriteBack with synthetic turns — no LangGraph, no GPU."""
# ... (sys.path setup same as other tests)

from agent.session.dialog import DialogWriteBack, TurnData
from neo4j import GraphDatabase

NEO4J_URI = os.environ["NEO4J_URI"]
driver = GraphDatabase.driver(NEO4J_URI, auth=(os.environ.get("NEO4J_USER", "neo4j"), os.environ["NEO4J_PASSWORD"]))

# Create a test Segment so the HAS_DIALOG MATCH succeeds
SESSION_ID = "test-dialog-ltm10"
with driver.session() as s:
    s.run("MERGE (seg:Segment {session_id: $sid})", sid=SESSION_ID)

turns = [
    TurnData(0, "user", "user", "What is the EU AI Act?", "2025-01-01T10:00:00Z"),
    TurnData(1, "agent", "GraphKnows-Agent", "The EU AI Act is a regulation...", "2025-01-01T10:00:05Z"),
    TurnData(2, "user", "user", "Who proposed it?", "2025-01-01T10:00:10Z"),
    TurnData(3, "agent", "GraphKnows-Agent", "The European Commission proposed it in 2021.", "2025-01-01T10:00:15Z"),
]

wb = DialogWriteBack(neo4j_driver=driver)
result = wb.write(session_id=SESSION_ID, turns=turns)

assert result.utterance_count == 4, f"Expected 4 utterances, got {result.utterance_count}"
assert not result.errors, f"Errors: {result.errors}"

with driver.session() as s:
    dial_count = s.run("MATCH (d:Dialog {session_id: $sid}) RETURN count(d) AS c", sid=SESSION_ID).single()["c"]
    utt_count = s.run("MATCH (:Segment {session_id: $sid})-[:HAS_DIALOG]->(:Dialog)-[:PARTICIPANT]->(:Participant)-[:SAID]->(u:Utterance) RETURN count(u) AS c", sid=SESSION_ID).single()["c"]
    assert dial_count == 1, f"Expected 1 Dialog, got {dial_count}"
    assert utt_count == 4, f"Expected 4 Utterances via path, got {utt_count}"
    # Cleanup
    s.run("MATCH (seg:Segment {session_id: $sid}) DETACH DELETE seg", sid=SESSION_ID)
    s.run("MATCH (d:Dialog {session_id: $sid}) DETACH DELETE d", sid=SESSION_ID)

driver.close()
print("PASS  Dialog node created")
print("PASS  4 Utterances written via full path")
print("PASS  Cleanup complete")
print("All LTM-10 dialog writeback tests passed.")
```

## Acceptance Criteria

1. `services/agent/session/dialog.py` with `DialogWriteBack`, `TurnData`, `DialogResult`
2. Dialog, Participant, Utterance nodes written with MERGE (idempotent)
3. `(Segment)-[HAS_DIALOG]->(Dialog)` edge created for each session
4. NER runs on utterance text → `(Utterance)-[MENTIONS]->(Entity)` for existing entities only
5. `end.py` calls `_extract_turns_from_state()` and `DialogWriteBack.write()` after entity merge
6. `test_dialog_writeback.py` passes: 1 Dialog + 4 Utterances via full path + cleanup
7. Existing `test_ltm_pipeline.py` still 5/5 + schema + orphan prune green
8. Commit: `LTM-10: dialog writeback — SESSION→DIALOG→UTTERANCE→ENTITY from agent turns`

## Constraints

- Python 3.11, type hints, Google docstrings, 100-char lines, no bare except
- Neo4j driver: use sync `GraphDatabase.driver` for dialog writes (same as ensure_schema)
- `q=query` convention (not `query=query`)
- Dialog writeback in `end.py` must be non-fatal: wrap in try/except, log errors, never raise
- Do NOT create new Entity nodes from dialog — MATCH only, not MERGE
- Run: `cd /root/graphknows && /root/graphknows/services/agent/.venv/bin/python -B test_dialog_writeback.py`
- PYTHONPATH: `/root/graphknows/services:/root/graphknows/services/graphknows:/root/graphknows/services/graphknows/graphknows`
- .env at `/root/graphknows/.env`
- Do NOT commit. Leave staged for review.

## Known Facts

- `end.py` path: `/root/graphknows/services/agent/session/end.py`
- `writeback.py` path: `/root/graphknows/services/agent/session/writeback.py`
- Sync driver pattern already used in `writeback.py` `ensure_schema()` — reuse it
- Neo4j: bolt://172.18.0.1:7687 (NEO4J_URI in .env)
- Memgraph: bolt://172.18.0.1:17687, no auth
- LangGraph message types: `HumanMessage`, `AIMessage` from `langchain_core.messages`
- State dict key for messages: `state["messages"]` (list of LangGraph message objects)
- Project root: `/root/graphknows`, branch: `main`, commit: `d76735f`

# STM-001: Schema — SESSION node, full DDL, delete legacy

## Context
Wiki design doc: /root/graphknows/llm-wiki/architecture/memory-layer-v2.md
Project root: /root/graphknows  (ALWAYS work from here)
Read .hermes/instructions/python-style.instructions.md before writing any code.
Read .hermes/instructions/tdd-test-driven-development.instructions.md before writing any test.
Model: github-copilot/claude-sonnet-4.6

## Goal
Clean break from legacy ingestion. New SESSION-centric schema. No backward compatibility.

## Delete these files entirely (git rm)
- services/graphknows/graphknows/document_ingestor.py
- services/graphknows/graphknows/ingestion/pipeline.py
- services/graphknows/graphknows/migrations/001_ltm8_schema.py
- services/graphknows/graphknows/topic_summarizer.py  (will be replaced in STM-006)
- services/graphknows/graphknows/recall.py  (will be replaced in STM-007)

Also remove from pyproject.toml any entry for `graphknows-memory` if it exists.
Check: grep -n 'graphknows-memory' services/graphknows/pyproject.toml

## Rewrite services/graphknows/graphknows/schema.py

Replace ALL existing content with:

```python
"""Graph schema: constraints and indexes for Neo4j (LTM) and Memgraph (STM)."""
from __future__ import annotations
import logging
from typing import Any

log = logging.getLogger("graphknows.schema")

# Neo4j LTM constraints
CONSTRAINTS: list[str] = [
    "CREATE CONSTRAINT session_id_unique IF NOT EXISTS FOR (s:SESSION) REQUIRE s.id IS UNIQUE",
    "CREATE CONSTRAINT file_id_unique IF NOT EXISTS FOR (f:FILE) REQUIRE f.id IS UNIQUE",
    "CREATE CONSTRAINT chunk_id_unique IF NOT EXISTS FOR (c:CHUNK) REQUIRE c.id IS UNIQUE",
    "CREATE CONSTRAINT entity_name_unique IF NOT EXISTS FOR (e:ENTITY) REQUIRE e.name IS UNIQUE",
    "CREATE CONSTRAINT dialog_id_unique IF NOT EXISTS FOR (d:DIALOG) REQUIRE d.id IS UNIQUE",
    "CREATE CONSTRAINT turn_id_unique IF NOT EXISTS FOR (t:TURN) REQUIRE t.id IS UNIQUE",
    "CREATE CONSTRAINT topic_id_unique IF NOT EXISTS FOR (tp:TOPIC) REQUIRE tp.id IS UNIQUE",
]

INDEXES: list[str] = [
    "CREATE INDEX entity_session_idx IF NOT EXISTS FOR (e:ENTITY) ON (e.session_id)",
    "CREATE INDEX chunk_session_idx IF NOT EXISTS FOR (c:CHUNK) ON (c.session_id)",
    "CREATE INDEX turn_session_idx IF NOT EXISTS FOR (t:TURN) ON (t.session_id)",
    "CREATE FULLTEXT INDEX chunk_text_idx IF NOT EXISTS FOR (c:CHUNK) ON EACH [c.text]",
    "CREATE FULLTEXT INDEX turn_text_idx IF NOT EXISTS FOR (t:TURN) ON EACH [t.text]",
]

# Memgraph STM indexes (Memgraph syntax — no IF NOT EXISTS on all)
MEMGRAPH_INDEXES: list[str] = [
    "CREATE INDEX ON :SESSION(id)",
    "CREATE INDEX ON :FILE(id)",
    "CREATE INDEX ON :CHUNK(session_id)",
    "CREATE INDEX ON :ENTITY(session_id)",
    "CREATE INDEX ON :ENTITY(name)",
    "CREATE INDEX ON :DIALOG(session_id)",
    "CREATE INDEX ON :TURN(session_id)",
]


def apply_schema(session: Any) -> dict[str, Any]:
    """Apply Neo4j constraints and indexes. Idempotent. Returns counts."""
    applied: dict[str, Any] = {"constraints": 0, "indexes": 0, "errors": []}
    for stmt in CONSTRAINTS:
        try:
            session.run(stmt)
            applied["constraints"] += 1
        except Exception as exc:
            applied["errors"].append(f"constraint: {exc}")
    for stmt in INDEXES:
        try:
            session.run(stmt)
            applied["indexes"] += 1
        except Exception as exc:
            applied["errors"].append(f"index: {exc}")
    log.info("apply_schema: %d constraints, %d indexes, %d errors",
             applied["constraints"], applied["indexes"], len(applied["errors"]))
    return applied


def apply_memgraph_schema(session: Any) -> dict[str, Any]:
    """Apply Memgraph indexes. Idempotent (best-effort). Returns counts."""
    applied: dict[str, Any] = {"indexes": 0, "errors": []}
    for stmt in MEMGRAPH_INDEXES:
        try:
            session.run(stmt).consume()
            applied["indexes"] += 1
        except Exception as exc:
            applied["errors"].append(f"index: {exc}")
    return applied
```

## Update services/graphknows/graphknows/retrieval/stores/pgvector.py

Add these DDL tables at the top (replace the old _DDL constant entirely):

```sql
CREATE EXTENSION IF NOT EXISTS vector;

-- Short-term memory (ephemeral, session-scoped)
CREATE TABLE IF NOT EXISTS stm_chunks (
    id           TEXT PRIMARY KEY,
    session_id   TEXT NOT NULL,
    file_id      TEXT,
    text         TEXT NOT NULL,
    heading_path TEXT,
    embedding    VECTOR(1536),
    metadata     JSONB,
    created_at   TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS stm_chunks_session_idx ON stm_chunks (session_id);
CREATE INDEX IF NOT EXISTS stm_chunks_emb_idx ON stm_chunks USING ivfflat (embedding vector_cosine_ops) WITH (lists=20);

CREATE TABLE IF NOT EXISTS stm_turns (
    id           TEXT PRIMARY KEY,
    session_id   TEXT NOT NULL,
    dialog_id    TEXT,
    turn_index   INT,
    role         TEXT NOT NULL,  -- 'user' | 'agent' | 'system'
    text         TEXT NOT NULL,
    embedding    VECTOR(1536),
    created_at   TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS stm_turns_session_idx ON stm_turns (session_id);
CREATE INDEX IF NOT EXISTS stm_turns_emb_idx ON stm_turns USING ivfflat (embedding vector_cosine_ops) WITH (lists=20);

-- Long-term memory (permanent, flushed from STM)
CREATE TABLE IF NOT EXISTS ltm_chunks (
    id           TEXT PRIMARY KEY,
    session_id   TEXT,
    file_id      TEXT,
    text         TEXT NOT NULL,
    heading_path TEXT,
    embedding    VECTOR(1536),
    metadata     JSONB,
    flushed_at   TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS ltm_chunks_session_idx ON ltm_chunks (session_id);
CREATE INDEX IF NOT EXISTS ltm_chunks_emb_idx ON ltm_chunks USING ivfflat (embedding vector_cosine_ops) WITH (lists=50);

CREATE TABLE IF NOT EXISTS ltm_turns (
    id           TEXT PRIMARY KEY,
    session_id   TEXT,
    dialog_id    TEXT,
    turn_index   INT,
    role         TEXT,
    text         TEXT NOT NULL,
    embedding    VECTOR(1536),
    flushed_at   TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS ltm_turns_session_idx ON ltm_turns (session_id);
CREATE INDEX IF NOT EXISTS ltm_turns_emb_idx ON ltm_turns USING ivfflat (embedding vector_cosine_ops) WITH (lists=50);

CREATE TABLE IF NOT EXISTS ltm_topics (
    topic_id     TEXT PRIMARY KEY,
    summary      TEXT,
    embedding    VECTOR(1536),
    entity_names TEXT[],
    session_ids  TEXT[],
    created_at   TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS ltm_topics_emb_idx ON ltm_topics USING ivfflat (embedding vector_cosine_ops) WITH (lists=10);
```

Also add these methods to PgVectorStore class:
- `async write_stm_chunk(chunk: dict) -> None` — INSERT into stm_chunks
- `async write_stm_turn(turn: dict) -> None` — INSERT into stm_turns
- `async search_stm_chunks(session_id: str, embedding: list[float], top_k: int) -> list[dict]`
- `async search_stm_turns(session_id: str, embedding: list[float], top_k: int) -> list[dict]`
- `async promote_to_ltm(session_id: str) -> tuple[int, int]` — copy stm_chunks+stm_turns → ltm_*
- `async clear_stm(session_id: str) -> None` — DELETE from stm_* WHERE session_id=?
- `async write_ltm_topic(topic: dict) -> None` — UPSERT into ltm_topics

## Update retrieval/stores/memgraph.py
- Replace ensure_indexes() to call MEMGRAPH_INDEXES from schema.py
- Add write_session_metadata(session_id, metadata: dict) -> None
- Add write_file_node(session_id, file_id, title, parser_name, source_hint) -> None  
- Add write_chunk_node(session_id, file_id, chunk_id, text, heading_path, metadata) -> None
- Add write_entity_node(session_id, entity_name, entity_type, confidence) -> None
- Add write_chunk_entity_edge(chunk_id, entity_name, session_id) -> None
- Add write_dialog_node(session_id, dialog_id, started_at) -> None
- Add write_turn_node(dialog_id, turn_id, turn_index, role, text, timestamp) -> None
- Add write_turn_entity_edge(turn_id, entity_name, session_id, confidence) -> None
- Add write_tool_call_node(turn_id, name, args_hash, latency_ms) -> None
- Keep existing write_entities/write_relations — mark as deprecated with comment

## Create migrations/002_stm_ltm_split.py
Simple script that drops old tables (chunks, entities) and creates the new ones
via PgVectorStore.connect() + DDL execution.

## Write tests/test_schema.py
Test that apply_schema() is idempotent (call twice, no errors).
Use MagicMock for the Neo4j session object.
Test apply_memgraph_schema() similarly.
Test PgVectorStore DDL methods exist (check method signatures).

## Acceptance criteria
- `git rm` confirms deleted files are gone
- `ruff format services/ && ruff check --no-fix --config ruff.toml services/` — clean
- `python3 scripts/check_imports.py` — clean
- All tests pass: `python3 -B -m pytest tests/ -x -q -m "not integration"`
- NO sentence-transformers, NO fetch() calls, NO HTTP in schema/pgvector/memgraph code

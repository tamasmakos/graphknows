---
title: "Memory Layer v2 — Agent-Native STM/LTM"
created: 2026-05-21
updated: 2026-05-21
type: architecture
tags: [memory, ingestion, parser, stm, ltm, dialog, session, cli]
confidence: high
---

# Memory Layer v2 — Agent-Native STM/LTM

## Decision

Replace the fetch-based `DocumentIngestor` with a text-first, modular, session-aware memory layer.
Delete all legacy ingestion code. Strict STM/LTM separation with explicit CLI flush command.

**Key principles:**
- Entry point is always **text** — no HTTP fetch inside the memory layer
- Parsers output **Markdown** with preserved heading hierarchy — chunks follow heading boundaries
- Extraction receives beautiful, complete MD chunks (no words cut mid-sentence)
- Dialog (agent/user turns) is a first-class memory type with its own graph structure
- Short-term memory (Memgraph + stm_* pgvector tables) is ephemeral per session
- Long-term memory (Neo4j + ltm_* pgvector tables) is permanent, written only on explicit flush
- Topic summarization (Leiden/LLM) runs **only** at flush time — never at ingestion
- CLI is the primary interface — deep, composable, scriptable

---

## Schema

### STM (Memgraph)

```
Evidence graph:
(SESSION {id, started_at, metadata})-[:IN_SESSION]->
(FILE {id, title, source_hint, parser})-[:HAS_CHUNK]->
(CHUNK {id, text, char_offset, heading_path, token_count})-[:HAS_ENTITY]->
(ENTITY {name, type, session_id, confidence})

ENTITY <-[:RELATES {type, confidence}]-> ENTITY  (from DSPy relation extraction)

Interaction graph:
(SESSION)-[:IN_SESSION]->
(DIALOG {id, started_at, session_id})-[:HAS_TURN]->
(TURN {id, index, role, text, timestamp})-[:NEXT]->(TURN)
(TURN)-[:SPOKEN_BY]->(PARTICIPANT {id, role})  -- role: 'user' | 'agent' | 'system'
(TURN)-[:MENTIONS {confidence}]->(ENTITY)
(TURN)-[:USED_TOOL]->(TOOL_CALL {name, args_hash, latency_ms})
```

### LTM (Neo4j) — same structure + topics

```
Evidence: same as STM + (ENTITY)<-[:IN_TOPIC]-(TOPIC {id, summary, embedding, community_id})
Interaction: same as STM + (DIALOG)-[:HAS_SUMMARY]->(DIALOG_SUMMARY {text, embedding})
```

### pgvector

```sql
-- STM (ephemeral, session-scoped)
stm_chunks(id, session_id, file_id, text, heading_path, embedding VECTOR(1536), metadata JSONB, created_at)
stm_turns(id, session_id, dialog_id, turn_index, role, text, embedding VECTOR(1536), created_at)

-- LTM (permanent, flushed from STM)
ltm_chunks(id, session_id, file_id, text, heading_path, embedding VECTOR(1536), metadata JSONB, flushed_at)
ltm_turns(id, session_id, dialog_id, turn_index, role, text, embedding VECTOR(1536), flushed_at)
ltm_topics(topic_id TEXT PK, summary, embedding VECTOR(1536), entity_names TEXT[], session_ids TEXT[], created_at)
```

---

## Parser Pipeline

All parsers implement `BaseParser` and output `ParsedFile` with `TextSegment` list.

**Critical rule:** Parser outputs Markdown. Chunks follow heading hierarchy, never split mid-sentence.
Chunking strategy: parse heading tree, assign text to deepest heading, merge small sections upward.

```
BaseParser (ABC)
├── PlainTextParser     — .txt, .md, raw string; headings extracted from ##/# lines
├── CodeParser          — .py, .js, .ts; Python AST → class/function as heading + docstring + body
├── DocxParser          — .docx; python-docx headings → hierarchy; body text normalized to MD
└── [future] PdfParser, HtmlParser ...

ParsedFile:
  source_id: str          # deterministic hash
  title: str
  parser_name: str
  segments: list[TextSegment]

TextSegment:
  text: str               # Markdown-formatted, complete sentences
  heading_path: str       # e.g. "## Methods / ### Entity Extraction"
  char_offset: int
  metadata: dict          # parser-specific: {language, function_name, lineno} etc.
```

---

## Module Layout

```
services/graphknows/graphknows/
  ingestion/
    parsers/
      __init__.py
      base.py           ← BaseParser, ParsedFile, TextSegment
      text.py           ← PlainTextParser (txt, md, raw str)
      code.py           ← CodeParser (py, js, ts via AST)
      docx.py           ← DocxParser (python-docx)
      registry.py       ← get_parser(path_or_hint) dispatch
    pipeline.py         ← DELETED (replaced by memory/ingestor.py)
  memory/
    __init__.py
    ingestor.py         ← MemoryIngestor (text → STM evidence graph)
    dialog.py           ← DialogWriter (turns → STM interaction graph)
    flush.py            ← SessionFlusher (STM → LTM, entity merge, topics)
  retrieval/
    recall_stm.py       ← recall_short_term (Memgraph + stm_* tables)
    recall_ltm.py       ← recall_long_term (Neo4j + ltm_* tables)
    retriever.py        ← HybridRetriever updated with scope routing
  cli/
    main.py             ← graphknows-cli root (updated memory + retrieve groups)
    memory.py           ← 'memory' subcommand group (ingest, dialog, flush, status)
    retrieve.py         ← 'retrieve' subcommand group (--short-term / --long-term)
    doctor.py           ← connectivity health checks
    forget.py           ← session forget + LTM prune
```

---

## CLI Commands

```
# Memory layer (NEW — primary)
graphknows-cli memory ingest --text "..." --session-id <id> [--title "..."] [--json]
graphknows-cli memory ingest --file path/to/file.py --session-id <id> [--json]
graphknows-cli memory dialog start --session-id <id> --dialog-id <id>
graphknows-cli memory dialog turn --dialog-id <id> --role agent --text "..."
graphknows-cli memory flush --session-id <id> [--no-topics] [--no-clear] [--run-gds] [--confirm]
graphknows-cli memory status --session-id <id>   # node/edge/chunk counts in STM

# Retrieve (updated with scope)
graphknows-cli retrieve query "..." --short-term --session-id <id>
graphknows-cli retrieve query "..." --long-term [--session-id <id>]
graphknows-cli retrieve explain "..." --short-term --session-id <id>  # show RRF per-source

# Doctor
graphknows-cli doctor

# Forget
graphknows-cli forget session --session-id <id> [--confirm]
graphknows-cli forget session --all --confirm
graphknows-cli forget isolated
graphknows-cli forget ttl [--days 90]
```

---

## Flush Process (STM → LTM)

1. Read all SESSION subgraph from Memgraph (entities, chunks, dialog turns)
2. Entity resolution: group by canonical name, pick highest-confidence label
3. MERGE into Neo4j: SESSION → FILE → CHUNK → ENTITY, SESSION → DIALOG → TURN
4. INSERT stm_chunks → ltm_chunks, stm_turns → ltm_turns (ON CONFLICT DO NOTHING)
5. If `--topics` (default): Leiden community detection → LLM summarization per community → ltm_topics
6. If `--run-gds`: trigger feature engineering (PageRank, betweenness)
7. If `--no-clear` NOT set: DETACH DELETE from Memgraph + DELETE from stm_* tables
8. Return FlushResult with counts

---

## Deleted Legacy Files

- `document_ingestor.py` — replaced by `memory/ingestor.py`
- `ingestion/pipeline.py` — replaced by `memory/ingestor.py`
- `migrations/001_ltm8_schema.py` — replaced by `migrations/002_session_schema.py` + `003_stm_ltm_split.py`

---

## Recall API

```python
# Short-term: Memgraph entity graph + stm_chunks/stm_turns ANN
from graphknows.retrieval.recall_stm import recall_short_term
results = recall_short_term(mg_driver, pg_pool, query="...", session_id="...", top_k=5)

# Long-term: Neo4j + ltm_chunks/ltm_turns/ltm_topics ANN + RRF
from graphknows.retrieval.recall_ltm import recall_long_term
results = recall_long_term(neo4j_driver, pg_pool, query="...", top_k=5)

# RecallResult.scope = "short_term" | "long_term"
```

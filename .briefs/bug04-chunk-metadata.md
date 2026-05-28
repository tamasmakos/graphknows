# BUG-04: Populate chunk metadata JSONB — atom_type + memory_class + entity_names

## Working directory
/root/graphknows

## Context

The MCP `memory_ingest` tool calls `PlainTextParser.parse()` which outputs segments with
`seg.metadata = {}` always. This means `stm_chunks.metadata` is `{}` for every chunk.

Two problems result:
1. `memory_recall` returns `entities: []` for ANN hits post-flush — entity names are extracted
   and stored in Memgraph but NOT persisted in pgvector `stm_chunks.metadata`. After flush,
   Memgraph STM is cleared. Entity names are then permanently lost from recall hits.
2. Score shaping in `_shape_score()` always defaults to `raw_evidence` (×0.6 de-boost)
   because `memory_class` is never set in metadata.

## Files to read FIRST

1. `graphknows/ingestion/parsers/text.py` — PlainTextParser, understand what `seg.metadata` contains
2. `graphknows/ingestion/parsers/base.py` — TextSegment dataclass, understand `metadata` field
3. `graphknows/memory/ingestor.py` — `_ingest_parsed()`, where `write_stm_chunk` is called
4. `graphknows/retrieval/recall_stm.py` — `_run_stm_ann()`, see where `entity_names` stays empty
5. `graphknows/retrieval/stores/pgvector.py` — `write_stm_chunk()` and `search_stm_chunks()` — confirm metadata is written and returned

## Required Changes

### Change 1: `graphknows/memory/ingestor.py` — store entity names in metadata

In `_ingest_parsed()`, after extracting `entities`, include the entity names in the chunk
metadata written to pgvector:

```python
# BEFORE (current):
_run(
    self._pg.write_stm_chunk({
        "id": chunk_id,
        "session_id": session_id,
        "file_id": parsed.source_id,
        "text": clean_text,
        "heading_path": seg.heading_path,
        "embedding": embedding,
        "metadata": {**seg.metadata, **metadata},
    })
)

# AFTER:
entity_names_list = [e["name"] for e in entities]
_run(
    self._pg.write_stm_chunk({
        "id": chunk_id,
        "session_id": session_id,
        "file_id": parsed.source_id,
        "text": clean_text,
        "heading_path": seg.heading_path,
        "embedding": embedding,
        "metadata": {
            **seg.metadata,
            **metadata,
            "atom_type": seg.metadata.get("atom_type", "text"),
            "memory_class": seg.metadata.get("memory_class", "raw_evidence"),
            "entity_names": entity_names_list,
        },
    })
)
```

### Change 2: `graphknows/retrieval/recall_stm.py` — populate entity_names from metadata

In `_run_stm_ann()`, when building `RecallResult` from pgvector ANN hits, read `entity_names`
from chunk metadata:

```python
# BEFORE:
results[cid] = (
    rrf,
    RecallResult(
        text=chunk["text"],
        source_type="chunk",
        scope="short_term",
        score=rrf,
        cosine_score=chunk.get("cosine_score", 0.0),
        heading_path=chunk.get("heading_path", ""),
        provenance={...},
        metadata=chunk.get("metadata", {}),
    ),
)

# AFTER: add entity_names= from metadata
results[cid] = (
    rrf,
    RecallResult(
        text=chunk["text"],
        source_type="chunk",
        scope="short_term",
        score=rrf,
        cosine_score=chunk.get("cosine_score", 0.0),
        entity_names=chunk.get("metadata", {}).get("entity_names", []),
        heading_path=chunk.get("heading_path", ""),
        provenance={...},
        metadata=chunk.get("metadata", {}),
    ),
)
```

## Tests to write

File: `tests/test_ingestor_metadata.py` (new file)

```python
"""Tests for chunk metadata persistence in stm_chunks."""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch, call
from graphknows.memory.ingestor import MemoryIngestor


def test_entity_names_stored_in_chunk_metadata():
    """entity_names extracted by GLiNER2 must be persisted in stm_chunks.metadata."""
    mg = MagicMock()
    pg = MagicMock()
    extractor = MagicMock()
    extractor.extract.return_value = [
        {"name": "GraphKnows", "type": "ORGANIZATION", "score": 0.9},
        {"name": "Neo4j", "type": "PRODUCT", "score": 0.8},
    ]
    embed_fn = MagicMock(return_value=[0.1] * 1536)

    # Patch _run to execute synchronously
    with patch("graphknows.memory.ingestor._run", side_effect=lambda coro: None):
        ingestor = MemoryIngestor(mg, pg, extractor, embed_fn=embed_fn)
        ingestor.ingest_text("GraphKnows uses Neo4j for storage.", session_id="test-meta")

    # Find the write_stm_chunk call
    write_calls = [c for c in pg.method_calls if "write_stm_chunk" in str(c)]
    assert write_calls, "write_stm_chunk must be called"
    chunk_dict = write_calls[0].args[0]
    meta = chunk_dict.get("metadata", {})
    assert "entity_names" in meta, "entity_names must be in chunk metadata"
    assert "GraphKnows" in meta["entity_names"]
    assert "Neo4j" in meta["entity_names"]
    assert meta.get("atom_type") == "text"
    assert meta.get("memory_class") == "raw_evidence"


def test_entity_names_from_metadata_in_recall_result():
    """RecallResult.entity_names must be populated from stm_chunks.metadata."""
    from graphknows.retrieval.recall_stm import _run_stm_ann, RecallResult

    pg = MagicMock()
    chunk = {
        "id": "chunk-001",
        "text": "GraphKnows uses Neo4j.",
        "heading_path": "",
        "file_id": "file-001",
        "cosine_score": 0.85,
        "metadata": {
            "entity_names": ["GraphKnows", "Neo4j"],
            "memory_class": "fact",
        },
    }
    with patch("graphknows.memory.ingestor._run", return_value=[chunk]):
        results = {}
        _run_stm_ann(pg, "test-meta", [0.1]*1536, 5, results)

    assert "chunk-001" in results
    _, recall = results["chunk-001"]
    assert recall.entity_names == ["GraphKnows", "Neo4j"]
```

## Coding Standards

- Read `.hermes/instructions/python-style.instructions.md` before writing any code.
- Type annotations on all new public APIs.
- No bare `except:`.
- 80-char line length.
- Run `ruff format graphknows/ && ruff check --no-fix --config ruff.toml graphknows/` before finishing.
- Run `uv run pytest tests/test_ingestor_metadata.py -xvs` and confirm green.

## Acceptance Criteria

1. `stm_chunks.metadata` contains `entity_names`, `atom_type`, `memory_class` after ingest.
2. `memory_recall` hits include `entities: ["GraphKnows", "Neo4j", ...]` not `[]`.
3. New test file `tests/test_ingestor_metadata.py` passes.
4. `ruff check` passes with no new errors.
5. All 199+ existing tests still pass.

## End-of-task summary (mandatory)

```
Files changed: <list every file touched>
What was modified: <one line per file>
Files intentionally not touched: <list>
Follow-up needed: <list or "none">
```

Do NOT commit. Do NOT push. Do NOT modify any file outside the scope above.

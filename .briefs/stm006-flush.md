# STM-006: SessionFlusher — STM → LTM promotion with entity merge + topic summarization

## Context
Wiki: /root/graphknows/llm-wiki/architecture/memory-layer-v2.md
Project root: /root/graphknows
Read .hermes/instructions/python-style.instructions.md first.
Read .hermes/instructions/tdd-test-driven-development.instructions.md first.
Model: github-copilot/claude-sonnet-4.6

Depends on: STM-001 (pg promote_to_ltm, clear_stm methods), STM-003, STM-005

## Goal
SessionFlusher reads all STM data for a session and promotes it to Neo4j LTM + ltm_* pgvector tables.
Entity merging, topic summarization (Leiden+LLM), and GDS enrichment run here — NEVER at ingestion time.

## Read existing topic_summarizer.py BEFORE writing
The file services/graphknows/graphknows/topic_summarizer.py exists but will be REPLACED.
Read it to understand its TopicSummarizer class. You will reuse its LLM summarization logic
inside flush.py (not as a standalone class anymore — inline or adapted).

## Create services/graphknows/graphknows/memory/flush.py

```python
"""SessionFlusher: promote STM (Memgraph + stm_* pgvector) to LTM (Neo4j + ltm_* pgvector)."""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger("graphknows.memory.flush")


@dataclass
class FlushOptions:
    """Options controlling flush behavior."""
    summarize_topics: bool = True   # Leiden + LLM summarization (slow, only here)
    run_gds: bool = False           # GDS PageRank/betweenness (even slower)
    clear_stm: bool = True          # Delete Memgraph session + stm_* rows after flush
    merge_entities: bool = True     # Canonical entity name resolution before LTM write


@dataclass
class FlushResult:
    """Output of a flush operation."""
    session_id: str
    nodes_written: int = 0
    edges_written: int = 0
    chunks_promoted: int = 0
    turns_promoted: int = 0
    topics_created: int = 0
    merged_entities: int = 0
    elapsed_s: float = 0.0
    errors: list[str] = field(default_factory=list)


class SessionFlusher:
    """Promote a session's STM to LTM and optionally clear STM.

    Flush steps (in order):
    1. Read SESSION subgraph from Memgraph (entities, chunks, dialog turns)
    2. Optional entity resolution: group by canonical name, pick highest-confidence label
    3. MERGE into Neo4j: SESSION → FILE → CHUNK → ENTITY, SESSION → DIALOG → TURN
    4. Copy stm_chunks → ltm_chunks, stm_turns → ltm_turns (ON CONFLICT DO NOTHING)
    5. Optional Leiden community detection → LLM summarization → write ltm_topics
    6. Optional GDS feature engineering (PageRank, betweenness)
    7. Optional STM clear: DETACH DELETE from Memgraph + DELETE from stm_*

    Attributes:
        _mg: MemgraphStore (read STM, optional clear).
        _pg: PgVectorStore (promote + ltm_topics write).
        _neo: Neo4jStore (write LTM graph).
        _openrouter_key: API key for LLM summarization.
        _llm_model: LLM model name for topic summaries.
    """

    def __init__(
        self,
        mg_store: Any,   # MemgraphStore
        pg_store: Any,   # PgVectorStore
        neo_store: Any,  # Neo4jStore
        openrouter_key: str = "",
        llm_model: str = "meta-llama/llama-3.1-8b-instruct",
    ) -> None:
        self._mg = mg_store
        self._pg = pg_store
        self._neo = neo_store
        self._openrouter_key = openrouter_key
        self._llm_model = llm_model

    async def flush(self, session_id: str, options: FlushOptions | None = None) -> FlushResult:
        """Run the full STM→LTM promotion pipeline."""
        opts = options or FlushOptions()
        t0 = time.monotonic()
        result = FlushResult(session_id=session_id)

        # Step 1: Read STM graph from Memgraph
        stm_data = self._read_stm(session_id)

        # Step 2: Entity resolution
        if opts.merge_entities:
            stm_data["entities"] = self._resolve_entities(stm_data.get("entities", []))
            result.merged_entities = stm_data.get("_merge_count", 0)

        # Step 3: Write to Neo4j LTM graph
        graph_counts = await self._write_ltm_graph(session_id, stm_data)
        result.nodes_written = graph_counts["nodes"]
        result.edges_written = graph_counts["edges"]

        # Step 4: Promote pgvector STM → LTM
        chunks_promoted, turns_promoted = await self._pg.promote_to_ltm(session_id)
        result.chunks_promoted = chunks_promoted
        result.turns_promoted = turns_promoted

        # Step 5: Topic summarization
        if opts.summarize_topics:
            try:
                result.topics_created = await self._summarize_topics(session_id)
            except Exception as exc:
                result.errors.append(f"topics: {exc}")
                log.warning("Topic summarization failed: %s", exc)

        # Step 6: GDS enrichment
        if opts.run_gds:
            try:
                await self._run_gds(session_id)
            except Exception as exc:
                result.errors.append(f"gds: {exc}")
                log.warning("GDS enrichment failed: %s", exc)

        # Step 7: Clear STM
        if opts.clear_stm:
            try:
                await self._clear_stm(session_id)
            except Exception as exc:
                result.errors.append(f"clear_stm: {exc}")

        result.elapsed_s = time.monotonic() - t0
        log.info(  # NOSONAR
            "flush session=%s nodes=%d edges=%d chunks=%d turns=%d topics=%d elapsed=%.1fs",
            session_id, result.nodes_written, result.edges_written,
            result.chunks_promoted, result.turns_promoted, result.topics_created, result.elapsed_s,
        )
        return result

    def _read_stm(self, session_id: str) -> dict[str, Any]:
        """Read all STM nodes/edges for this session from Memgraph."""
        # Use MemgraphStore to query:
        # MATCH (s:SESSION {id: $sid})-[:IN_SESSION]->(f:FILE)-[:HAS_CHUNK]->(c:CHUNK)-[:HAS_ENTITY]->(e:ENTITY)
        # MATCH (s)-[:IN_SESSION]->(d:DIALOG)-[:HAS_TURN]->(t:TURN)
        # Return structured dict with lists of files, chunks, entities, dialogs, turns
        ...

    def _resolve_entities(self, entities: list[dict]) -> list[dict]:
        """Canonical entity name resolution: group by lowercase name, keep highest confidence."""
        from collections import defaultdict
        groups: dict[str, list[dict]] = defaultdict(list)
        for e in entities:
            groups[e["name"].lower()].append(e)
        resolved = []
        merge_count = 0
        for _, group in groups.items():
            best = max(group, key=lambda x: x.get("confidence", 1.0))
            if len(group) > 1:
                merge_count += len(group) - 1
            resolved.append(best)
        return resolved

    async def _write_ltm_graph(self, session_id: str, stm_data: dict) -> dict[str, int]:
        """MERGE all STM nodes/edges into Neo4j LTM with MERGE semantics."""
        ...

    async def _summarize_topics(self, session_id: str) -> int:
        """Run Louvain community detection on ENTITY nodes, LLM summarize each community."""
        # 1. Project entity graph in GDS (or run Louvain directly via Cypher)
        # 2. For each community: collect top entities, build prompt, call OpenRouter LLM
        # 3. Embed summary text
        # 4. Write to ltm_topics via pg.write_ltm_topic()
        # Return topics_created count
        ...

    async def _run_gds(self, session_id: str) -> None:
        """Run GDS PageRank + betweenness on entity graph."""
        # Use gds.louvain.write + gds.pageRank.write (GDS 2026.04 confirmed working)
        # See Louvain YIELD fields: communityCount, modularity, ranLevels
        ...

    async def _clear_stm(self, session_id: str) -> None:
        """Delete all STM data for session from Memgraph and stm_* pgvector tables."""
        # Memgraph: MATCH (n {session_id: $sid}) DETACH DELETE n
        # pgvector: await pg.clear_stm(session_id)
        ...
```

## IMPORTANT: Leiden is broken in GDS 2026.04
Use `gds.louvain.write` for community detection, NOT `gds.leiden.write`.
Valid YIELD fields for GDS 2026.04 Louvain:
communityCount, modularity, modularities, ranLevels, communityDistribution,
preProcessingMillis, computeMillis, postProcessingMillis, writeMillis, nodePropertiesWritten

## Delete services/graphknows/graphknows/topic_summarizer.py
This file is replaced by the _summarize_topics() method above.
Run `git rm services/graphknows/graphknows/topic_summarizer.py`

## Write tests/test_flush.py

Test cases (all unit, MagicMock for mg/pg/neo stores):
1. flush() with default options calls _read_stm, _resolve_entities, _write_ltm_graph, promote_to_ltm
2. flush(options=FlushOptions(summarize_topics=False)) does NOT call _summarize_topics
3. flush(options=FlushOptions(clear_stm=False)) does NOT call _clear_stm
4. flush(options=FlushOptions(merge_entities=False)) does NOT call _resolve_entities
5. _resolve_entities() merges entities with same lowercase name, keeps highest confidence
6. _resolve_entities() preserves entities with unique names unchanged
7. topic summarization failure adds to result.errors, does NOT raise
8. clear_stm failure adds to result.errors, does NOT raise
9. FlushResult has correct counts from mock returns
10. flush() is idempotent: calling twice with clear_stm=False does not duplicate LTM data
    (test by verifying MERGE semantics in _write_ltm_graph mock calls)

## Acceptance criteria
- All 10 tests pass
- ruff + mypy clean
- check_imports.py clean
- topic_summarizer.py deleted from git

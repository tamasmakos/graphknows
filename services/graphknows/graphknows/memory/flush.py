"""SessionFlusher: promote STM (Memgraph + stm_* pgvector) to LTM (Neo4j + ltm_* pgvector)."""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger("graphknows.memory.flush")


@dataclass
class FlushOptions:
    """Options controlling flush behavior."""

    summarize_topics: bool = True
    run_gds: bool = False
    clear_stm: bool = True
    merge_entities: bool = True


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
    1. Read SESSION subgraph from Memgraph.
    2. Optional entity resolution: canonical name, highest confidence.
    3. MERGE into Neo4j: SESSION → FILE → CHUNK → ENTITY, SESSION → DIALOG → TURN.
    4. Copy stm_chunks → ltm_chunks, stm_turns → ltm_turns.
    5. Optional Louvain community detection → LLM summarization → ltm_topics.
    6. Optional GDS PageRank + betweenness.
    7. Optional STM clear.

    Attributes:
        _mg: MemgraphStore (read STM, optional clear).
        _pg: PgVectorStore (promote + ltm_topics write).
        _neo: Neo4jStore or compatible (write LTM graph).
        _openrouter_key: API key for LLM summarization.
        _llm_model: LLM model name for topic summaries.
    """

    def __init__(
        self,
        mg_store: Any,
        pg_store: Any,
        neo_store: Any,
        openrouter_key: str = "",
        llm_model: str = "meta-llama/llama-3.1-8b-instruct",
    ) -> None:
        self._mg = mg_store
        self._pg = pg_store
        self._neo = neo_store
        self._openrouter_key = openrouter_key
        self._llm_model = llm_model

    async def flush(self, session_id: str, options: FlushOptions | None = None) -> FlushResult:
        """Run the full STM→LTM promotion pipeline.

        Args:
            session_id: Session to flush.
            options: Flush configuration (defaults to FlushOptions()).

        Returns:
            FlushResult with counts and any non-fatal errors.
        """
        opts = options or FlushOptions()
        t0 = time.monotonic()
        result = FlushResult(session_id=session_id)

        # Ensure pgvector is connected
        if hasattr(self._pg, "connect") and not getattr(self._pg, "_pool", None):
            await self._pg.connect()

        stm_data = await self._read_stm(session_id)

        if opts.merge_entities:
            stm_data["entities"], result.merged_entities = self._resolve_entities(
                stm_data.get("entities", [])
            )

        graph_counts = await self._write_ltm_graph(session_id, stm_data)
        result.nodes_written = graph_counts["nodes"]
        result.edges_written = graph_counts["edges"]

        try:
            chunks_promoted, turns_promoted = await self._pg.promote_to_ltm(session_id)
            result.chunks_promoted = chunks_promoted
            result.turns_promoted = turns_promoted
        except Exception as exc:
            result.errors.append(f"promote_to_ltm: {exc}")

        if opts.summarize_topics:
            try:
                result.topics_created = await self._summarize_topics(session_id)
            except Exception as exc:
                result.errors.append(f"topics: {exc}")
                log.warning("Topic summarization failed: %s", exc)

        if opts.run_gds:
            try:
                await self._run_gds(session_id)
            except Exception as exc:
                result.errors.append(f"gds: {exc}")
                log.warning("GDS enrichment failed: %s", exc)

        if opts.clear_stm:
            try:
                await self._clear_stm(session_id)
            except Exception as exc:
                result.errors.append(f"clear_stm: {exc}")

        result.elapsed_s = time.monotonic() - t0
        log.info(
            "flush session=%s nodes=%d edges=%d chunks=%d turns=%d topics=%d elapsed=%.1fs",
            session_id,
            result.nodes_written,
            result.edges_written,
            result.chunks_promoted,
            result.turns_promoted,
            result.topics_created,
            result.elapsed_s,
        )
        return result

    async def _read_stm(self, session_id: str) -> dict[str, Any]:
        """Read all STM nodes/edges for this session from Memgraph."""
        try:
            return await self._mg.read_session_subgraph(session_id)
        except Exception as exc:
            log.warning("read_stm %s: %s", session_id, exc)
            return {"entities": [], "files": [], "chunks": [], "dialogs": [], "turns": []}

    def _resolve_entities(self, entities: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
        """Canonical entity name resolution: group by lowercase name, keep highest confidence.

        Returns:
            Tuple of (resolved_entities, merge_count).
        """
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for entity in entities:
            groups[entity["name"].lower()].append(entity)

        resolved: list[dict[str, Any]] = []
        merge_count = 0
        for group in groups.values():
            best = max(group, key=lambda x: x.get("confidence", 1.0))
            if len(group) > 1:
                merge_count += len(group) - 1
            resolved.append(best)

        return resolved, merge_count

    async def _write_ltm_graph(self, session_id: str, stm_data: dict[str, Any]) -> dict[str, int]:
        """MERGE all STM nodes/edges into Neo4j LTM with MERGE semantics."""
        nodes = 0
        edges = 0
        try:
            result = await self._neo.write_session_subgraph(session_id, stm_data)
            nodes = result.get("nodes", 0)
            edges = result.get("edges", 0)
        except Exception as exc:
            log.warning("write_ltm_graph %s: %s", session_id, exc)
        return {"nodes": nodes, "edges": edges}

    async def _summarize_topics(self, session_id: str) -> int:
        """Run Louvain community detection on ENTITY nodes, LLM summarize each community.

        Uses gds.louvain.write (NOT leiden — broken in GDS 2026.04).
        """
        count = await self._neo.summarize_topics(
            session_id=session_id,
            openrouter_key=self._openrouter_key,
            llm_model=self._llm_model,
            pg_store=self._pg,
        )
        return count

    async def _run_gds(self, session_id: str) -> None:
        """Run GDS PageRank + betweenness on entity graph."""
        await self._neo.run_gds_enrichment(session_id)

    async def _clear_stm(self, session_id: str) -> None:
        """Delete all STM data for session from Memgraph and stm_* pgvector tables."""
        await self._mg.delete_session(session_id)
        await self._pg.clear_stm(session_id)

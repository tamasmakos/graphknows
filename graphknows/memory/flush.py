"""SessionFlusher: promote STM (Memgraph + stm_* pgvector) to LTM (Neo4j + ltm_* pgvector)."""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from graphknows.memory.ltm import LTMEnrichmentService, LTMWriter

log = logging.getLogger("graphknows.memory.flush")


# COMPAT: FlushOptions retained for callers (cli/mcp). Canonical DTO: PromotionRequest.
@dataclass
class FlushOptions:
    """Options controlling flush behavior."""

    summarize_topics: bool = True  # Deprecated: topic summarization always runs.
    run_gds: bool = True
    clear_stm: bool = True
    merge_entities: bool = True


# COMPAT: FlushResult retained for backward compatibility. Canonical DTO: PromotionRecord.
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
    # GDS community-detection stats (populated when run_gds=True)
    community_count: int = 0
    modularity: float = 0.0
    ran_iterations: int = 0
    elapsed_s: float = 0.0
    errors: list[str] = field(default_factory=list)


class SessionFlusher:
    """Promote a session's STM to LTM and optionally clear STM.

    Flush steps (in order):
    1. Read SESSION subgraph from Memgraph.
    2. Optional entity resolution: canonical name, highest confidence.
    3. MERGE into Neo4j via LTMWriter: SESSION → FILE → CHUNK → ENTITY, SESSION → DIALOG → TURN.
    4. Copy stm_chunks → ltm_chunks, stm_turns → ltm_turns.
    5. Optional Louvain community detection → LLM summarization → ltm_topics (via LTMEnrichmentService).
    6. Optional GDS PageRank + betweenness (via LTMEnrichmentService).
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
        relation_extraction_mode: str = "gliner2",
    ) -> None:
        self._mg = mg_store
        self._pg = pg_store
        self._neo = neo_store
        self._openrouter_key = openrouter_key
        self._llm_model = llm_model
        self._relation_extraction_mode = relation_extraction_mode

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

        graph_counts = await self._write_ltm_graph(session_id, stm_data, errors=result.errors)
        result.nodes_written = graph_counts["nodes"]
        result.edges_written = graph_counts["edges"]

        try:
            chunks_promoted, turns_promoted = await self._pg.promote_to_ltm(session_id)
            result.chunks_promoted = chunks_promoted
            result.turns_promoted = turns_promoted
        except Exception as exc:
            result.errors.append(f"promote_to_ltm: {exc}")

        try:
            result.topics_created = await self._summarize_topics(session_id)
        except Exception as exc:
            result.errors.append(f"topics: {exc}")
            log.warning("Topic summarization failed: %s", exc)

        if opts.run_gds:
            try:
                gds_result = await self._run_gds()
                result.community_count = gds_result.community_count
                result.modularity = gds_result.modularity
                result.ran_iterations = gds_result.ran_iterations
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
            result: dict[str, Any] = await self._mg.read_session_subgraph(session_id)
        except Exception as exc:
            log.warning("read_stm %s: %s", session_id, exc)
            return {"entities": [], "files": [], "chunks": [], "dialogs": [], "turns": []}
        else:
            return result

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

    async def _write_ltm_graph(
        self,
        session_id: str,
        stm_data: dict[str, Any],
        errors: list[str] | None = None,
    ) -> dict[str, int]:
        """MERGE all STM nodes/edges into Neo4j LTM via LTMWriter."""
        return await LTMWriter.write(
            session_id=session_id,
            stm_data=stm_data,
            neo_store=self._neo,
            errors=errors,
            openrouter_key=self._openrouter_key,
            relation_extraction_mode=self._relation_extraction_mode,
        )

    async def _summarize_topics(self, session_id: str) -> int:
        """Run topic summarization via LTMEnrichmentService."""
        svc = LTMEnrichmentService()
        return await svc.run_topic_summarization(
            session_id=session_id,
            driver=self._neo,
            pg_store=self._pg,
            llm_key=self._openrouter_key,
            llm_model=self._llm_model,
        )

    async def _run_gds(self) -> Any:
        """Run GDS enrichment via LTMEnrichmentService. Returns GDSResult."""
        from graphknows.memory.models.gds import GDSResult

        svc = LTMEnrichmentService()
        driver = getattr(self._neo, "_driver", None)
        if driver is None:
            log.warning("GDS enrichment skipped: neo_store has no _driver attribute")
            return GDSResult()
        return await svc.run_gds_async(driver)

    async def _clear_stm(self, session_id: str) -> None:
        """Delete all STM data for session from Memgraph and stm_* pgvector tables."""
        await self._mg.delete_session(session_id)
        await self._pg.clear_stm(session_id)

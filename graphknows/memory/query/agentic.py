"""AgenticRetriever — DSPy ReAct retriever extending BaseGraphRetriever."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

import dspy

from graphknows.config.llm import normalize_openrouter_model
from graphknows.memory.query.base import BaseGraphRetriever

if TYPE_CHECKING:
    from graphknows.config import GraphKnowsSettings
    from graphknows.memory.query.context import RetrievalContext

logger = logging.getLogger(__name__)


class AgenticRetriever(BaseGraphRetriever):
    """LLM-guided retriever over graphknows memory using DSPy ReAct.

    Three synchronous tool closures wrap the three stores. Sync tools call
    async store methods via the shared background loop from _async_bridge.
    The main ``retrieve()`` method is fully async.

    Inherits from BaseGraphRetriever so it is interchangeable with
    HybridRetriever at the factory layer.
    """

    def __init__(
        self,
        settings: "GraphKnowsSettings",
        pg: Any,
        neo: Any,
        memgraph: Any,
    ) -> None:
        self._settings = settings
        self._pg = pg
        self._neo = neo
        self._memgraph = memgraph
        self._react: Any = None
        self._lm: Any = None
        self._pg_bridge: Any = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_pg_bridge(self) -> Any:
        """Return a PgVectorStore connected on _BG_LOOP for sync DSPy tools."""
        if self._pg_bridge is None:
            from graphknows.config.builders.stores import build_pgvector_store
            from graphknows.memory.stm._async_bridge import _run

            self._pg_bridge = build_pgvector_store(self._settings)
            _run(self._pg_bridge.connect())
        return self._pg_bridge

    def _get_react(self, *, session_id: str = "") -> Any:
        if self._react is not None:
            return self._react

        from graphknows.memory.stm._async_bridge import _run as bg_run

        pg_bridge = self._get_pg_bridge()
        neo = self._neo
        memgraph = self._memgraph
        embed_model = self._settings.embed_model
        # Chunks collected by tool calls during ReAct execution.
        # retrieve() reads this list after ReAct finishes.
        self._collected_chunks: list[dict] = []
        collected = self._collected_chunks

        # ---- Tool 1: pgvector ANN chunk search -------------------------
        def search_vectors(query: str, scope: str = "ltm", top_k: int = 10) -> list[dict]:
            """Search pgvector for relevant text chunks.

            Args:
                query: Natural-language search query.
                scope: 'stm' for short-term chunks, 'ltm' for all chunks.
                top_k: Max results to return (default 10).

            Returns:
                List of dicts with chunk_id, doc_id, text, cosine_score.
            """
            from graphknows.memory.stores.embedder import embed

            emb_array = bg_run(asyncio.to_thread(embed, [query], embed_model))
            emb = emb_array[0].tolist()
            actual_sid = session_id if scope == "stm" else ""
            results = bg_run(pg_bridge.search_chunks(emb, session_id=actual_sid, k=top_k))
            # Do NOT add to collected — these are covered by the explicit fallback
            # vector search in retrieve(). collected is reserved for graph-traversal
            # chunks that pure vector search cannot find.
            return results

        # ---- Tool 2: STM Memgraph entity lookup -----------------------
        def search_stm_graph(query: str) -> list[dict]:
            """Search the short-term memory graph for entities matching the query.

            Args:
                query: Entity name or concept to look up in the STM graph.

            Returns:
                List of entity dicts whose name contains the query string.
            """
            result = bg_run(memgraph.get_session_graph(session_id=session_id))
            nodes = result.get("nodes", [])
            q_lower = query.lower()
            return [n for n in nodes if q_lower in str(n.get("name", "")).lower()]

        # ---- Tool 3: LTM Neo4j entity + neighbor search ---------------
        def search_ltm_graph(query: str, top_k: int = 10) -> list[dict]:
            """Search long-term memory for entity-linked chunks using ANN entity search
            followed by Neo4j graph traversal.

            Args:
                query: Natural-language query to find relevant entities.
                top_k: Max chunk results to return (default 10).

            Returns:
                List of chunk dicts (chunk_id, doc_id, text, entity) for the found entities.
            """
            from graphknows.memory.stores.embedder import embed

            emb_array = bg_run(asyncio.to_thread(embed, [query], embed_model))
            emb = emb_array[0].tolist()
            entity_rows = bg_run(pg_bridge.search_entities(emb, k=min(top_k, 10)))
            chunks: list[dict] = []
            seen: set[str] = set()
            for ent in entity_rows:
                entity_name = ent.get("name", "")
                if not entity_name:
                    continue
                entity_chunks = bg_run(neo.get_chunks_for_entity(entity_name, top_k=5))
                for c in entity_chunks:
                    cid = c.get("chunk_id", "")
                    if cid and cid not in seen:
                        seen.add(cid)
                        row = {**c, "entity": entity_name, "cosine_score": ent.get("score", 0.0)}
                        chunks.append(row)
                        collected.append(row)
                if len(chunks) >= top_k:
                    break
            return chunks[:top_k]

        # ---- DSPy signature and ReAct module --------------------------
        class RetrievalSignature(dspy.Signature):
            """Retrieve the most relevant knowledge-graph context for a query.

            Use the available tools to search vectors and graphs.
            Synthesize findings to identify the most relevant information.
            """

            query: str = dspy.InputField(
                desc="Natural-language question to retrieve context for."
            )
            session_id: str = dspy.InputField(
                desc="Session namespace for scoping short-term memory searches."
            )
            answer: str = dspy.OutputField(
                desc="2-3 sentence summary of the most relevant retrieved information."
            )

        react_model = self._settings.extraction_model or self._settings.llm_model
        self._lm = dspy.LM(
            model=normalize_openrouter_model(react_model),
            api_key=self._settings.openrouter_api_key,
            api_base="https://openrouter.ai/api/v1",
            temperature=0.0,
            max_tokens=2048,
            num_retries=1,
        )
        with dspy.settings.context(lm=self._lm):
            self._react = dspy.ReAct(
                RetrievalSignature,
                tools=[search_vectors, search_stm_graph, search_ltm_graph],
            )
        return self._react

    # ------------------------------------------------------------------
    # BaseGraphRetriever implementation
    # ------------------------------------------------------------------

    async def retrieve(
        self,
        query: str,
        session_id: str,
        top_k: int = 10,
        scope: str = "both",
    ) -> "RetrievalContext":
        from graphknows.memory.query.context import ChunkMatch, RetrievalContext
        from graphknows.memory.stores.embedder import embed

        # Reset ReAct per call so closures bind to a fresh collected list.
        self._react = None
        react = self._get_react(session_id=session_id)
        lm = self._lm

        def _run_react() -> Any:
            with dspy.settings.context(lm=lm):
                return react(query=query, session_id=session_id)

        react_failed = False
        try:
            await asyncio.to_thread(_run_react)
        except Exception as exc:
            logger.warning("DSPy ReAct retrieval failed: %s", exc)
            react_failed = True

        # Dedup chunks gathered by ReAct tool calls, keeping highest cosine_score.
        react_chunks: dict[str, dict] = {}
        for r in self._collected_chunks:
            cid = r["chunk_id"]
            if cid not in react_chunks or r.get("cosine_score", 0) > react_chunks[cid].get("cosine_score", 0):
                react_chunks[cid] = r

        # Fallback vector search fills gaps when ReAct found few chunks.
        emb_array = await asyncio.to_thread(embed, [query], self._settings.embed_model)
        emb = emb_array[0].tolist()
        fallback_sid = session_id if scope == "stm" else ("" if scope == "ltm" else session_id)
        fallback_chunks = await self._pg.search_chunks(emb, session_id=fallback_sid, k=top_k)

        # ReAct-found chunks rank first; fallback fills remaining slots.
        react_set = set(react_chunks.keys())
        fallback_only = [r for r in fallback_chunks if r["chunk_id"] not in react_set]
        ranked = (
            sorted(react_chunks.values(), key=lambda r: -r.get("cosine_score", 0.0))
            + sorted(fallback_only, key=lambda r: -r.get("cosine_score", 0.0))
        )[:top_k]

        semantic_matches = [
            ChunkMatch(
                chunk_id=r["chunk_id"],
                doc_id=r.get("doc_id", ""),
                text=r.get("text", ""),
                metadata={"cosine_score": r.get("cosine_score", 0.0), **r.get("metadata", {})},
                entities=[],
            )
            for r in ranked
        ]
        sources = "agentic" if not react_failed else "agentic-fallback"
        fused = [
            {
                "chunk_id": cm.chunk_id,
                "doc_id": cm.doc_id,
                "text": cm.text,
                "metadata": cm.metadata,
                "rrf_score": cm.metadata.get("cosine_score", 0.0),
                "sources": sources,
                "rank": i + 1,
                "entities": [],
            }
            for i, cm in enumerate(semantic_matches)
        ]
        return RetrievalContext(
            query=query,
            session_id=session_id,
            semantic_matches=semantic_matches,
            fused_results=fused,
            search_type="AGENTIC",
        )

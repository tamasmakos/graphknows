"""AgenticRetriever — DSPy ReAct retriever extending BaseGraphRetriever."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

try:
    import dspy
except ImportError:  # pragma: no cover - optional deployment dependency
    dspy = None  # type: ignore[assignment]

from graphknows.memory.query.base import BaseGraphRetriever

if TYPE_CHECKING:
    from graphknows.config import GraphKnowsSettings
    from graphknows.memory.query.context import RetrievalContext

log = logging.getLogger(__name__)

_DEFAULT_MODEL = "moonshotai/kimi-k2"


def _normalize_openrouter_model(model: str | None) -> str:
    """Return *model* with an ``openrouter/`` prefix."""
    if not model:
        return f"openrouter/{_DEFAULT_MODEL}"
    clean = str(model)
    if clean.startswith("openrouter/"):
        clean = clean[len("openrouter/") :]
    return f"openrouter/{clean}"


def _require_dspy() -> Any:
    """Return the DSPy module or raise a clear optional dependency error."""
    if dspy is None:
        raise ImportError("dspy is required for agentic retrieval")
    return dspy


class AgenticRetriever(BaseGraphRetriever):
    """LLM-guided retriever over graphknows memory using DSPy ReAct.

    Three synchronous tool closures wrap the three stores. Sync tools call
    async store methods via the shared background loop from _async_bridge.
    The main ``retrieve()`` method is fully async.

    Inherits from BaseGraphRetriever so it is interchangeable with
    DETRetriever at the factory layer.
    """

    def __init__(
        self,
        settings: GraphKnowsSettings,
        pg: Any,
        neo: Any,
        stm: Any,
    ) -> None:
        self._settings = settings
        self._pg = pg
        self._neo = neo
        self._stm = stm
        self._react: Any = None
        self._lm: Any = None
        self._pg_bridge: Any = None
        self._last_session_id: str | None = None
        self._collected_chunks: list[dict] = []

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_pg_bridge(self) -> Any:
        """Return the vector store for use in sync DSPy tool closures."""
        return self._pg

    def _get_lm(self) -> Any:
        """Return the cached DSPy LM instance, building it once."""
        if self._lm is not None:
            return self._lm
        dspy_mod = _require_dspy()
        react_model = self._settings.extraction_model or self._settings.llm_model
        self._lm = dspy_mod.LM(
            model=_normalize_openrouter_model(react_model),
            api_key=self._settings.openrouter_api_key,
            api_base="https://openrouter.ai/api/v1",
            temperature=self._settings.llm_temperature,
            max_tokens=self._settings.llm_max_tokens,
            num_retries=1,
        )
        return self._lm

    def _get_react(self, *, session_id: str = "") -> Any:
        """Build and cache the DSPy ReAct retriever for one session.

        The LM is built once and reused. The ReAct module (which closes over
        session_id via tool closures) is rebuilt only when the session changes.
        """
        dspy_mod = _require_dspy()
        if self._react is not None and session_id == self._last_session_id:
            return self._react

        from graphknows.memory.stm._async_bridge import _run as bg_run

        pg_bridge = self._get_pg_bridge()
        neo = self._neo
        stm = self._stm
        embed_model = self._settings.embed_model
        # Reuse or initialize the chunks buffer; closures below capture this reference.
        self._collected_chunks.clear()
        collected = self._collected_chunks

        # ---- Tool 1: vector ANN chunk search --------------------------
        def search_vectors(query: str, scope: str = "ltm", top_k: int = 10) -> list[dict]:
            """Search the vector store for relevant text chunks.

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

        # ---- Tool 2: STM entity lookup --------------------------------
        def search_stm_graph(query: str) -> list[dict]:
            """Search the short-term memory graph for entities matching the query.

            Args:
                query: Entity name or concept to look up in the STM graph.

            Returns:
                List of entity dicts whose name contains the query string.
            """
            result = bg_run(stm.get_session_graph(session_id=session_id))
            nodes = result.get("nodes", [])
            q_lower = query.lower()
            return [n for n in nodes if q_lower in str(n.get("name", "")).lower()]

        # ---- Tool 3: LTM entity + neighbor search ---------------------
        def search_ltm_graph(query: str, top_k: int = 10) -> list[dict]:
            """Search long-term memory for entity-linked chunks.

            Uses ANN entity search followed by ArcadeDB graph traversal.

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

        # ---- Tool 4: TOPIC keyword search -----------------------------
        def search_topics(query: str, top_k: int = 5) -> list[dict]:
            """Search topic nodes in long-term memory by keyword match.

            Args:
                query: Keywords to match against topic entity names and summaries.
                top_k: Maximum topics to return (default 5).

            Returns:
                List of dicts with topic_id, label, summary, entities, score.
            """
            cypher = """
            UNWIND $words AS word
            MATCH (e:ENTITY)
            WHERE toLower(e.name) CONTAINS word
            WITH collect(DISTINCT e.name) AS matched_names
            MATCH (t:TOPIC)
            OPTIONAL MATCH (e2)-[:IN_TOPIC]->(t)
            WHERE e2.name IN matched_names
            WITH t, collect(DISTINCT e2.name) AS matched_entities,
                 count(DISTINCT e2) AS overlap
            ORDER BY overlap DESC
            LIMIT $top_k
            RETURN t.id AS topic_id, t.label AS label, t.summary AS summary,
                   matched_entities AS entities, overlap AS score
            """
            words = [w for w in query.lower().split() if len(w) > 3]
            if not words:
                return []
            try:
                return bg_run(neo._qry(cypher, words=words, top_k=top_k))
            except Exception as exc:
                log.warning("search_topics failed: %s", exc)
                return []

        # ---- DSPy signature and ReAct module --------------------------
        class RetrievalSignature(dspy_mod.Signature):  # type: ignore[misc, name-defined]
            """Retrieve the most relevant knowledge-graph context for a query.

            Use the available tools to search across four retrieval sources:
            semantic vectors, STM entity graph, LTM entity graph, and topic nodes.
            Synthesize findings to identify the most relevant information.
            """

            query: str = dspy_mod.InputField(
                desc="Natural-language question to retrieve context for."
            )
            session_id: str = dspy_mod.InputField(
                desc="Session namespace for scoping short-term memory searches."
            )
            answer: str = dspy_mod.OutputField(
                desc="2-3 sentence summary of the most relevant retrieved information."
            )

        lm = self._get_lm()
        with dspy_mod.settings.context(lm=lm):
            self._react = dspy_mod.ReAct(
                RetrievalSignature,
                tools=[search_vectors, search_stm_graph, search_ltm_graph, search_topics],
            )
        self._last_session_id = session_id
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
    ) -> RetrievalContext:
        """Retrieve relevant context with DSPy ReAct and vector fallback."""
        from graphknows.memory.query.context import ChunkMatch, RetrievalContext
        from graphknows.memory.stores.embedder import embed

        # _get_react() reuses the existing ReAct module for same-session calls
        # and clears _collected_chunks before each run.
        react = self._get_react(session_id=session_id)
        lm = self._get_lm()

        def _run_react() -> Any:
            with dspy.settings.context(lm=lm):
                return react(query=query, session_id=session_id)

        react_failed = False
        try:
            await asyncio.to_thread(_run_react)
        except Exception as exc:
            log.warning("DSPy ReAct retrieval failed: %s", exc)
            react_failed = True

        # Dedup chunks gathered by ReAct tool calls, keeping highest cosine_score.
        react_chunks: dict[str, dict] = {}
        for r in self._collected_chunks:
            cid = r["chunk_id"]
            if cid not in react_chunks or r.get("cosine_score", 0) > react_chunks[cid].get(
                "cosine_score", 0
            ):
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

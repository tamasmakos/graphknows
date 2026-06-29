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

_DEFAULT_MODEL = ""


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
            # scope='stm'  → STM only (session-scoped)
            # scope='ltm'  → LTM only (no session filter)
            # scope='both' → merge STM + LTM, dedup by chunk_id
            if scope == "stm":
                results = bg_run(pg_bridge.search_chunks(emb, session_id=session_id, k=top_k))
            elif scope == "ltm":
                results = bg_run(pg_bridge.search_chunks(emb, session_id="", k=top_k))
            else:  # 'both'
                stm_res = bg_run(pg_bridge.search_chunks(emb, session_id=session_id, k=top_k))
                ltm_res = bg_run(pg_bridge.search_chunks(emb, session_id="", k=top_k))
                _seen: set[str] = set()
                results = []
                for _r in stm_res + ltm_res:
                    if _r["chunk_id"] not in _seen:
                        _seen.add(_r["chunk_id"])
                        results.append(_r)
                results = results[:top_k]
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
            # Build a name→score map for annotation after the batched fetch.
            name_to_score: dict[str, float] = {}
            for ent in entity_rows:
                name = ent.get("name", "")
                if name:
                    name_to_score[name] = ent.get("score", 0.0)
            if not name_to_score:
                return []
            # One batched query for all entity names (mirrors retriever.py convention:
            # dynamic OR conditions — ArcadeDB openCypher does not support list IN).
            safe_names = [n.replace("'", "") for n in name_to_score]
            where_clause = " OR ".join(f"toLower(e.name) = toLower('{n}')" for n in safe_names)
            per_entity_k = min(5, max(1, top_k))
            cypher = (
                f"MATCH (ch:CHUNK)-[:MENTIONS]->(e:ENTITY) "
                f"WHERE {where_clause} "
                f"OPTIONAL MATCH (f:FILE)-[:HAS_CHUNK]->(ch) "
                f"RETURN ch.id AS chunk_id, coalesce(ch.file_id, f.id, '') AS doc_id, "
                f"ch.text AS text, e.name AS entity_name "
                f"LIMIT $k"
            )
            try:
                raw_rows = bg_run(neo._qry(cypher, k=per_entity_k * len(safe_names)))
            except Exception as exc:
                log.warning("search_ltm_graph batched query failed: %s", exc)
                return []
            chunks: list[dict] = []
            seen: set[str] = set()
            for r in raw_rows:
                cid = r.get("chunk_id", "")
                if not cid or cid in seen:
                    continue
                seen.add(cid)
                ename = r.get("entity_name", "")
                score = name_to_score.get(ename, 0.0)
                row = {
                    "chunk_id": cid,
                    "doc_id": r.get("doc_id", ""),
                    "text": r.get("text", ""),
                    "entity": ename,
                    "cosine_score": score,
                }
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
                rows = bg_run(neo._qry(cypher, words=words, top_k=top_k))
            except Exception as exc:
                log.warning("search_topics failed: %s", exc)
                return []
            # Push the matched topic summary AND its member-entity chunks into the
            # collected buffer so they reach the answer context — a topic summary
            # is pre-aggregated multi-entity content for synthesis/multi-hop.
            for t in rows:
                tid = t.get("topic_id") or ""
                summ = (t.get("summary") or "").strip()
                if summ and tid:
                    collected.append(
                        {
                            "chunk_id": f"topic:{tid}",
                            "doc_id": "",
                            "text": f"[Topic: {t.get('label', '')}] {summ}",
                            "cosine_score": 0.5,
                        }
                    )
            # Collect all member entity names across all matched topics, then fetch
            # their chunks in ONE batched query instead of N*M individual calls.
            all_member_names: list[str] = []
            seen_member: set[str] = set()
            for t in rows:
                for ename in (t.get("entities") or [])[:5]:
                    if ename and ename not in seen_member:
                        seen_member.add(ename)
                        all_member_names.append(ename)
            if all_member_names:
                safe_members = [n.replace("'", "") for n in all_member_names]
                where_members = " OR ".join(
                    f"toLower(e.name) = toLower('{n}')" for n in safe_members
                )
                member_cypher = (
                    f"MATCH (ch:CHUNK)-[:MENTIONS]->(e:ENTITY) "
                    f"WHERE {where_members} "
                    f"OPTIONAL MATCH (f:FILE)-[:HAS_CHUNK]->(ch) "
                    f"RETURN ch.id AS chunk_id, coalesce(ch.file_id, f.id, '') AS doc_id, "
                    f"ch.text AS text "
                    f"LIMIT $k"
                )
                try:
                    member_rows = bg_run(
                        neo._qry(member_cypher, k=3 * len(safe_members))
                    )
                    seen_cids: set[str] = set()
                    for c in member_rows:
                        cid = c.get("chunk_id", "")
                        if cid and cid not in seen_cids:
                            seen_cids.add(cid)
                            collected.append({**c, "cosine_score": 0.4})
                except Exception as exc:
                    log.debug("topic member traversal skipped: %s", exc)
            return rows

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

    async def _apply_community_weights(
        self, scored: list[tuple[float, dict]]
    ) -> list[tuple[float, dict]]:
        """Scale fused RRF scores by each chunk's community weight.

        GDS (llm_assisted) persists ``CHUNK.community_weight``; chunks in denser,
        more cohesive communities are boosted (bounded 1-3x, matching the
        DETRetriever ``_community_boost``). A single batched lookup keeps it cheap;
        chunks without a weight (or synthetic ``topic:`` entries) are left at 1.0,
        so this never regresses when GDS did not run.
        """
        if not scored:
            return scored
        ids = [r["chunk_id"] for _, r in scored if not str(r["chunk_id"]).startswith("topic:")]
        if not ids:
            return scored
        try:
            rows = await self._neo._qry(
                "MATCH (c:CHUNK) WHERE c.id IN $ids RETURN c.id AS id, c.community_weight AS w",
                ids=ids,
            )
        except Exception as exc:
            log.debug("community-weight lookup skipped: %s", exc)
            return scored
        weights = {r["id"]: r.get("w") for r in rows if r.get("w") is not None}
        if not weights:
            return scored
        boosted: list[tuple[float, dict]] = []
        for score, r in scored:
            w = weights.get(r["chunk_id"])
            boost = max(1.0, min(float(w), 3.0)) if w else 1.0
            boosted.append((score * boost, r))
        return boosted

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
        from graphknows.memory.query._rrf_common import rrf_score
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

        # Query-relevant vector search covering both tiers when scope='both'.
        emb_array = await asyncio.to_thread(embed, [query], self._settings.embed_model)
        emb = emb_array[0].tolist()
        if scope == "stm":
            vector_chunks = await self._pg.search_chunks(emb, session_id=session_id, k=top_k)
        elif scope == "ltm":
            vector_chunks = await self._pg.search_chunks(emb, session_id="", k=top_k)
        else:  # 'both'
            _stm = await self._pg.search_chunks(emb, session_id=session_id, k=top_k)
            _ltm = await self._pg.search_chunks(emb, session_id="", k=top_k)
            _seen_v: set[str] = set()
            vector_chunks = []
            for _r in _stm + _ltm:
                if _r["chunk_id"] not in _seen_v:
                    _seen_v.add(_r["chunk_id"])
                    vector_chunks.append(_r)
            vector_chunks = vector_chunks[:top_k]

        # RRF-fuse the two channels by rank, not raw score: entity-graph cosine
        # scores saturate near 1.0 and would otherwise crowd out the genuinely
        # query-relevant vector hits. Rank-based fusion keeps both competitive
        # (same approach as DETRetriever).
        fused_scores: dict[str, tuple[float, dict]] = {}
        for channel in (
            sorted(vector_chunks, key=lambda r: -r.get("cosine_score", 0.0)),
            sorted(react_chunks.values(), key=lambda r: -r.get("cosine_score", 0.0)),
        ):
            for rank, r in enumerate(channel):
                cid = r["chunk_id"]
                prev = fused_scores.get(cid)
                fused_scores[cid] = (
                    rrf_score(rank) + (prev[0] if prev else 0.0),
                    prev[1] if prev else r,
                )
        # Community-weighted re-rank (llm_assisted only): GDS writes
        # CHUNK.community_weight, so chunks in denser, more cohesive communities
        # outrank incidental mentions. No-op when weights are absent.
        candidates = await self._apply_community_weights(list(fused_scores.values()))
        ranked = sorted(candidates, key=lambda x: -x[0])[:top_k]

        sources = "agentic" if not react_failed else "agentic-fallback"
        semantic_matches: list[ChunkMatch] = []
        fused: list[dict[str, Any]] = []
        for i, (score, r) in enumerate(ranked):
            meta = {"cosine_score": r.get("cosine_score", 0.0), **r.get("metadata", {})}
            semantic_matches.append(
                ChunkMatch(
                    chunk_id=r["chunk_id"],
                    doc_id=r.get("doc_id", ""),
                    text=r.get("text", ""),
                    metadata=meta,
                    entities=[],
                )
            )
            fused.append(
                {
                    "chunk_id": r["chunk_id"],
                    "doc_id": r.get("doc_id", ""),
                    "text": r.get("text", ""),
                    "metadata": meta,
                    "rrf_score": round(score, 6),
                    "sources": sources,
                    "rank": i + 1,
                    "entities": [],
                }
            )
        return RetrievalContext(
            query=query,
            session_id=session_id,
            semantic_matches=semantic_matches,
            fused_results=fused,
            search_type="AGENTIC",
        )

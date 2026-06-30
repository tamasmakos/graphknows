"""AgenticRetriever — DSPy ReAct retriever extending BaseGraphRetriever.

Architecture (WS-4):
- A ``DETRetriever`` instance is constructed from the same stores as the
  ``AgenticRetriever``.  The ReAct tools call ``DETRetriever.retrieve()`` as
  the single per-hop retrieval primitive instead of going through weaker,
  competing vector channels.
- The separate weaker vector channel that previously ran in parallel with
  ReAct has been removed.  All retrieval now flows through DETRetriever.
- Query decomposition (``decompose.py``) optionally splits multi-hop
  questions into atomic sub-questions before the ReAct loop runs.
- Evidence is aggregated across hops with RRF (imported from
  ``_rrf_common.rrf_score``; the file itself is not edited).
- If the agent errors for any reason, the fallback is a plain
  ``DETRetriever.retrieve()`` call — the strong deterministic path, not a
  weaker vector search.
- Reasoning hops are capped at 4 (MAX_REACT_HOPS).
"""

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

# Maximum ReAct reasoning hops per sub-question.
MAX_REACT_HOPS: int = 4


def _normalize_openrouter_model(model: str | None) -> str:
    """Return *model* with an ``openrouter/`` prefix."""
    if not model:
        return f"openrouter/{_DEFAULT_MODEL}"
    clean = str(model)
    if clean.startswith("openrouter/"):
        clean = clean[len("openrouter/"):]
    return f"openrouter/{clean}"


def _require_dspy() -> Any:
    """Return the DSPy module or raise a clear optional dependency error."""
    if dspy is None:
        raise ImportError("dspy is required for agentic retrieval")
    return dspy


class AgenticRetriever(BaseGraphRetriever):
    """LLM-guided retriever over graphknows memory using DSPy ReAct.

    Each ReAct tool calls ``DETRetriever.retrieve()`` as the single per-hop
    retrieval primitive.  Evidence gathered across hops is fused with RRF.
    Falls back to ``DETRetriever.retrieve()`` directly if the agent errors.

    Inherits from BaseGraphRetriever so it is interchangeable with
    DETRetriever at the factory layer.
    """

    def __init__(
        self,
        settings: "GraphKnowsSettings",
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
        # DETRetriever instance — built once, shared across all retrieve() calls.
        self._det: Any = None

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
            timeout=120,  # per-request; without it litellm defaults to 6000s → a hung call wedges the run
        )
        return self._lm

    def _get_det_retriever(self) -> Any:
        """Return the cached DETRetriever instance, building it once from the same stores."""
        if self._det is not None:
            return self._det
        from graphknows.memory.query.retriever import DETRetriever

        self._det = DETRetriever(
            pg=self._pg,
            neo=self._neo,
            mg=self._stm,
            embed_model=self._settings.embed_model,
        )
        return self._det

    def _get_react(self, *, session_id: str = "") -> Any:
        """Build and cache the DSPy ReAct retriever for one session.

        The LM is built once and reused.  The ReAct module (which closes over
        session_id via tool closures) is rebuilt only when the session changes.

        Tools call ``DETRetriever.retrieve()`` as the single retrieval primitive
        so the agent amplifies — not competes with — the deterministic path.
        """
        dspy_mod = _require_dspy()
        if self._react is not None and session_id == self._last_session_id:
            return self._react

        from graphknows.memory.stm._async_bridge import _run as bg_run
        from graphknows.memory.temporal import filter_by_validity_window

        det = self._get_det_retriever()
        stm = self._stm
        # Reuse or initialize the chunks buffer; closures below capture this reference.
        self._collected_chunks.clear()
        collected = self._collected_chunks

        # ---- Tool 1: DET-backed per-hop retrieval ----------------------
        def retrieve(subquery: str, scope: str = "both", k: int = 10) -> list[dict]:
            """Retrieve relevant chunks for a single atomic sub-question.

            Delegates to the deterministic retriever (entity traversal + vector
            ANN + full-text + temporal), which is the strongest available
            retrieval primitive.

            Args:
                subquery: A single, atomic natural-language question.
                scope: ``'stm'`` for short-term only, ``'ltm'`` for long-term
                       only, or ``'both'`` (default).
                k: Maximum results to return (default 10).

            Returns:
                List of chunk dicts with chunk_id, doc_id, text, rrf_score.
            """
            try:
                ctx = bg_run(det.retrieve(subquery, session_id=session_id, top_k=k, scope=scope))
                chunks = ctx.fused_results or []
                for c in chunks:
                    collected.append(c)
                return chunks
            except Exception as exc:
                log.warning("retrieve tool failed for %r: %s", subquery, exc)
                return []

        # ---- Tool 2: entity lookup (STM graph) ------------------------
        def lookup_entity(name: str) -> list[dict]:
            """Look up an entity by name in the short-term memory graph.

            Args:
                name: Entity name or partial name to look up in the STM graph.

            Returns:
                List of entity dicts whose name contains *name* (case-insensitive).
            """
            try:
                result = bg_run(stm.get_session_graph(session_id=session_id))
                nodes = result.get("nodes", [])
                q_lower = name.lower()
                return [n for n in nodes if q_lower in str(n.get("name", "")).lower()]
            except Exception as exc:
                log.warning("lookup_entity failed for %r: %s", name, exc)
                return []

        # ---- Tool 3: temporal window filter ---------------------------
        def temporal_filter(window: str) -> list[dict]:
            """Filter already-retrieved chunks by a bi-temporal validity window.

            Applies ``filter_by_validity_window`` from the temporal module to
            the evidence collected so far in this ReAct session, retaining only
            chunks whose relation edges are valid at the requested date.

            Args:
                window: ISO date or datetime string (e.g. ``'2023-06'``) to use
                        as the ``as_of`` anchor for temporal filtering.

            Returns:
                Filtered list of chunk dicts enriched with ``_temporal_score``
                and ``_is_current``.  Returns all collected chunks unchanged
                when ``window`` is empty or filtering fails.
            """
            if not window or not window.strip():
                return list(collected)
            try:
                filtered = filter_by_validity_window(
                    list(collected), as_of=window.strip(), include_open=True
                )
                return filtered
            except Exception as exc:
                log.warning("temporal_filter failed for window=%r: %s", window, exc)
                return list(collected)

        # ---- DSPy signature and ReAct module --------------------------
        class RetrievalSignature(dspy_mod.Signature):  # type: ignore[misc, name-defined]
            """Retrieve the most relevant knowledge-graph context for a query.

            Use the available tools to search for relevant information:
            - ``retrieve(subquery, scope, k)``: Main retrieval tool — calls the
              deterministic retriever (entity traversal, vector ANN, full-text,
              temporal). Use this for each reasoning hop.
            - ``lookup_entity(name)``: Look up a specific entity by name in the
              short-term memory graph.
            - ``temporal_filter(window)``: Filter collected evidence by a date
              window (ISO string).
            Synthesize findings across hops to produce the final answer context.
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
                tools=[retrieve, lookup_entity, temporal_filter],
                max_iters=MAX_REACT_HOPS,
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
    ) -> "RetrievalContext":
        """Retrieve relevant context with DSPy ReAct backed by DETRetriever.

        Flow:
        1. Decompose the query into atomic sub-questions (with deterministic
           fallback to [query] when the LM is unavailable).
        2. Run a DSPy ReAct loop (≤ MAX_REACT_HOPS hops) whose tools all
           delegate to DETRetriever.retrieve() as the single retrieval primitive.
        3. Aggregate evidence across hops with RRF fusion.
        4. If the agent errors, fall back to a plain DETRetriever.retrieve() call.
        """
        from graphknows.memory.query._rrf_common import rrf_score
        from graphknows.memory.query.context import ChunkMatch, RetrievalContext
        from graphknows.memory.query.decompose import decompose

        det = self._get_det_retriever()

        # Step 1 — Decompose the query into atomic sub-questions.
        lm = self._get_lm()
        try:
            sub_questions = await asyncio.to_thread(decompose, query, lm=lm)
        except Exception as exc:
            log.debug("decompose call failed, using [query]: %s", exc)
            sub_questions = [query]
        if not sub_questions:
            sub_questions = [query]

        # Step 2 — Run the ReAct loop; tools populate self._collected_chunks.
        # _get_react() clears _collected_chunks before each new session build.
        react = self._get_react(session_id=session_id)
        self._collected_chunks.clear()

        react_failed = False
        try:
            def _run_react() -> Any:
                with dspy.settings.context(lm=lm):
                    # Run one ReAct call per sub-question so each hop is
                    # bounded by MAX_REACT_HOPS.  Tools push hits to
                    # _collected_chunks for later aggregation.
                    for sq in sub_questions:
                        try:
                            react(query=sq, session_id=session_id)
                        except Exception as sq_exc:
                            log.debug("ReAct sub-question %r failed: %s", sq, sq_exc)

            await asyncio.to_thread(_run_react)
        except Exception as exc:
            log.warning("DSPy ReAct retrieval failed: %s", exc)
            react_failed = True

        # Step 3 — If agent failed completely, fall back to DETRetriever directly.
        # Graceful degradation chain: ReAct → DETRetriever → pg.search_chunks.
        if react_failed or not self._collected_chunks:
            if react_failed:
                log.info("Falling back to DETRetriever.retrieve() after agent failure.")
            try:
                fallback_ctx = await det.retrieve(query, session_id=session_id, top_k=top_k, scope=scope)
                # Tag fallback results so the caller can distinguish them.
                for r in fallback_ctx.fused_results:
                    r["sources"] = "agentic-fallback"
                return RetrievalContext(
                    query=query,
                    session_id=session_id,
                    semantic_matches=fallback_ctx.semantic_matches,
                    fused_results=fallback_ctx.fused_results,
                    search_type="AGENTIC",
                )
            except Exception as fb_exc:
                # Last-resort fallback: direct pg.search_chunks (simpler but always available).
                log.warning("DETRetriever fallback failed (%s); using pg.search_chunks.", fb_exc)
                try:
                    from graphknows.memory.stores.embedder import embed

                    emb_array = await asyncio.to_thread(embed, [query], self._settings.embed_model)
                    emb = emb_array[0].tolist()
                    if scope == "stm":
                        vector_chunks = await self._pg.search_chunks(emb, session_id=session_id, k=top_k)
                    elif scope == "ltm":
                        vector_chunks = await self._pg.search_chunks(emb, session_id="", k=top_k)
                    else:
                        _stm_c = await self._pg.search_chunks(emb, session_id=session_id, k=top_k)
                        _ltm_c = await self._pg.search_chunks(emb, session_id="", k=top_k)
                        _seen_v: set[str] = set()
                        vector_chunks = []
                        for _r in _stm_c + _ltm_c:
                            if _r["chunk_id"] not in _seen_v:
                                _seen_v.add(_r["chunk_id"])
                                vector_chunks.append(_r)
                        vector_chunks = vector_chunks[:top_k]

                    fused_fb: list[dict] = []
                    for i, r in enumerate(vector_chunks):
                        meta = {"cosine_score": r.get("cosine_score", 0.0), **r.get("metadata", {})}
                        fused_fb.append(
                            {
                                "chunk_id": r["chunk_id"],
                                "doc_id": r.get("doc_id", ""),
                                "text": r.get("text", ""),
                                "metadata": meta,
                                "rrf_score": round(rrf_score(i), 6),
                                "sources": "agentic-fallback",
                                "rank": i + 1,
                                "entities": [],
                            }
                        )
                    return RetrievalContext(
                        query=query,
                        session_id=session_id,
                        fused_results=fused_fb,
                        search_type="AGENTIC",
                    )
                except Exception as last_exc:
                    log.error("All fallbacks failed: %s", last_exc)
                    return RetrievalContext(
                        query=query,
                        session_id=session_id,
                        fused_results=[],
                        search_type="AGENTIC",
                    )

        # Step 4 — Aggregate evidence across all ReAct hops with RRF.
        # Deduplicate chunks from self._collected_chunks, keeping highest rrf_score.
        hop_chunks: dict[str, dict] = {}
        for r in self._collected_chunks:
            cid = r.get("chunk_id", "")
            if not cid:
                continue
            existing = hop_chunks.get(cid)
            if existing is None:
                hop_chunks[cid] = r
            else:
                # Keep the entry with the highest rrf_score (or cosine_score as tiebreak).
                new_score = r.get("rrf_score", r.get("cosine_score", 0.0))
                old_score = existing.get("rrf_score", existing.get("cosine_score", 0.0))
                if new_score > old_score:
                    hop_chunks[cid] = r

        # RRF-fuse by rank (rank-based, not raw score, to keep channels competitive).
        fused_scores: dict[str, tuple[float, dict]] = {}
        sorted_hops = sorted(
            hop_chunks.values(),
            key=lambda r: -(r.get("rrf_score", r.get("cosine_score", 0.0))),
        )
        for rank, r in enumerate(sorted_hops):
            cid = r["chunk_id"]
            prev = fused_scores.get(cid)
            fused_scores[cid] = (
                rrf_score(rank) + (prev[0] if prev else 0.0),
                prev[1] if prev else r,
            )

        # Community-weighted re-rank (llm_assisted only).
        candidates = await self._apply_community_weights(list(fused_scores.values()))
        ranked = sorted(candidates, key=lambda x: -x[0])[:top_k]

        sources = "agentic"
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
                    "entities": r.get("entities", []),
                }
            )
        return RetrievalContext(
            query=query,
            session_id=session_id,
            semantic_matches=semantic_matches,
            fused_results=fused,
            search_type="AGENTIC",
        )

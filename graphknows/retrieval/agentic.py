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

from graphknows.exceptions import MissingExtraError
from graphknows.retrieval.base import BaseGraphRetriever

if TYPE_CHECKING:
    from graphknows.retrieval.models.context import RetrievalContext
    from graphknows.settings import GraphKnowsSettings

log = logging.getLogger(__name__)

# Maximum ReAct reasoning hops per sub-question.
MAX_REACT_HOPS: int = 4


def _require_dspy() -> Any:
    """Return the DSPy module or raise a clear optional dependency error."""
    if dspy is None:
        raise MissingExtraError("Agentic retrieval", "assisted")
    return dspy


def _relevance(r: dict) -> float:
    """A chunk's own deterministic relevance, cosine as the tiebreak."""
    return r.get("rrf_score", r.get("cosine_score", 0.0))


def _fuse_hops(collected: list[dict]) -> list[tuple[float, dict]]:
    """Deduplicate chunks across ReAct hops and score them for the final rank.

    Fusion PRESERVES the deterministic relevance magnitude. Collapsing every
    chunk to pure positional ``rrf_score(rank)`` made the agentic hit-scores
    identical for every query (avg_hit stdev≈0), so the generator got no
    relevance cue and abstained even when the gold chunk was in context. Each
    chunk's own cosine-fused DET score stays the primary signal; positional RRF
    is added only as a small tiebreak that respects cross-hop rank.
    """
    from graphknows.ranking.rrf import FUSION_ALPHA, rrf_score

    hop_chunks: dict[str, dict] = {}
    for r in collected:
        cid = r.get("chunk_id", "")
        if not cid:
            continue
        existing = hop_chunks.get(cid)
        if existing is None or _relevance(r) > _relevance(existing):
            hop_chunks[cid] = r

    ranked = sorted(hop_chunks.values(), key=lambda r: -_relevance(r))
    return [(_relevance(r) + FUSION_ALPHA * rrf_score(rank), r) for rank, r in enumerate(ranked)]


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
        settings: GraphKnowsSettings,
        store: Any,
    ) -> None:
        self._settings = settings
        self._store = store
        self._lm: Any = None
        self._vec_bridge: Any = None
        # DETRetriever instance — built once, shared across all retrieve() calls.
        self._det: Any = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_lm(self) -> Any:
        """Return the cached DSPy LM instance, building it once."""
        if self._lm is not None:
            return self._lm
        _require_dspy()
        from graphknows.llm import build_lm

        react_model = self._settings.extraction_model or self._settings.llm_model
        self._lm = build_lm(
            react_model,
            api_key=self._settings.llm_api_key,
            api_base=self._settings.llm_api_base,
            temperature=self._settings.llm_temperature,
            # Reasoning models spend much of the budget on reasoning_content, so a
            # small budget truncates the ReAct tool-call output and that hop silently
            # yields no chunks. Give it headroom.
            max_tokens=max(self._settings.llm_max_tokens, 8192),
            num_retries=1,
            timeout=120,  # per-request; without it litellm defaults to 6000s → a hung call wedges the run
        )
        return self._lm

    def _get_det_retriever(self) -> Any:
        """Return the cached DETRetriever instance, building it once from the same stores."""
        if self._det is not None:
            return self._det
        from graphknows.channels.registry import default_channels as build_channels
        from graphknows.retrieval.retriever import DETRetriever

        self._det = DETRetriever(
            store=self._store,
            embed_model=self._settings.embed_model,
            channels=build_channels(self._settings),
        )
        return self._det

    def _build_react(self, session_id: str, collected: list[dict]) -> Any:
        """Build a DSPy ReAct module whose tools append their hits to *collected*.

        The module is built per ``retrieve()`` call rather than cached, because
        its tool closures capture per-call state (``session_id`` and the
        ``collected`` buffer). Caching it on the instance would let concurrent
        retrievals rebind each other's tools and share one hit buffer, so a call
        would see another session's chunks — ``Memory`` promises concurrency
        safety. The LM and the DETRetriever stay cached; they hold no per-call
        state and are the expensive parts to build.

        Tools call ``DETRetriever.retrieve()`` as the single retrieval primitive
        so the agent amplifies — not competes with — the deterministic path.
        """
        dspy_mod = _require_dspy()

        from graphknows._async_bridge import _run as bg_run
        from graphknows.channels.temporal import filter_by_validity_window

        det = self._get_det_retriever()
        store = self._store

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
            except Exception as exc:
                log.warning("retrieve tool failed for %r: %s", subquery, exc)
                return []
            else:
                return chunks

        # ---- Tool 2: entity lookup (graph store) ----------------------
        def lookup_entity(name: str) -> list[dict]:
            """Look up chunks mentioning an entity by name in the memory graph.

            Args:
                name: Entity name or partial name to look up (case-insensitive).

            Returns:
                List of chunk dicts (chunk_id, text) that mention the entity.
            """
            try:
                return bg_run(store.query_entity_chunks([name], session_id or None, 10, None))
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
            except Exception as exc:
                log.warning("temporal_filter failed for window=%r: %s", window, exc)
                return list(collected)
            else:
                return filtered

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
            return dspy_mod.ReAct(
                RetrievalSignature,
                tools=[retrieve, lookup_entity, temporal_filter],
                max_iters=MAX_REACT_HOPS,
            )

    async def _degrade(
        self, det: Any, query: str, session_id: str, top_k: int, scope: str
    ) -> RetrievalContext:
        """Graceful degradation chain once the ReAct loop yields nothing.

        Tries the deterministic retriever, then a direct ANN over the store, then
        an empty context. Each rung is strictly simpler and more available than
        the one before, so a failure never propagates to the caller — recall
        degrades instead of erroring.
        """
        from graphknows.retrieval.models.context import RetrievalContext

        def _empty() -> RetrievalContext:
            return RetrievalContext(
                query=query, session_id=session_id, fused_results=[], search_type="AGENTIC"
            )

        try:
            ctx = await det.retrieve(query, session_id=session_id, top_k=top_k, scope=scope)
        except Exception as exc:
            log.warning("DETRetriever fallback failed (%s); using store.search_chunks_ann.", exc)
        else:
            # Tag fallback results so the caller can distinguish them.
            for r in ctx.fused_results:
                r["sources"] = "agentic-fallback"
            return RetrievalContext(
                query=query,
                session_id=session_id,
                semantic_matches=ctx.semantic_matches,
                fused_results=ctx.fused_results,
                search_type="AGENTIC",
            )

        try:
            return RetrievalContext(
                query=query,
                session_id=session_id,
                fused_results=await self._ann_results(query, session_id, top_k, scope),
                search_type="AGENTIC",
            )
        except Exception:
            log.exception("All fallbacks failed")
            return _empty()

    async def _ann_results(self, query: str, session_id: str, top_k: int, scope: str) -> list[dict]:
        """Last-resort recall: direct ANN over the graph store."""
        from graphknows.ranking.rrf import rrf_score
        from graphknows.retrieval.retriever import _scope_to_state
        from graphknows.storage.embedder import embed

        emb_array = await asyncio.to_thread(embed, [query], self._settings.embed_model)
        vector_chunks = await self._store.search_chunks_ann(
            emb_array[0].tolist(), top_k, session_id or None, _scope_to_state(scope)
        )
        return [
            {
                "chunk_id": r["chunk_id"],
                "doc_id": r.get("doc_id", ""),
                "text": r.get("text", ""),
                "metadata": {"cosine_score": r.get("cosine_score", 0.0), **r.get("metadata", {})},
                "rrf_score": round(rrf_score(i), 6),
                "sources": "agentic-fallback",
                "rank": i + 1,
                "entities": [],
            }
            for i, r in enumerate(vector_chunks)
        ]

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
        """Retrieve relevant context with DSPy ReAct backed by DETRetriever.

        Flow:
        1. Decompose the query into atomic sub-questions (with deterministic
           fallback to [query] when the LM is unavailable).
        2. Run a DSPy ReAct loop (≤ MAX_REACT_HOPS hops) whose tools all
           delegate to DETRetriever.retrieve() as the single retrieval primitive.
        3. Aggregate evidence across hops with RRF fusion.
        4. If the agent errors, fall back to a plain DETRetriever.retrieve() call.
        """
        from graphknows.retrieval.models.context import ChunkMatch, RetrievalContext
        from graphknows.retrieval.query.decompose import decompose

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

        # Step 2 — Run the ReAct loop; its tools append hits to `collected`.
        # Both the buffer and the module are per-call so concurrent retrievals
        # cannot clear or read each other's evidence.
        collected: list[dict] = []
        react = self._build_react(session_id, collected)

        react_failed = False
        try:

            def _run_react() -> Any:
                with dspy.settings.context(lm=lm):
                    # Run one ReAct call per sub-question so each hop is
                    # bounded by MAX_REACT_HOPS.  Tools push hits to
                    # the per-call buffer for later aggregation.
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
        # Graceful degradation chain: ReAct → DETRetriever → vec.search_chunks.
        if react_failed or not collected:
            if react_failed:
                log.info("Falling back to DETRetriever.retrieve() after agent failure.")
            return await self._degrade(det, query, session_id, top_k, scope)

        # Always add a direct DET pass on the VERBATIM query so the agentic
        # context is a superset of the llm_free context — the ReAct loop can
        # only add to it, never retrieve worse. This fixes agentic-specific
        # over-abstention where ReAct reformulated the query and lost the gold
        # chunk that the deterministic path on the literal question finds.
        try:
            base_ctx = await det.retrieve(query, session_id=session_id, top_k=top_k, scope=scope)
            collected.extend(base_ctx.fused_results)
        except Exception as exc:
            log.debug("base DET pass skipped: %s", exc)

        # Step 4 — Aggregate evidence across all ReAct hops with RRF.
        candidates = _fuse_hops(collected)
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

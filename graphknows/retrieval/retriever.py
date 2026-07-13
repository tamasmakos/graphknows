"""Deterministic entity-traversal + vector hybrid RAG with RRF fusion.

Collectors run concurrently over a single per-namespace ``GraphStore``, each
filtered by a lifecycle ``state`` derived from the query scope (raw /
consolidated / both): entity traversal, dense vector ANN, BM25, temporal, and
topic match. The spine is channel-blind — it fuses and reranks only.

Topic nodes are LLM-assisted enrichment artifacts; deterministic retrieval uses
community metadata as score shaping rather than returning synthetic topic hits.

WS-2 additions
--------------
* BM25 sparse channel — replaces the homegrown two-pass full-text with a proper
  BM25 ranking (``rank_bm25`` when importable, clean tokenised fallback otherwise)
  over chunk texts, feeding one RRF source.
* Cross-encoder rerank — the top ~50 RRF candidates are re-scored by a local
  CPU cross-encoder (``rerank.py``); no-op when the model is unavailable.
* PageRank boost — ``e.pagerank`` from ENTITY nodes is folded into the fusion
  score with a bounded log-compressed multiplier.  Absent/zero → no effect.
* IDF discriminativeness — entities are smooth-IDF-down-weighted instead of
  hard-dropped when they appear in >50% of the corpus.
* Temporal recency boost — relation-edge ``_temporal_score`` from
  ``filter_by_validity_window`` is folded into temporal-traversal candidates.
"""

from __future__ import annotations

import asyncio
import logging
import math
import os
import re
from typing import Any

from graphknows.ranking.rrf import (
    FUSION_ALPHA,
    RRF_RELATION_BOOST,
    community_boost,
    pagerank_boost,
    rrf_score,
)
from graphknows.ranking.rrf import shape_score as _shape_score
from graphknows.retrieval.base import BaseGraphRetriever
from graphknows.channels.base import ChannelContext
from graphknows.retrieval.models.context import RetrievalContext
from graphknows.ranking.rerank import rerank as _ce_rerank
from graphknows.storage.arcadedb.graph_store import GraphStore
from graphknows.storage.embedder import embed
from graphknows.channels.temporal import query_date_candidates

# ---------------------------------------------------------------------------
# Optional BM25 import — lazy, graceful fallback to tf-style scoring
# ---------------------------------------------------------------------------
try:
    from rank_bm25 import BM25Okapi as _BM25  # type: ignore[import-untyped]

    _HAS_BM25 = True
except ImportError:
    _HAS_BM25 = False
    _BM25 = None  # type: ignore[assignment,misc]

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# BM25 in-memory scorer
# ---------------------------------------------------------------------------

def _bm25_rank(
    query_tokens: list[str],
    chunks: list[dict[str, Any]],
) -> list[tuple[float, dict[str, Any]]]:
    """Rank *chunks* by BM25 relevance to *query_tokens*.

    Uses ``rank_bm25.BM25Okapi`` when available; falls back to a simple
    normalised term-frequency scorer so the channel is always live.

    Args:
        query_tokens: Pre-filtered, lower-cased query tokens.
        chunks: Dicts that must contain at least ``"text"`` (str) and
            ``"chunk_id"`` (str).

    Returns:
        List of ``(score, chunk_dict)`` pairs sorted by score descending.
        Empty list when *query_tokens* or *chunks* is empty.
    """
    if not query_tokens or not chunks:
        return []

    tokenised = [
        re.findall(r"[a-z0-9][a-z0-9']*", (c.get("text") or "").lower())
        for c in chunks
    ]

    if _HAS_BM25:
        bm25 = _BM25(tokenised)
        scores = bm25.get_scores(query_tokens).tolist()
    else:
        # Fallback: normalised TF with IDF approximation.
        # IDF: log((N+1) / (df+1)) — same formula BM25Okapi uses.
        n = len(chunks)
        df: dict[str, int] = {}
        for toks in tokenised:
            for t in set(toks):
                df[t] = df.get(t, 0) + 1
        idf_map = {
            t: math.log((n + 1) / (d + 1)) + 1.0
            for t, d in df.items()
        }
        scores = []
        for toks in tokenised:
            if not toks:
                scores.append(0.0)
                continue
            tf: dict[str, float] = {}
            for t in toks:
                tf[t] = tf.get(t, 0.0) + 1.0
            length = len(toks)
            sc = sum(
                idf_map.get(qt, 0.0) * tf.get(qt, 0.0) / length
                for qt in query_tokens
            )
            scores.append(sc)

    paired = sorted(zip(scores, chunks), key=lambda x: -x[0])
    return [(s, c) for s, c in paired if s > 0.0]


def _scope_to_state(scope: str) -> str | None:
    """Map a retrieval scope to a golden-layer lifecycle state filter.

    ``stm`` -> ``raw`` (freshly ingested), ``ltm`` -> ``consolidated``,
    ``both`` (or anything else) -> ``None`` (no state filter — all chunks).
    """
    if scope == "stm":
        return "raw"
    if scope == "ltm":
        return "consolidated"
    return None


_Q_STRIP = re.compile(
    r"^(?:what(?:\s+(?:is|are|was|were|do|does|did|would|could))?|"
    r"when(?:\s+(?:did|was|were|is|are))?|"
    r"where(?:\s+(?:did|was|were|is|are))?|"
    r"who(?:\s+(?:is|are|was|were))?|"
    r"which(?:\s+(?:is|are|was|were))?|"
    r"how(?:\s+(?:many|much|long|often|did|was|were|is|are))?|"
    r"why(?:\s+(?:did|was|were|is|are))?|"
    r"(?:did|does|do|was|were|is|are|has|have|had|would|could|should))\s+",
    re.IGNORECASE,
)


def _declarative_variant(question: str) -> str | None:
    """Strip interrogative prefix to produce a keyword-dense form closer to answer text.

    'What is Caroline's profession?' → 'Caroline's profession'
    Embedding the declarative form gets the ANN closer to how answers appear in chunks.
    """
    q = question.strip().rstrip("?").strip()
    stripped = _Q_STRIP.sub("", q, count=1).strip()
    if stripped and stripped.lower() != q.lower() and len(stripped) >= 5:
        return stripped
    return None


# Entities mentioned in more than this fraction of the corpus carry no
# discriminative signal (the conversation participants appear in nearly every
# chunk), so seeding entity traversal on them floods recall with the whole
# conversation. Such entities are dropped from traversal — the semantic and
# full-text channels still cover their chunks.
_ENTITY_SATURATION_RATIO: float = float(os.environ.get("GRAPHKNOWS_ENTITY_SATURATION_RATIO", "0.5"))

class DETRetriever(BaseGraphRetriever):
    """Deterministic three-collector hybrid retriever."""

    def __init__(
        self,
        store: GraphStore,
        embed_model: str = "",
        channels: "list[Any] | None" = None,
    ) -> None:
        self.store = store
        self.embed_model = embed_model
        # Channels provide ranked candidates; this spine fuses/reranks them and is
        # channel-blind. Defaults to the seven built-in collectors (behaviour
        # identical to the original hardcoded gather); callers pass a richer set
        # (e.g. + frames) built from settings via channels.registry.
        if channels is None:
            from graphknows.channels.collectors import DEFAULT_COLLECTOR_CHANNELS
            channels = [cls() for cls in DEFAULT_COLLECTOR_CHANNELS]
        self._channels = channels

    async def retrieve(
        self,
        query: str,
        session_id: str,
        top_k: int = 10,
        scope: str = "both",
    ) -> RetrievalContext:
        """Run deterministic recall and return RRF-merged results."""
        log.debug(
            "DET retrieve query=%r session=%s scope=%s top_k=%d",
            query,
            session_id,
            scope,
            top_k,
        )

        # Embed original query + optional declarative variant in one batch call.
        var_text = _declarative_variant(query)
        queries = [query, var_text] if var_text else [query]
        emb_array = await asyncio.to_thread(embed, queries, self.embed_model)
        emb = emb_array[0].tolist()
        emb2: list[float] | None = emb_array[1].tolist() if var_text else None
        if emb2:
            log.debug("Query variant: %r", var_text)

        # words_graph: include 3-char tokens (month names, short proper nouns) for
        # entity/graph matching where substring-of-name is the lookup key.
        # words: >3-char only for full-text search to reduce noise.
        words_graph = [w for w in re.findall(r"[a-z0-9][a-z0-9']*", query.lower()) if len(w) >= 3]
        words = [w for w in words_graph if len(w) > 3]

        # Channels run concurrently; a failure in ANY of them raises loudly
        # (no return_exceptions swallow) so a broken channel surfaces immediately
        # rather than silently degrading recall. Each returns chunk_id -> (score,
        # info); the spine below is channel-blind (fuses + reranks only).
        ctx = ChannelContext(
            query=query, session_id=session_id, scope=scope,
            emb=emb, emb2=emb2, words=words, words_graph=words_graph,
            state=_scope_to_state(scope),
        )
        sources = await asyncio.gather(
            *(c.collect(ctx, self, top_k) for c in self._channels)
        )
        collectors_fired = {
            c.name: len(s) for c, s in zip(self._channels, sources, strict=False)
        }

        combined: dict[str, tuple[float, dict[str, Any]]] = {}
        for src in sources:
            for cid, (score, info) in src.items():
                if cid in combined:
                    old_score, old_info = combined[cid]
                    combined[cid] = (
                        old_score + score,
                        {**old_info, "sources": old_info["sources"] | info["sources"]},
                    )
                else:
                    combined[cid] = (score, info)

        # Fetch metadata for all candidates that lack it so shaping can run
        # before we cut to top_k — shaping affects ordering, not just display.
        needs_meta = [cid for cid, (_, info) in combined.items() if not info.get("metadata")]
        text_map = await self.store.get_chunks_by_ids(needs_meta) if needs_meta else {}

        # Apply _community_boost, PageRank boost, and _shape_score to every
        # candidate BEFORE sorting so the final order reflects the shaped score.
        shaped_combined: list[tuple[str, float, dict[str, Any], dict[str, Any]]] = []
        for cid, (base_score, info) in combined.items():
            fc = text_map.get(cid, {})
            meta = fc.get("metadata") or info.get("metadata") or {}
            # PageRank centrality boost: e.pagerank from WS-6 ENTITY projection.
            # Absent/zero → pagerank_boost returns 1.0 (no effect).
            pr = float(meta.get("pagerank", 0.0) or 0.0)
            pr_mult = pagerank_boost(pr)
            shaped = _shape_score(
                base_score * _community_boost(meta) * pr_mult,
                meta.get("memory_class"),
            )
            shaped_combined.append((cid, shaped, info, fc))

        shaped_combined.sort(key=lambda x: -x[1])

        pre_rerank: list[dict[str, Any]] = []
        for rank_i, (cid, shaped, info, fc) in enumerate(shaped_combined):
            meta = fc.get("metadata") or info.get("metadata") or {}
            ents = info.get("entities", [])
            pre_rerank.append(
                {
                    "chunk_id": cid,
                    "doc_id": fc.get("doc_id") or info.get("doc_id", ""),
                    "text": fc.get("text") or info.get("text", ""),
                    "metadata": meta,
                    "rrf_score": round(shaped, 6),
                    "sources": ",".join(sorted(info["sources"])),
                    "rank": rank_i + 1,
                    "entities": ents if isinstance(ents, list) else list(ents),
                }
            )

        # Cross-encoder rerank the whole pool (no-op when the model is
        # unavailable), then cut to top_k and renumber.
        ranked = await asyncio.to_thread(_ce_rerank, query, pre_rerank, len(pre_rerank))
        fused = ranked[:top_k]
        for i, item in enumerate(fused):
            item["rank"] = i + 1

        # Transparency trace: how each surviving candidate scored.
        trace = {
            "query": query,
            "session_id": session_id,
            "collectors_fired": collectors_fired,
            "candidates": [
                {
                    "rank": fc["rank"],
                    "chunk_id": fc["chunk_id"],
                    "doc_id": fc.get("doc_id", ""),
                    "sources": fc.get("sources", ""),
                    "score": fc.get("rrf_score", 0.0),
                    "text": (fc.get("text", "") or "")[:200],
                }
                for fc in fused
            ],
        }

        return RetrievalContext(
            query=query,
            session_id=session_id,
            fused_results=fused,
            search_type="DET",
            trace=trace,
        )

    async def execute_cypher(
        self,
        cypher_template: str,
        params: dict | None = None,
    ) -> list[dict]:
        """Execute a parameterized read-only Cypher query against the graph store."""
        if re.search(r"\{[^}]+\}", cypher_template):
            raise ValueError(
                "Cypher template must not contain {var} interpolation; "
                "use $param placeholders and pass values via params"
            )
        forbidden = re.compile(r"\b(CREATE|MERGE|SET|DELETE|DETACH|REMOVE|DROP)\b", re.IGNORECASE)
        if forbidden.search(cypher_template):
            raise ValueError("Only read-only Cypher is permitted")
        rows = await self.store._qry(cypher_template, **(params or {}))
        return [dict(r) for r in rows]

    async def entity_boost(
        self,
        words: list[str],
        session_id: str,
        state: str | None,
        top_k: int,
    ) -> dict[str, tuple[float, dict[str, Any]]]:
        """Collector: chunks reachable from query-token entity nodes (MENTIONS)."""
        result: dict[str, tuple[float, dict[str, Any]]] = {}
        if not words:
            return result
        entity_chunks = await self.store.query_entity_chunks(
            words, session_id or None, top_k, state
        )
        for rank, ec in enumerate(entity_chunks):
            cid = ec["chunk_id"]
            _add_result(
                result,
                cid,
                # boost=2: entity hits are high-precision; outcompete pure-ANN noise.
                rrf_score(rank, boost=2.0),
                {
                    "text": ec.get("text", ""),
                    "entities": ec.get("entities", []),
                    "sources": {"entity"},
                    "doc_id": "",
                    "metadata": {},
                },
            )
        return result

    _FT_STOPWORDS: frozenset[str] = frozenset(
        {
            "when",
            "what",
            "where",
            "which",
            "that",
            "this",
            "with",
            "from",
            "have",
            "been",
            "does",
            "were",
            "they",
            "their",
            "there",
            "about",
            "like",
            "just",
            "some",
            "than",
            "then",
            "into",
            "will",
            "your",
            "would",
            "could",
            "should",
            "might",
            "much",
        }
    )

    async def bm25(
        self,
        words: list[str],
        session_id: str,
        state: str | None,
        top_k: int,
    ) -> dict[str, tuple[float, dict[str, Any]]]:
        """BM25 sparse channel over chunks.

        Fetches a broad OR-pool from the store's full-text scan, then re-ranks
        with BM25 for one clean RRF source. Uses ``rank_bm25.BM25Okapi`` when
        importable; falls back to a normalised TF-IDF scorer so the channel is
        always live.
        """
        result: dict[str, tuple[float, dict[str, Any]]] = {}
        specific = [w for w in words if w not in self._FT_STOPWORDS]
        if not specific:
            return result

        or_chunks = await self.store.full_text_search_chunks(
            specific, session_id or None, top_k * 3, require_all=False, state=state
        )
        ranked = _bm25_rank(specific, or_chunks)
        for rank, (bm25_score, c) in enumerate(ranked[:top_k]):
            cid = c["chunk_id"]
            # BM25 score folds in as a tiebreaker on top of RRF rank weight.
            rrf = rrf_score(rank, boost=3.0) + FUSION_ALPHA * bm25_score
            result[cid] = (
                rrf,
                {
                    "text": c.get("text", ""),
                    "entities": [],
                    "sources": {"text"},
                    "doc_id": "",
                    "metadata": {},
                },
            )
        return result

    async def temporal_traversal(
        self,
        query: str,
        session_id: str,
        state: str | None,
        top_k: int,
    ) -> dict[str, tuple[float, dict[str, Any]]]:
        """Match chunks by canonical TEMPORAL node (MENTIONS_DATE).

        No-op when the query contains no resolvable date.
        """
        result: dict[str, tuple[float, dict[str, Any]]] = {}
        candidates = query_date_candidates(query)
        if not candidates:
            return result
        rows = await self.store.query_temporal_chunks(
            candidates, session_id or None, top_k, state
        )
        for rank, row in enumerate(rows):
            _add_result(
                result,
                row["chunk_id"],
                rrf_score(rank, RRF_RELATION_BOOST),
                {
                    "text": row.get("text", ""),
                    "entities": [],
                    "sources": {"temporal"},
                    "doc_id": "",
                    "metadata": {},
                },
            )
        return result

    async def topic_match(
        self,
        emb: list[float],
        session_id: str,
        state: str | None,
        top_k: int,
    ) -> dict[str, tuple[float, dict[str, Any]]]:
        """Semantic topic match -> inject the topic summary as a synthetic passage.

        Topics are pre-aggregated community summaries written at consolidation
        (llm_assisted only). No-op when no topics exist, so it never regresses
        the base pipeline.
        """
        result: dict[str, tuple[float, dict[str, Any]]] = {}
        topics = await self.store.search_topics(emb, top_k=2)
        for t_rank, topic in enumerate(topics):
            summary = (topic.get("summary") or "").strip()
            tid = topic.get("topic_id") or ""
            if not summary:
                continue
            _add_result(
                result,
                f"topic:{tid}",
                rrf_score(t_rank, boost=1.5),
                {
                    "text": summary,
                    "entities": [],
                    "sources": {"topic"},
                    "doc_id": "",
                    "metadata": {"is_topic_summary": True},
                },
            )
        return result

    async def semantic_ann(
        self,
        emb: list[float],
        session_id: str,
        state: str | None,
        top_k: int,
        emb2: list[float] | None = None,
    ) -> dict[str, tuple[float, dict[str, Any]]]:
        """Collector: dense vector ANN over chunk embeddings (state-filtered)."""
        result: dict[str, tuple[float, dict[str, Any]]] = {}
        k = top_k * 2
        sid = session_id or None

        coros = [self.store.search_chunks_ann(emb, k, sid, state)]
        if emb2 is not None:
            coros.append(self.store.search_chunks_ann(emb2, k, sid, state))

        all_rows = await asyncio.gather(*coros)  # failures raise loudly

        # First pass: best (minimum) rank per chunk across the original query and
        # the declarative variant, so a chunk surfaced by both embeddings does not
        # receive 2x RRF mass from a single logical 'semantic' source.
        best_rank: dict[str, int] = {}
        best_row: dict[str, dict] = {}
        for rows in all_rows:
            for rank, row in enumerate(rows):
                cid = row["chunk_id"]
                if cid not in best_rank or rank < best_rank[cid]:
                    best_rank[cid] = rank
                    best_row[cid] = row

        # Second pass: emit ONE RRF contribution per chunk for the semantic source.
        for cid, rank in best_rank.items():
            row = best_row[cid]
            cosine = float(row.get("cosine_score", 0.0) or 0.0)
            score = rrf_score(rank) + FUSION_ALPHA * cosine
            _add_result(
                result,
                cid,
                score,
                {
                    "text": row.get("text", ""),
                    "entities": [],
                    "sources": {"semantic"},
                    "doc_id": "",
                    "metadata": {},
                },
            )
        return result


def _add_result(
    result: dict[str, tuple[float, dict[str, Any]]],
    chunk_id: str,
    score: float,
    info: dict[str, Any],
) -> None:
    if chunk_id in result:
        old_score, old_info = result[chunk_id]
        result[chunk_id] = (
            old_score + score,
            {**old_info, "sources": old_info["sources"] | info["sources"]},
        )
    else:
        result[chunk_id] = (score, info)


def _community_boost(metadata: dict[str, Any]) -> float:
    """Return a bounded score multiplier from community metadata."""
    return community_boost(metadata.get("community_weight", 1.0))

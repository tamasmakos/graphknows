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
  CPU cross-encoder (``rerank.py``); raises if the model is unavailable.
* PageRank boost and IDF discriminativeness were never wired: the former read
  ``pagerank`` from CHUNK metadata while the value lives on ENTITY nodes (so the
  multiplier was always 1.0), and nothing ever called the latter. Both have been
  removed. Reintroducing PageRank shaping needs a chunk-level rollup
  (cf. ``chunk_community_weights``) and should be measured before being believed.
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

import numpy as np

from graphknows.channels.base import ChannelContext
from graphknows.channels.temporal import query_date_candidates
from graphknows.ranking.rerank import rerank as _ce_rerank
from graphknows.ranking.rrf import (
    FUSION_ALPHA,
    RRF_RELATION_BOOST,
    community_boost,
    rrf_score,
)
from graphknows.ranking.rrf import shape_score as _shape_score
from graphknows.retrieval.base import BaseGraphRetriever
from graphknows.retrieval.models.context import RetrievalContext
from graphknows.storage.arcadedb.graph_store import GraphStore
from graphknows.storage.embedder import embed

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

    tokenised = [re.findall(r"[a-z0-9][a-z0-9']*", (c.get("text") or "").lower()) for c in chunks]

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
        idf_map = {t: math.log((n + 1) / (d + 1)) + 1.0 for t, d in df.items()}
        scores = []
        for toks in tokenised:
            if not toks:
                scores.append(0.0)
                continue
            tf: dict[str, float] = {}
            for t in toks:
                tf[t] = tf.get(t, 0.0) + 1.0
            length = len(toks)
            sc = sum(idf_map.get(qt, 0.0) * tf.get(qt, 0.0) / length for qt in query_tokens)
            scores.append(sc)

    paired = sorted(zip(scores, chunks, strict=False), key=lambda x: -x[0])
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


# Weight of the query-entity affinity term in the pool rerank: how much to boost
# a candidate for each of the query's entities it mentions (via the MENTIONS
# graph), on top of its query cosine. 0.20 won an offline grid on both
# benchmarks; 0 restores pure cosine ordering.
_RERANK_QENT_WEIGHT: float = float(os.environ.get("GRAPHKNOWS_RERANK_QENT_WEIGHT", "0.20"))
# Entities matched to the query are capped here before normalising, so a chunk
# that mentions many query entities does not swamp the cosine signal.
_QENT_CAP: int = 5
_QENT_TOKEN = re.compile(r"[a-z0-9']+")

# After cutting to top_k, append each result's ±N seq-adjacent same-document
# chunks. Long ingested messages split across consecutive chunks, so the answer
# often sits in a sibling of the matched chunk — a LONG-DOCUMENT fix.
#
# Default 0 (off): it HELPS long-document corpora (BEAM: accuracy 0.371 -> 0.426
# at radius 2) but HURTS conversational memory, where short turns don't split
# and the siblings are just context-rot noise (LoCoMo: 0.895 -> 0.814 at radius
# 1). Enable per-corpus via GRAPHKNOWS_NEIGHBOR_RADIUS.
_NEIGHBOR_RADIUS: int = int(os.environ.get("GRAPHKNOWS_NEIGHBOR_RADIUS", "0"))


def _fact_context_enabled() -> bool:
    """Whether the D8 structured fact-sheet is fed to the generator (default on).

    Read at call time (not import time) so the eval A/B can flip
    ``GRAPHKNOWS_FACT_CONTEXT`` per MCP-server subprocess.
    """
    return os.environ.get("GRAPHKNOWS_FACT_CONTEXT", "true").strip().lower() not in (
        "0",
        "false",
        "no",
    )


class DETRetriever(BaseGraphRetriever):
    """Deterministic three-collector hybrid retriever."""

    def __init__(
        self,
        store: GraphStore,
        embed_model: str = "",
        channels: list[Any] | None = None,
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

        ctx = await self._build_channel_context(query, session_id, scope)

        # Channels run concurrently; a failure in ANY of them raises loudly
        # (no return_exceptions swallow) so a broken channel surfaces immediately
        # rather than silently degrading recall. Each returns chunk_id -> (score,
        # info); the spine below is channel-blind (fuses + reranks only).
        sources = await asyncio.gather(*(c.collect(ctx, self, top_k) for c in self._channels))
        collectors_fired = {c.name: len(s) for c, s in zip(self._channels, sources, strict=False)}

        combined: dict[str, tuple[float, dict[str, Any]]] = {}
        for src in sources:
            for cid, (score, info) in src.items():
                _add_result(combined, cid, score, info)

        pre_rerank = await self._shape_and_rank(combined)

        # Order the pool by query-chunk cosine + query-entity affinity, then
        # cross-encoder rerank (raises if the reranker is unavailable), then cut.
        ranked = await self._rerank_pool(ctx.emb, query, pre_rerank)
        ranked = await asyncio.to_thread(_ce_rerank, query, ranked, len(ranked))
        fused = ranked[:top_k]
        fused = await self._expand_neighbors(fused)
        for i, item in enumerate(fused):
            item["rank"] = i + 1

        return RetrievalContext(
            query=query,
            session_id=session_id,
            fused_results=fused,
            search_type="DET",
            trace=_build_trace(query, session_id, collectors_fired, fused),
            facts=await self._fact_sheet(ctx.words_graph),
        )

    async def _expand_neighbors(self, fused: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Merge each result's seq-adjacent same-file chunks INTO its own text.

        Ingested long messages split across consecutive same-file chunks, so the
        answer often lives in a sibling of the chunk that matched the query. Each
        result's ±_NEIGHBOR_RADIUS siblings are stitched into that result's text
        in seq order, rather than added as separate passages — the passage COUNT
        is unchanged, so the downstream ``[:gen_context_k]`` cut still keeps the
        same number of distinct seeds instead of spending its budget on the top
        seeds' siblings. Measured on BEAM's real evidence labels, over the 25
        passages the generator actually sees, merging lifts evidence recall
        0.305 -> 0.547 where the old append-then-cut dropped it to 0.305 (below
        the 0.353 no-expansion baseline). Each sibling is stitched into only the
        first seed that claims it, so shared siblings are not duplicated. Radius
        0 disables it.
        """
        if _NEIGHBOR_RADIUS < 1 or not fused:
            return fused
        ids = [str(f.get("chunk_id") or "") for f in fused]
        try:
            neighbors = await self.store.neighbor_chunks(ids, _NEIGHBOR_RADIUS)
        except Exception as exc:
            log.warning("neighbor expansion unavailable (%s); skipping", exc)
            return fused
        if not neighbors:
            return fused
        claimed = set(ids)
        for item in fused:
            cid = str(item.get("chunk_id") or "")
            sibs = [
                s for s in neighbors.get(cid, []) if str(s.get("chunk_id") or "") not in claimed
            ]
            if not sibs:
                continue
            claimed.update(str(s.get("chunk_id") or "") for s in sibs)
            # Seed first (the query-relevant chunk), then its siblings in seq
            # order — the matched text leads, its split-message context follows.
            sibs.sort(key=lambda s: s.get("seq", 0))
            parts = [str(item.get("text") or "")] + [s.get("text", "") for s in sibs]
            item["text"] = " ".join(p for p in parts if p)
            item.setdefault("metadata", {})["merged_neighbors"] = len(sibs)
        return fused

    async def _rerank_pool(
        self, query_emb: list[float], query: str, candidates: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Order the fused pool by query-chunk cosine + query-entity affinity.

        The channels' job is pool MEMBERSHIP; ordering is ``cosine + w · (query
        entities this chunk mentions)``. Pure cosine is topic-similarity only —
        for a multi-evidence question it ranks near-duplicates about one
        sub-topic first. The affinity term spends the graph's own MENTIONS
        structure: a chunk that actually names the query's subjects is more
        likely to carry an answer than one that merely reads similar.

        Measured offline over cached candidate pools (grid-searched):
        - vs old RRF (conv-26): recall@10 0.739 -> 0.845.
        - vs pure cosine at the operating point (k=25): LoCoMo evidence_recall
          multi-hop 0.871 -> 0.906 (real labels), BEAM weak-category coverage
          0.673 -> 0.697. Diversity-coverage variants (new-entity / new-doc
          greedy) tied pure cosine and were dropped.

        Chunk vectors + entity lists are already persisted, so this is two
        round trips plus O(n) arithmetic — no model, no new dependency.
        Candidates with no stored embedding keep their RRF order behind the
        scored ones rather than being dropped. Weight 0 restores pure cosine.
        """
        if len(candidates) < 2 or not query_emb:
            return candidates
        ids = [str(c.get("chunk_id") or "") for c in candidates]
        try:
            meta = await self.store.chunk_rerank_meta(ids)
        except Exception as exc:
            log.warning("pool rerank unavailable (%s); keeping RRF order", exc)
            return candidates
        if not meta:
            return candidates

        qtoks = {w for w in _QENT_TOKEN.findall(query.lower()) if len(w) > 2}
        q = np.asarray(query_emb, dtype=np.float32)
        qn = float(np.linalg.norm(q)) or 1.0
        scored: list[tuple[float, int, dict[str, Any]]] = []
        unscorable: list[dict[str, Any]] = []
        for i, cand in enumerate(candidates):
            m = meta.get(str(cand.get("chunk_id") or ""))
            if not m or not m.get("emb"):
                # No stored vector: keep behind everything scored, in RRF order.
                unscorable.append(cand)
                continue
            v = np.asarray(m["emb"], dtype=np.float32)
            cos = float(v @ q / ((float(np.linalg.norm(v)) or 1.0) * qn))
            # Query-entity affinity: an entity matches on a whole word, mirroring
            # resolve_query_entities, so "there" never matches "theresa".
            affinity = sum(1 for e in m.get("ents") or () if qtoks & set(str(e).split()))
            cand["cosine"] = round(cos, 4)
            score = cos + _RERANK_QENT_WEIGHT * min(affinity, _QENT_CAP) / _QENT_CAP
            scored.append((score, i, cand))
        scored.sort(key=lambda t: (-t[0], t[1]))
        return [c for _, _, c in scored] + unscorable

    async def _build_channel_context(
        self, query: str, session_id: str, scope: str
    ) -> ChannelContext:
        """Embed the query (plus its declarative variant) and tokenise it."""
        # Embed original query + optional declarative variant in one batch call.
        var_text = _declarative_variant(query)
        queries = [query, var_text] if var_text else [query]
        emb_array = await asyncio.to_thread(embed, queries, self.embed_model)
        emb2: list[float] | None = emb_array[1].tolist() if var_text else None
        if emb2:
            log.debug("Query variant: %r", var_text)

        # words_graph: include 3-char tokens (month names, short proper nouns) for
        # entity/graph matching where substring-of-name is the lookup key.
        # words: >3-char only for full-text search to reduce noise.
        words_graph = [w for w in re.findall(r"[a-z0-9][a-z0-9']*", query.lower()) if len(w) >= 3]
        return ChannelContext(
            query=query,
            session_id=session_id,
            scope=scope,
            emb=emb_array[0].tolist(),
            emb2=emb2,
            words=[w for w in words_graph if len(w) > 3],
            words_graph=words_graph,
            state=_scope_to_state(scope),
        )

    async def _shape_and_rank(
        self, combined: dict[str, tuple[float, dict[str, Any]]]
    ) -> list[dict[str, Any]]:
        """Score-shape every candidate, sort, and render the pre-rerank rows.

        Shaping runs over the whole pool before any cut, because it affects
        ordering and not just display.
        """
        # Fetch metadata for all candidates that lack it so shaping can run
        # before we cut to top_k.
        needs_meta = [cid for cid, (_, info) in combined.items() if not info.get("metadata")]
        text_map = await self.store.get_chunks_by_ids(needs_meta) if needs_meta else {}
        # Community boost (WS4): derived from each chunk's MENTIONS-linked
        # entities' native-Louvain community_id, not stored per-chunk metadata.
        community_weights = await self.store.chunk_community_weights(list(combined.keys()))

        shaped_combined: list[tuple[str, float, dict[str, Any], dict[str, Any]]] = []
        for cid, (base_score, info) in combined.items():
            fc = text_map.get(cid, {})
            meta = fc.get("metadata") or info.get("metadata") or {}
            if cid in community_weights:
                meta = {**meta, "community_weight": community_weights[cid]}
            shaped = _shape_score(
                base_score * _community_boost(meta),
                meta.get("memory_class"),
            )
            shaped_combined.append((cid, shaped, info, fc))

        shaped_combined.sort(key=lambda x: -x[1])

        rows: list[dict[str, Any]] = []
        for rank_i, (cid, shaped, info, fc) in enumerate(shaped_combined):
            ents = info.get("entities", [])
            rows.append(
                {
                    "chunk_id": cid,
                    "doc_id": fc.get("doc_id") or info.get("doc_id", ""),
                    "text": fc.get("text") or info.get("text", ""),
                    "metadata": fc.get("metadata") or info.get("metadata") or {},
                    "rrf_score": round(shaped, 6),
                    "sources": ",".join(sorted(info["sources"])),
                    "rank": rank_i + 1,
                    "entities": ents if isinstance(ents, list) else list(ents),
                }
            )
        return rows

    async def _fact_sheet(self, words_graph: list[str]) -> list[str]:
        """Structured fact-sheet (D8) for the query's resolved entities.

        Surfaces their typed relations + frame-instance role fills. Soft-fails —
        a fact-context error never degrades recall.
        """
        if not _fact_context_enabled():
            return []
        try:
            resolved = await self.store.resolve_query_entities(words_graph)
            return await self.store.entity_fact_context(resolved) if resolved else []
        except Exception as exc:  # pragma: no cover - defensive soft-fail
            log.debug("fact context failed (soft): %s", exc)
            return []

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
        rows = await self.store.query(cypher_template, **(params or {}))
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
        rows = await self.store.query_temporal_chunks(candidates, session_id or None, top_k, state)
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


def _build_trace(
    query: str,
    session_id: str,
    collectors_fired: dict[str, int],
    fused: list[dict[str, Any]],
) -> dict[str, Any]:
    """Transparency trace: which collectors fired and how each survivor scored."""
    return {
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


def _add_result(
    result: dict[str, tuple[float, dict[str, Any]]],
    chunk_id: str,
    score: float,
    info: dict[str, Any],
) -> None:
    """Merge a candidate into *result*: sum scores, union source labels."""
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

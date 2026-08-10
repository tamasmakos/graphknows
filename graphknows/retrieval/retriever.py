"""Deterministic hybrid RAG: four baseline collectors, RRF fusion, one ranking.

The four collectors below (entity traversal, dense vector ANN, BM25, temporal)
run concurrently over one per-namespace ``GraphStore``, each filtered by a
lifecycle ``state`` derived from the query scope, alongside whatever optional
channels ``channels.registry.default_channels`` turned on. The spine is
signal-blind: it merges the candidates, sorts by fused score, and cuts to top_k.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
from typing import Any

try:
    from rank_bm25 import BM25Okapi
except ModuleNotFoundError as exc:  # pragma: no cover - exercised only on broken installs
    from graphknows.exceptions import BrokenInstallError

    raise BrokenInstallError("The BM25 retrieval channel", "rank-bm25") from exc

from graphknows.channels.base import ChannelContext
from graphknows.ranking.rrf import (
    FUSION_ALPHA,
    RRF_RELATION_BOOST,
    rrf_score,
)
from graphknows.retrieval.models.context import RetrievalContext
from graphknows.storage.arcadedb.graph_store import GraphStore
from graphknows.storage.embedder import embed
from graphknows.temporal import asks_for_a_date, query_date_candidates

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# BM25 in-memory scorer
# ---------------------------------------------------------------------------


def _bm25_rank(
    query_tokens: list[str],
    chunks: list[dict[str, Any]],
) -> list[tuple[float, dict[str, Any]]]:
    """Rank *chunks* by BM25 relevance to *query_tokens*.

    Uses ``rank_bm25.BM25Okapi``.

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

    bm25 = BM25Okapi(tokenised)
    scores = bm25.get_scores(query_tokens).tolist()

    # Keep chunks that actually contain a query term, NOT chunks with a positive
    # score. BM25Okapi's IDF is log((N - df + 0.5)/(df + 0.5)), which goes
    # NEGATIVE once a term appears in more than about half the corpus —
    # routine for the small candidate pools this is called with. A
    # `score > 0` filter therefore discards genuine matches, and could empty
    # the channel entirely.
    matched = [bool(set(query_tokens) & set(toks)) for toks in tokenised]
    paired = sorted(
        ((s, c) for s, c, hit in zip(scores, chunks, matched, strict=False) if hit),
        key=lambda x: -x[0],
    )
    return paired


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


# After cutting to top_k, append each result's ±N seq-adjacent same-document
# chunks. Long ingested messages split across consecutive chunks, so the answer
# often sits in a sibling of the matched chunk — a LONG-DOCUMENT fix.
#
# Default 0 (off): it HELPS long-document corpora (BEAM: accuracy 0.371 -> 0.426
# at radius 2) but HURTS conversational memory, where short turns don't split
# and the siblings are just context-rot noise (LoCoMo: 0.895 -> 0.814 at radius
# 1). Enable per-corpus via GRAPHKNOWS_NEIGHBOR_RADIUS.
_NEIGHBOR_RADIUS: int = int(os.environ.get("GRAPHKNOWS_NEIGHBOR_RADIUS", "0"))

# Candidate pool handed to fusion, as a multiple of the caller's top_k with a
# floor. Retrieve wide, return narrow: fusion can only rank what it was given,
# and asking each channel for exactly top_k made the pool the same size as the
# answer. Measured on conv-30 evidence_recall@25 without a reranker: pool=25
# gives 0.589, pool=100 gives 0.692.
_POOL_FACTOR: int = int(os.environ.get("GRAPHKNOWS_POOL_FACTOR", "4"))
_MIN_POOL: int = int(os.environ.get("GRAPHKNOWS_MIN_POOL", "100"))


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


class DETRetriever:
    """Deterministic three-collector hybrid retriever."""

    def __init__(
        self,
        store: GraphStore,
        embed_model: str = "",
        channels: list[Any] | None = None,
    ) -> None:
        self.store = store
        self.embed_model = embed_model
        # OPTIONAL channels only (frames, ontology) — the four baseline
        # collectors are methods on this class and always run. This spine fuses
        # + ranks whatever they return and is otherwise signal-blind.
        self._channels = channels or []

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

        # Collectors run concurrently; a failure in ANY of them raises loudly
        # (no return_exceptions swallow) so a broken signal surfaces immediately
        # rather than silently degrading recall. Each returns chunk_id -> (score,
        # info); the spine below is signal-blind (fuses + ranks only).
        # Over-fetch: they are asked for a POOL, not for the caller's page.
        # Fusion can only rank what it was handed, and asking each for exactly
        # top_k made the pool the same size as the answer, so a chunk no single
        # collector ranked in its own top-25 could never be recovered by
        # agreement between them. Measured on conv-30, evidence_recall@25:
        # pool=25 0.589, pool=100 0.692.
        pool_k = max(top_k * _POOL_FACTOR, _MIN_POOL)
        sources = await asyncio.gather(
            self.entity_boost(ctx.words_graph, session_id, ctx.state, pool_k),
            self.semantic_ann(ctx.emb, session_id, ctx.state, pool_k, emb2=ctx.emb2),
            self.bm25(ctx.words, session_id, ctx.state, pool_k),
            # words_graph so a "when did X happen" query — which states no date
            # for the date-keyed path to match — still finds dated chunks about X.
            self.temporal_traversal(query, session_id, ctx.state, pool_k, ctx.words_graph),
            *(c.collect(ctx, self, pool_k) for c in self._channels),
        )

        combined: dict[str, tuple[float, dict[str, Any]]] = {}
        for src in sources:
            for cid, (score, info) in src.items():
                _add_result(combined, cid, score, info)

        # The fused RRF score IS the ranking. A cross-encoder used to re-score
        # this pool and was measured WORSE at every depth a caller uses
        # (evidence_recall@25 0.567 against 0.692 without it) at 3.5x the wall
        # clock; see tests/retrieval/test_no_cross_encoder.py.
        fused = (await self._shape_and_rank(combined))[:top_k]
        fused = await self._expand_neighbors(fused)
        for i, item in enumerate(fused):
            item["rank"] = i + 1

        return RetrievalContext(
            query=query,
            session_id=session_id,
            fused_results=fused,
            search_type="DET",
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
            emb=emb_array[0].tolist(),
            emb2=emb2,
            words=[w for w in words_graph if len(w) > 3],
            words_graph=words_graph,
            state=_scope_to_state(scope),
        )

    async def _shape_and_rank(
        self, combined: dict[str, tuple[float, dict[str, Any]]]
    ) -> list[dict[str, Any]]:
        """Fetch missing metadata, build the row dicts, and sort by fused score.

        This sort IS the ranking — it decides which candidates survive the
        top-k cut. It was previously described as a mere tie-break ahead of a
        cross-encoder; that reranker measured worse than this order at every
        useful depth and is gone.
        """
        # Fetch the stored chunk for EVERY candidate. The old gate skipped any
        # candidate whose channel had already set a truthy ``metadata`` — but
        # ``get_chunks_by_ids`` returns no ``metadata`` key at all, so the gate
        # never saved the fetch it claimed to; it only starved those rows of
        # ``speaker``/``ts``, temporal hits worst of all.
        text_map = await self.store.get_chunks_by_ids(list(combined)) if combined else {}

        ordered = sorted(combined.items(), key=lambda kv: -kv[1][0])

        rows: list[dict[str, Any]] = []
        for rank_i, (cid, (base_score, info)) in enumerate(ordered):
            fc = text_map.get(cid, {})
            ents = info.get("entities", [])
            rows.append(
                {
                    "chunk_id": cid,
                    "doc_id": fc.get("doc_id") or info.get("doc_id", ""),
                    "text": fc.get("text") or info.get("text", ""),
                    "metadata": fc.get("metadata") or info.get("metadata") or {},
                    "speaker": fc.get("speaker") or info.get("speaker") or "",
                    "ts": fc.get("ts") or info.get("ts") or "",
                    "rrf_score": round(base_score, 6),
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
                _hit(ec, "entity"),
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
        with ``rank_bm25.BM25Okapi`` for one clean RRF source. There is no
        fallback scorer: a missing ``rank-bm25`` raises ``BrokenInstallError`` at
        import (see the top of this module). The TF-style fallback that used to
        live here is exactly how this channel spent months looking live while
        never running BM25.
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
            # Clamped at 0: BM25Okapi's IDF goes negative for terms appearing in
            # more than ~half the pool, and a negative addend would *demote* a
            # chunk below worse-ranked ones — inverting the tiebreaker instead of
            # breaking ties. Ties at 0 keep the RRF rank order intact. Could not
            # arise before `rank-bm25` was declared, since the fallback scorer's
            # IDF is always positive.
            rrf = rrf_score(rank, boost=3.0) + FUSION_ALPHA * max(0.0, bm25_score)
            result[cid] = (rrf, _hit(c, "text", entities=[]))
        return result

    async def temporal_traversal(
        self,
        query: str,
        session_id: str,
        state: str | None,
        top_k: int,
        words: list[str] | None = None,
    ) -> dict[str, tuple[float, dict[str, Any]]]:
        """Match chunks by canonical TEMPORAL node (MENTIONS_DATE).

        Two shapes, because a date-keyed lookup only serves one of them:

        - the query STATES a date ("what happened on 1 February 2023") — match
          chunks linked to that TEMPORAL node;
        - the query ASKS for one ("when did Gina lose her job") — it contains no
          date at all, so the date-keyed path returns nothing. Measured on
          conv-30, this was 26 of 26 temporal questions: the channel fired on
          2 of 81 queries, both of which happened to quote a literal date.
          Here the ENTITY is the key and the date is the answer, so we ask for
          date-bearing chunks about the query's entities instead.

        The second shape is gated on the query actually ASKING for a date. It
        used to run whenever the query merely lacked one, which is nearly every
        query: measured on conv-30, 78 of 79 date-free queries collected
        temporal votes while only 56 of 369 chunks carry a MENTIONS_DATE edge.
        Because RRF sums across channels, that handed 15% of the corpus a free
        extra vote on almost every query and let breadth beat relevance — 84.4%
        of the top-25 pool went to chunks in 4-5 channels while 38.7% of the
        gold evidence sat in only 2-3, pushing gold to ranks 42-94 under a
        top_k of 25.
        """
        result: dict[str, tuple[float, dict[str, Any]]] = {}
        candidates = query_date_candidates(query)
        if not candidates:
            if not asks_for_a_date(query):
                return result
            return await self._dated_chunks_for_query_entities(
                words or [], session_id, state, top_k
            )
        rows = await self.store.query_temporal_chunks(candidates, session_id or None, top_k, state)
        for rank, row in enumerate(rows):
            _add_result(
                result,
                row["chunk_id"],
                rrf_score(rank, RRF_RELATION_BOOST),
                _hit(row, "temporal", entities=[]),
            )
        return result

    async def _dated_chunks_for_query_entities(
        self,
        words: list[str],
        session_id: str,
        state: str | None,
        top_k: int,
    ) -> dict[str, tuple[float, dict[str, Any]]]:
        """Dated chunks about the query's entities — the "when did X?" fallback.

        Takes the same ``words_graph`` tokens the entity channel matches on, so
        the two channels agree on what the query is about and the saturation
        filtering applied upstream carries over.
        """
        result: dict[str, tuple[float, dict[str, Any]]] = {}
        if not words:
            return result
        rows = await self.store.query_dated_chunks_by_entities(
            words, session_id or None, top_k, state
        )
        for rank, row in enumerate(rows):
            _add_result(
                result,
                row["chunk_id"],
                rrf_score(rank, RRF_RELATION_BOOST),
                _hit(row, "temporal", entities=[], metadata={"date": row.get("date", "")}),
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
                _hit(row, "semantic", entities=[]),
            )
        return result


def _hit(row: dict[str, Any], source: str, **over: Any) -> dict[str, Any]:
    """The candidate payload for one chunk row, from ONE place.

    Every collector used to hand-build this dict and each picked a different
    subset of the row, so a field the store had already fetched was silently
    dropped on five of six paths. ``ts`` was the expensive one: the answerer
    could date a memory only when the date happened to be written inside the
    passage text — true of a concatenated transcript, false of a real
    conversation where the timestamp is a field. Turn-fed temporal accuracy was
    0.077 against 0.962 document-fed because of it.

    Carrying the row's own fields here means the next one added reaches the
    answerer without touching six call sites.
    """
    return {
        "text": row.get("text", ""),
        "entities": row.get("entities", []),
        "sources": {source},
        "doc_id": "",
        "metadata": {},
        "speaker": row.get("speaker", "") or "",
        "ts": row.get("ts", "") or "",
        **over,
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

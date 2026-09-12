"""Deterministic hybrid RAG: four baseline collectors, RRF fusion, one ranking.

The four collectors below (entity mentions, dense vector ANN, BM25, fact
traversal) run concurrently over one per-namespace ``GraphStore``, alongside
whatever optional channels ``channels.registry.core_channels`` supplies. Every
candidate is a SEGMENT reached through one of the store's declared reads;
the fact collector reaches its segments through the FACTs asserted in them.
The spine is signal-blind: it merges the candidates, sorts by fused score, and
cuts to top_k.

``SymbolRecall`` below is the top-down half of the traversal (FR-002): it reads
back the two edges ``IngestPipeline`` wrote bottom-up and answers in facts with
their evidence, under a stated budget.
"""

from __future__ import annotations

import asyncio
import logging
import re
from typing import Any

try:
    from rank_bm25 import BM25Okapi
except ModuleNotFoundError as exc:  # pragma: no cover - exercised only on broken installs
    from graphknows.exceptions import BrokenInstallError

    raise BrokenInstallError("The BM25 retrieval channel", "rank-bm25") from exc

from graphknows.bounds import check_top_k
from graphknows.channels.base import ChannelContext
from graphknows.models.fact import Fact, Modality, Polarity, Validity
from graphknows.models.report import (
    Counters,
    Evidence,
    FactWithEvidence,
    RecallBudget,
    RecallResult,
)
from graphknows.models.source import session_uri
from graphknows.ranking.rrf import (
    FUSION_ALPHA,
    RRF_RELATION_BOOST,
    rrf_score,
)
from graphknows.retrieval.models.context import RetrievalContext
from graphknows.settings import get_settings
from graphknows.storage.arcadedb._base import ArcadeStoreBase
from graphknows.storage.arcadedb._sql import not_forgotten, quoted_list
from graphknows.storage.arcadedb.graph_store import GraphStore, name_norm
from graphknows.storage.arcadedb.writers.fact import moment
from graphknows.storage.embedder import embed

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# BM25 in-memory scorer
# ---------------------------------------------------------------------------


def _bm25_rank(
    query_tokens: list[str],
    segments: list[dict[str, Any]],
) -> list[tuple[float, dict[str, Any]]]:
    """Rank *segments* by BM25 relevance to *query_tokens*.

    Uses ``rank_bm25.BM25Okapi``.

    Args:
        query_tokens: Pre-filtered, lower-cased query tokens.
        segments: Dicts that must contain at least ``"text"`` (str).

    Returns:
        List of ``(score, segment_dict)`` pairs sorted by score descending.
        Empty list when *query_tokens* or *segments* is empty.
    """
    if not query_tokens or not segments:
        return []

    tokenised = [re.findall(r"[a-z0-9][a-z0-9']*", (c.get("text") or "").lower()) for c in segments]

    bm25 = BM25Okapi(tokenised)
    scores = bm25.get_scores(query_tokens).tolist()

    # Keep segments that actually contain a query term, NOT segments with a
    # positive score. BM25Okapi's IDF is log((N - df + 0.5)/(df + 0.5)), which
    # goes NEGATIVE once a term appears in more than about half the corpus —
    # routine for the small candidate pools this is called with. A
    # `score > 0` filter therefore discards genuine matches, and could empty
    # the channel entirely.
    matched = [bool(set(query_tokens) & set(toks)) for toks in tokenised]
    paired = sorted(
        ((s, c) for s, c, hit in zip(scores, segments, matched, strict=False) if hit),
        key=lambda x: -x[0],
    )
    return paired


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
    Embedding the declarative form gets the ANN closer to how answers appear in segments.
    """
    q = question.strip().rstrip("?").strip()
    stripped = _Q_STRIP.sub("", q, count=1).strip()
    if stripped and stripped.lower() != q.lower() and len(stripped) >= 5:
        return stripped
    return None


# After cutting to top_k, append each result's ±N NEXT-adjacent segments. Long
# ingested messages split across consecutive segments, so the answer often
# sits in a sibling of the matched segment — a LONG-DOCUMENT fix.
#
# Default 0 (off): it HELPS long-document corpora (BEAM: accuracy 0.371 -> 0.426
# at radius 2) but HURTS conversational memory, where short turns don't split
# and the siblings are just context-rot noise (LoCoMo: 0.895 -> 0.814 at radius
# 1). Enable per-corpus via GRAPHKNOWS_NEIGHBOR_RADIUS.
_NEIGHBOR_RADIUS: int = get_settings().neighbor_radius

# Candidate pool handed to fusion, as a multiple of the caller's top_k with a
# floor. Retrieve wide, return narrow: fusion can only rank what it was given,
# and asking each channel for exactly top_k made the pool the same size as the
# answer. Measured on conv-30 evidence_recall@25 without a reranker: pool=25
# gives 0.589, pool=100 gives 0.692.
_POOL_FACTOR: int = get_settings().pool_factor
_MIN_POOL: int = get_settings().min_pool


def _max_facts() -> int:
    """Fact-sheet size fed to the generator (default 40, the store's own cap)."""
    return get_settings().max_facts


def _fact_context_enabled() -> bool:
    """Whether the D8 structured fact-sheet is fed to the generator (default on).

    Read at call time (not import time) so the eval A/B can flip
    ``GRAPHKNOWS_FACT_CONTEXT`` per MCP-server subprocess.
    """
    return get_settings().fact_context


class DETRetriever:
    """Deterministic four-collector hybrid retriever over SEGMENT and FACT."""

    def __init__(
        self,
        store: GraphStore,
        embed_model: str = "",
        channels: list[Any] | None = None,
    ) -> None:
        self.store = store
        self.embed_model = embed_model
        # OPTIONAL channels only (ontology) — the four baseline collectors are
        # methods on this class and always run. This spine fuses + ranks
        # whatever they return and is otherwise signal-blind.
        self._channels = channels or []

    async def retrieve(
        self,
        query: str,
        session_id: str,
        top_k: int = 10,
    ) -> RetrievalContext:
        """Run deterministic recall and return RRF-merged results.

        ``session_id`` scopes recall to the SOURCEs that session was ingested
        as; empty is namespace-wide. A session nothing was ingested under
        answers with no candidates rather than the whole namespace.
        """
        # Bounded here too (FR-030): a direct caller of DETRetriever bypasses
        # the Memory facade's own check, and top_k multiplies straight into
        # pool_k below and the store's k_pre = top_k * 20.
        check_top_k(top_k)
        log.debug("DET retrieve query=%r session=%s top_k=%d", query, session_id, top_k)

        ctx = await self._build_channel_context(query, session_id)

        # Collectors run concurrently; a failure in ANY of them raises loudly
        # (no return_exceptions swallow) so a broken signal surfaces immediately
        # rather than silently degrading recall. Each returns segment id ->
        # (score, info); the spine below is signal-blind (fuses + ranks only).
        # Over-fetch: they are asked for a POOL, not for the caller's page.
        # Fusion can only rank what it was handed, and asking each for exactly
        # top_k made the pool the same size as the answer, so a segment no
        # single collector ranked in its own top-25 could never be recovered by
        # agreement between them. Measured on conv-30, evidence_recall@25:
        # pool=25 0.589, pool=100 0.692.
        pool_k = max(top_k * _POOL_FACTOR, _MIN_POOL)
        sources = await asyncio.gather(
            self.entity_boost(ctx.words_graph, ctx.source_ids, pool_k),
            self.semantic_ann(ctx.emb, ctx.source_ids, pool_k, emb2=ctx.emb2),
            self.bm25(ctx.words, ctx.source_ids, pool_k),
            self.fact_traversal(ctx.words_graph, ctx.source_ids, pool_k),
            *(c.collect(ctx, self, pool_k) for c in self._channels),
        )

        combined: dict[str, tuple[float, dict[str, Any]]] = {}
        for src in sources:
            for sid, (score, info) in src.items():
                _add_result(combined, sid, score, info)

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
            facts=await self._fact_sheet(ctx),
        )

    async def _expand_neighbors(self, fused: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Merge each result's NEXT-adjacent segments INTO its own text.

        Ingested long messages split across consecutive segments, so the
        answer often lives in a sibling of the segment that matched the query.
        Each result's ±_NEIGHBOR_RADIUS siblings are stitched into that
        result's text in byte order, rather than added as separate passages —
        the passage COUNT is unchanged, so the downstream ``[:gen_context_k]``
        cut still keeps the same number of distinct seeds instead of spending
        its budget on the top seeds' siblings. Measured on BEAM's real evidence
        labels, over the 25 passages the generator actually sees, merging
        lifts evidence recall 0.305 -> 0.547 where the old append-then-cut
        dropped it to 0.305 (below the 0.353 no-expansion baseline). Each
        sibling is stitched into only the first seed that claims it, so shared
        siblings are not duplicated. Radius 0 disables it.
        """
        if _NEIGHBOR_RADIUS < 1 or not fused:
            return fused
        ids = [str(f.get("chunk_id") or "") for f in fused]
        try:
            neighbors = await self.store.neighbor_segments(ids, _NEIGHBOR_RADIUS)
        except Exception as exc:
            log.warning("neighbor expansion unavailable (%s); skipping", exc)
            return fused
        if not neighbors:
            return fused
        claimed = set(ids)
        for item in fused:
            sid = str(item.get("chunk_id") or "")
            sibs = [s for s in neighbors.get(sid, []) if str(s.get("id") or "") not in claimed]
            if not sibs:
                continue
            claimed.update(str(s.get("id") or "") for s in sibs)
            # Seed first (the query-relevant segment), then its siblings in
            # byte order — the matched text leads, its split-message context
            # follows.
            sibs.sort(key=lambda s: s.get("byte_start", 0))
            parts = [str(item.get("text") or "")] + [s.get("text", "") for s in sibs]
            item["text"] = " ".join(p for p in parts if p)
            item.setdefault("metadata", {})["merged_neighbors"] = len(sibs)
        return fused

    async def _build_channel_context(self, query: str, session_id: str) -> ChannelContext:
        """Embed the query (plus its declarative variant), tokenise it, resolve its scope."""
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
        # A session is a SOURCE (possibly several, one per re-ingest of the
        # grown transcript), so the scope is the ids segments carry.
        source_ids: list[str] | None = None
        if session_id:
            rows = await self.store.sources(uri=session_uri(session_id))
            source_ids = [str(r["id"]) for r in rows if r.get("id")]
        return ChannelContext(
            query=query,
            session_id=session_id,
            emb=emb_array[0].tolist(),
            emb2=emb2,
            words=[w for w in words_graph if len(w) > 3],
            words_graph=words_graph,
            source_ids=source_ids,
        )

    async def _shape_and_rank(
        self, combined: dict[str, tuple[float, dict[str, Any]]]
    ) -> list[dict[str, Any]]:
        """Fetch missing metadata, build the row dicts, and sort by fused score.

        This sort IS the ranking — it decides which candidates survive the
        top-k cut. It was previously described as a mere tie-break ahead of a
        cross-encoder; that reranker measured worse than this order at every
        useful depth and is gone.

        The row keys are the ``Hit`` shape every consumer reads: ``chunk_id``
        carries the segment id, ``doc_id`` its source, ``speaker`` its role and
        ``ts`` when it was observed.
        """
        # Fetch the stored segment for EVERY candidate, so a channel that
        # returned only an id still hands the answerer the role and time it
        # needs to attribute and date a memory.
        text_map = await self.store.get_segments_by_ids(list(combined)) if combined else {}

        ordered = sorted(combined.items(), key=lambda kv: -kv[1][0])

        rows: list[dict[str, Any]] = []
        for rank_i, (sid, (base_score, info)) in enumerate(ordered):
            fc = text_map.get(sid, {})
            ents = info.get("entities", [])
            rows.append(
                {
                    "chunk_id": sid,
                    "doc_id": fc.get("source_id") or info.get("doc_id", ""),
                    "text": fc.get("text") or info.get("text", ""),
                    "metadata": fc.get("metadata") or info.get("metadata") or {},
                    "speaker": fc.get("role") or info.get("speaker") or "",
                    "ts": fc.get("observed_at") or info.get("ts") or "",
                    "rrf_score": round(base_score, 6),
                    "sources": ",".join(sorted(info["sources"])),
                    "rank": rank_i + 1,
                    "entities": ents if isinstance(ents, list) else list(ents),
                }
            )
        return rows

    async def _fact_sheet(self, ctx: ChannelContext) -> list[str]:
        """Structured fact-sheet (D8) for the query's resolved entities.

        The current facts with a resolved entity at either end, one line each,
        capped at ``max_facts``. Soft-fails — a fact-context error never
        degrades recall.
        """
        if not _fact_context_enabled():
            return []
        try:
            resolved = await ctx.resolve_entities(self.store)
            if not resolved:
                return []
            ids = [str(e["id"]) for e in resolved]
            rows = await self.store.facts_by_entity(ids, limit=_max_facts())
        except Exception as exc:  # pragma: no cover - defensive soft-fail
            log.debug("fact context failed (soft): %s", exc)
            return []
        return list(dict.fromkeys(line for row in rows if (line := _fact_line(row))))

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
        source_ids: list[str] | None,
        top_k: int,
    ) -> dict[str, tuple[float, dict[str, Any]]]:
        """Collector: segments reachable from query-token entity nodes (MENTIONS)."""
        result: dict[str, tuple[float, dict[str, Any]]] = {}
        if not words:
            return result
        rows = await self.store.segments_mentioning(words, top_k, source_ids)
        for rank, row in enumerate(rows):
            _add_result(
                result,
                row["id"],
                # boost=2: entity hits are high-precision; outcompete pure-ANN noise.
                rrf_score(rank, boost=2.0),
                _hit(row, "entity"),
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
        source_ids: list[str] | None,
        top_k: int,
    ) -> dict[str, tuple[float, dict[str, Any]]]:
        """BM25 sparse channel over segments.

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

        pool = await self.store.full_text_search_segments(specific, top_k * 3, source_ids)
        ranked = _bm25_rank(specific, pool)
        for rank, (bm25_score, row) in enumerate(ranked[:top_k]):
            # BM25 score folds in as a tiebreaker on top of RRF rank weight.
            # Clamped at 0: BM25Okapi's IDF goes negative for terms appearing in
            # more than ~half the pool, and a negative addend would *demote* a
            # segment below worse-ranked ones — inverting the tiebreaker instead
            # of breaking ties. Ties at 0 keep the RRF rank order intact.
            rrf = rrf_score(rank, boost=3.0) + FUSION_ALPHA * max(0.0, bm25_score)
            result[row["id"]] = (rrf, _hit(row, "text", entities=[]))
        return result

    async def fact_traversal(
        self,
        words: list[str],
        source_ids: list[str] | None,
        top_k: int,
    ) -> dict[str, tuple[float, dict[str, Any]]]:
        """Collector: the evidence segments of facts USING a predicate the query names.

        The FACT plane's own route into the pool: a query word that names a
        relation ("works", "lives") reaches the segments asserting facts under
        that predicate, whether or not the entity is named. Rows arrive in
        confidence order, so row position is the rank.
        """
        result: dict[str, tuple[float, dict[str, Any]]] = {}
        if not words:
            return result
        rows = await self.store.facts_by_predicate(words, top_k, source_ids)
        for rank, row in enumerate(rows):
            _add_result(
                result,
                row["segment_id"],
                rrf_score(rank, RRF_RELATION_BOOST),
                _hit(row, "fact", entities=[]),
            )
        return result

    async def semantic_ann(
        self,
        emb: list[float],
        source_ids: list[str] | None,
        top_k: int,
        emb2: list[float] | None = None,
    ) -> dict[str, tuple[float, dict[str, Any]]]:
        """Collector: dense vector ANN over segment embeddings."""
        result: dict[str, tuple[float, dict[str, Any]]] = {}
        k = top_k * 2

        coros = [self.store.search_segments_ann(emb, k, source_ids)]
        if emb2 is not None:
            coros.append(self.store.search_segments_ann(emb2, k, source_ids))

        all_rows = await asyncio.gather(*coros)  # failures raise loudly

        # First pass: best (minimum) rank per segment across the original query
        # and the declarative variant, so a segment surfaced by both embeddings
        # does not receive 2x RRF mass from a single logical 'semantic' source.
        best_rank: dict[str, int] = {}
        best_row: dict[str, dict] = {}
        for rows in all_rows:
            for rank, row in enumerate(rows):
                sid = row["id"]
                if sid not in best_rank or rank < best_rank[sid]:
                    best_rank[sid] = rank
                    best_row[sid] = row

        # Second pass: emit ONE RRF contribution per segment for the semantic source.
        for sid, rank in best_rank.items():
            row = best_row[sid]
            cosine = float(row.get("cosine_score", 0.0) or 0.0)
            score = rrf_score(rank) + FUSION_ALPHA * cosine
            _add_result(
                result,
                sid,
                score,
                _hit(row, "semantic", entities=[]),
            )
        return result


def _hit(row: dict[str, Any], source: str, **over: Any) -> dict[str, Any]:
    """The candidate payload for one segment row, from ONE place.

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
        "doc_id": row.get("source_id", "") or "",
        "metadata": {},
        "speaker": row.get("role", "") or "",
        "ts": row.get("observed_at", "") or "",
        **over,
    }


def _fact_line(row: dict[str, Any]) -> str:
    """One fact row as the ``"subject predicate object"`` line the generator reads.

    The predicate id's local part, underscores to spaces, lower-cased —
    ``p:works_at`` reads as "works at".
    """
    subject = str(row.get("subject_name") or "").strip()
    predicate = str(row.get("predicate") or "").strip()
    obj = str(row.get("object_name") or "").strip()
    if not subject or not predicate or not obj:
        return ""
    return f"{subject} {predicate.rsplit(':', 1)[-1].replace('_', ' ').lower()} {obj}"


def _add_result(
    result: dict[str, tuple[float, dict[str, Any]]],
    segment_id: str,
    score: float,
    info: dict[str, Any],
) -> None:
    """Merge a candidate into *result*: sum scores, union source labels."""
    if segment_id in result:
        old_score, old_info = result[segment_id]
        result[segment_id] = (
            old_score + score,
            {**old_info, "sources": old_info["sources"] | info["sources"]},
        )
    else:
        result[segment_id] = (score, info)


# ---------------------------------------------------------------------------
# Top-down activation (FR-002)
# ---------------------------------------------------------------------------

# What a fact needs to come back with the segment it was read from: the
# assertion, and the evidence's place in its source. Written once because both
# activation queries answer in the same shape.
_FACT_ROW = (
    "f.id AS id, f.subject AS subject, f.predicate AS predicate, f.object AS object, "
    "f.polarity AS polarity, f.modality AS modality, f.confidence AS confidence, "
    "f.valid_from AS valid_from, f.valid_to AS valid_to, "
    "g.text AS text, g.byte_start AS byte_start, g.byte_end AS byte_end, s.uri AS source_uri"
)

_ORDERING = "fact confidence"
"""The stated ordering the budget cuts along, reported as ``truncated_by``."""


class SymbolRecall:
    """Symbol-keyed recall: an activated symbol pulls its facts back (FR-002).

    The mirror of :class:`~graphknows.ingestion.pipeline.IngestPipeline`'s
    bottom-up labelling. The query resolves to concepts, and activation reads
    exactly the two edges labelling wrote — ``SEGMENT-[EVOKES]->CONCEPT`` and
    ``ENTITY-[INSTANCE_OF]->CONCEPT`` — for its candidates. What comes back is
    facts with the evidence asserting them, never segments (FR-009) and never a
    fact without its source (SC-002), cut to a budget the caller stated (FR-010).

    Recall is also where deferred identity is settled: a recalled entity's
    ``SAME_AS_CANDIDATE`` edges commit to one entity, and the commitment is
    counted in the result (FR-018).
    """

    def __init__(self, store: ArcadeStoreBase) -> None:
        self._store = store

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._store.database!r})"

    async def recall(self, query: str, budget: RecallBudget | None = None) -> RecallResult:
        """The facts *query*'s symbols activate, with their evidence and counters.

        A query resolving to no symbol, or to symbols nothing hangs off, answers
        ``no_evidence=True`` — "nothing is known", which a caller can tell apart
        from a lookup that failed (FR-015).
        """
        bound = budget or RecallBudget()
        terms = _terms(query)
        names = _identifiers(query)
        resolved = await self._resolve(terms)
        uris = sorted(set(resolved.values()))
        named = await self._facts_of_named(names) if names else []
        matched = {str(row["name_norm"]) for row in named}
        rows = [*(await self._activate(uris) if uris else []), *named]
        committed = await self._commit_candidates(_entity_ids(rows))
        pool = _facts_with_evidence(_unified(rows, committed), bound.max_evidence_per_fact)
        pool.sort(key=lambda item: -item.fact.confidence)
        kept = pool[: bound.max_facts]
        return RecallResult(
            facts=kept,
            no_evidence=not kept,
            budget=bound,
            truncated_by=_ORDERING if len(kept) < len(pool) else None,
            counters=Counters(
                symbols_resolved=len(uris) + len(matched),
                symbols_unresolved=len(set(terms) - set(resolved)) + len(set(names) - matched),
                facts_returned=len(kept),
                facts_truncated_by_budget=len(pool) - len(kept),
                merges_committed=len(committed),
            ),
        )

    async def _resolve(self, terms: list[str]) -> dict[str, str]:
        """The concepts *terms* name, as ``{matched label: concept uri}``.

        Matched on the casefolded label, the key bottom-up labelling indexes its
        concepts under, so both directions agree on what a symbol answers to.
        The vocabulary is only what the loaded packs wrote, so it is read whole
        and matched here rather than by a dialect-specific predicate.
        """
        rows = await self._store.query("MATCH (c:CONCEPT) RETURN c.uri AS uri, c.label AS label")
        wanted = set(terms)
        resolved: dict[str, str] = {}
        for row in rows:
            label = str(row["label"]).casefold()
            if label in wanted:
                resolved[label] = str(row["uri"])
        return resolved

    async def _commit_candidates(self, entity_ids: set[str]) -> dict[str, str]:
        """Commit the ``SAME_AS_CANDIDATE`` edges of the recalled entities (FR-018).

        Ingest defers weak identity evidence to recall; this is where it is
        answered. Returns ``{recalled id: the one entity its candidates commit
        to}``, which is what makes the candidate plane read as well as written.
        """
        if not entity_ids:
            return {}
        rows = await self._store.query(
            "MATCH (e:ENTITY)-[c:SAME_AS_CANDIDATE]->(o:ENTITY) "
            f"WHERE e.id IN {quoted_list(sorted(entity_ids))} "
            f"AND {not_forgotten('e')} AND {not_forgotten('o')} "
            "RETURN e.id AS source, o.id AS target, c.at AS at"
        )
        return _commitment(rows)

    async def _activate(self, uris: list[str]) -> list[dict[str, Any]]:
        """Every fact the activated concepts reach, by either labelling edge."""
        evoked, instantiated = await asyncio.gather(
            self._facts_evoked(uris), self._facts_instantiated(uris)
        )
        return [*evoked, *instantiated]

    async def _facts_evoked(self, uris: list[str]) -> list[dict[str, Any]]:
        """Facts asserted in a segment that EVOKES one of the activated concepts."""
        return await self._store.query(
            "MATCH (g:SEGMENT)-[:EVOKES]->(c:CONCEPT) "
            f"WHERE c.uri IN {quoted_list(uris)} "
            "MATCH (f:FACT)-[:ASSERTED_IN]->(g), (g)-[:PART_OF]->(s:SOURCE) "
            f"WHERE {not_forgotten('f')} AND f.is_current = true "
            f"RETURN {_FACT_ROW}"
        )

    async def _facts_instantiated(self, uris: list[str]) -> list[dict[str, Any]]:
        """Facts whose subject entity is an INSTANCE_OF one of those concepts."""
        return await self._store.query(
            "MATCH (e:ENTITY)-[:INSTANCE_OF]->(c:CONCEPT) "
            f"WHERE c.uri IN {quoted_list(uris)} AND {not_forgotten('e')} "
            "MATCH (e)-[:SUBJECT_OF]->(f:FACT)-[:ASSERTED_IN]->(g:SEGMENT), "
            "(g)-[:PART_OF]->(s:SOURCE) "
            f"WHERE {not_forgotten('f')} AND f.is_current = true "
            f"RETURN {_FACT_ROW}"
        )

    async def _facts_of_named(self, names: list[str]) -> list[dict[str, Any]]:
        """Facts the entities *names* name are the subject of (FR-032, FR-033).

        The other half of resolution: a query term can name an entity outright
        ("graphknows/memory.py") rather than a concept, and the entity's own
        assertions are what it should pull back. The matched key rides along so
        the counters can say which names resolved.
        """
        return await self._store.query(
            "MATCH (e:ENTITY) "
            f"WHERE e.name_norm IN {quoted_list(names)} AND {not_forgotten('e')} "
            "MATCH (e)-[:SUBJECT_OF]->(f:FACT)-[:ASSERTED_IN]->(g:SEGMENT), "
            "(g)-[:PART_OF]->(s:SOURCE) "
            f"WHERE {not_forgotten('f')} AND f.is_current = true "
            f"RETURN {_FACT_ROW}, e.name_norm AS name_norm"
        )


def _terms(query: str) -> list[str]:
    """The casefolded word tokens a symbol can answer to."""
    return re.findall(r"[a-z0-9][a-z0-9']*", query.casefold())


_IDENTIFIER_SHAPED = re.compile(r"\w[_./]\w")
"""An identifier is a term glued by an inner ``_``, ``.`` or ``/``."""


def _identifiers(query: str) -> list[str]:
    """The identifier-shaped terms of *query*, whole, as ``ENTITY.name_norm`` keys.

    ``graphknows/memory.py`` names one entity, not four words: word tokenising
    it loses the only symbol it could resolve to. Normalised the way ingest
    normalises an entity name, so both directions agree on the key (FR-033).
    """
    tokens = (token.strip(".,;:!?()[]\"'`") for token in query.split())
    return sorted({name_norm(token) for token in tokens if _IDENTIFIER_SHAPED.search(token)})


def _entity_ids(rows: list[dict[str, Any]]) -> set[str]:
    """The entities the activated facts stand between."""
    return {str(row[side]) for row in rows for side in ("subject", "object") if row.get(side)}


def _commitment(rows: list[dict[str, Any]]) -> dict[str, str]:
    """Fold candidate edges into one committed entity per chain, oldest edge first.

    The edge's direction is the one identity resolution decided when it wrote
    the candidate, so a chain is followed to the end rather than re-decided.
    """
    proposed = {
        str(row["source"]): str(row["target"])
        for row in sorted(rows, key=lambda row: str(row.get("at") or ""))
        if row.get("source") and row.get("target")
    }
    return {source: _committed_to(proposed, source) for source in proposed}


def _committed_to(proposed: dict[str, str], source: str) -> str:
    """The end of *source*'s candidate chain: the id no candidate edge leaves."""
    seen = source
    for _ in range(len(proposed)):
        if (further := proposed.get(seen)) is None:
            break
        seen = further
    return seen


def _unified(rows: list[dict[str, Any]], committed: dict[str, str]) -> list[dict[str, Any]]:
    """*rows* with every committed entity id replaced by the id it committed to."""
    if not committed:
        return rows
    return [
        {
            **row,
            **{
                side: committed[value]
                for side in ("subject", "object")
                if (value := str(row.get(side) or "")) in committed
            },
        }
        for row in rows
    ]


def _facts_with_evidence(rows: list[dict[str, Any]], max_evidence: int) -> list[FactWithEvidence]:
    """Group activation rows by fact, keeping at most *max_evidence* segments each.

    Both labelling edges can reach one fact through the same segment, so
    evidence is deduplicated before the per-fact ceiling applies — otherwise a
    ceiling of one would spend itself on a repeat.
    """
    facts: dict[str, Fact] = {}
    evidence: dict[str, list[Evidence]] = {}
    for row in rows:
        fact_id = str(row["id"])
        facts.setdefault(fact_id, _fact(row))
        cited = evidence.setdefault(fact_id, [])
        item = _evidence(row)
        if item not in cited and len(cited) < max_evidence:
            cited.append(item)
    return [
        FactWithEvidence(fact=facts[fact_id], evidence=cited)
        for fact_id, cited in evidence.items()
        if cited
    ]


def _fact(row: dict[str, Any]) -> Fact:
    """The assertion one activation row carries."""
    return Fact(
        id=str(row["id"]),
        subject=str(row["subject"]),
        predicate=str(row["predicate"]),
        object=str(row["object"]),
        polarity=Polarity(str(row.get("polarity") or Polarity.ASSERTED)),
        modality=Modality(str(row["modality"])) if row.get("modality") else None,
        confidence=float(row.get("confidence") or 0.0),
        validity=Validity(
            valid_from=moment(row.get("valid_from")), valid_to=moment(row.get("valid_to"))
        ),
    )


def _evidence(row: dict[str, Any]) -> Evidence:
    """Where that row's fact was read: the source, the bytes, and the text."""
    return Evidence(
        source_uri=str(row.get("source_uri") or ""),
        byte_range=(int(row.get("byte_start") or 0), int(row.get("byte_end") or 0)),
        text=str(row.get("text") or ""),
    )

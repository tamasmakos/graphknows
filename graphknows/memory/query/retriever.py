"""Deterministic entity-traversal + vector hybrid RAG with RRF fusion.

Three collectors run concurrently:
  A. STM structural recall from ArcadeDB STM.
  B. LTM structural recall from ArcadeDB LTM.
  C. Semantic recall from ArcadeDB vector store.

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
import re
from typing import Any

from graphknows.ports.vector_store import IVectorStore
from graphknows.memory.query._rrf_common import (
    FUSION_ALPHA,
    RRF_RELATION_BOOST,
    idf_weight,
    pagerank_boost,
    rrf_score,
)
from graphknows.memory.query._rrf_common import shape_score as _shape_score
from graphknows.memory.query.base import BaseGraphRetriever
from graphknows.memory.channels.base import ChannelContext
from graphknows.memory.query.context import RetrievalContext
from graphknows.memory.query.rerank import rerank as _ce_rerank
from graphknows.memory.stores.arcadedb.stm_store import ArcadeDBSTMStore
from graphknows.memory.stores.embedder import embed
from graphknows.memory.temporal import filter_by_validity_window, query_date_candidates

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


_STRUCTURAL_REL_TYPES = frozenset(
    {
        "MENTIONS",
        "HAS_CHUNK",
        "IN_SESSION",
        "IN_TOPIC",
        "IN_COMMUNITY",
        "HAS_TURN",
        "IN_DIALOG",
        "PART_OF",
    }
)


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
_ENTITY_SATURATION_RATIO: float = 0.5

class DETRetriever(BaseGraphRetriever):
    """Deterministic three-collector hybrid retriever."""

    def __init__(
        self,
        vec: IVectorStore,
        ltm: Any,
        stm: ArcadeDBSTMStore,
        embed_model: str = "",
        channels: "list[Any] | None" = None,
    ) -> None:
        self.vec = vec
        self.ltm = ltm
        self.stm = stm
        self.embed_model = embed_model
        # Channels provide ranked candidates; this spine fuses/reranks them and is
        # channel-blind. Defaults to the seven built-in collectors (behaviour
        # identical to the original hardcoded gather); callers pass a richer set
        # (e.g. + frames) built from settings via config.builders.channels.
        if channels is None:
            from graphknows.memory.channels.collectors import DEFAULT_COLLECTOR_CHANNELS
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
        text_map = await self.vec.get_chunks_by_ids(needs_meta) if needs_meta else {}

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
        """Execute a parameterized read-only Cypher query against the LTM store."""
        if re.search(r"\{[^}]+\}", cypher_template):
            raise ValueError(
                "Cypher template must not contain {var} interpolation; "
                "use $param placeholders and pass values via params"
            )
        forbidden = re.compile(r"\b(CREATE|MERGE|SET|DELETE|DETACH|REMOVE|DROP)\b", re.IGNORECASE)
        if forbidden.search(cypher_template):
            raise ValueError("Only read-only Cypher is permitted")
        rows = await self.ltm._qry(cypher_template, **(params or {}))
        return [dict(r) for r in rows]

    async def stm_entity_boost(
        self,
        words: list[str],
        session_id: str,
        top_k: int,
    ) -> dict[str, tuple[float, dict[str, Any]]]:
        """Collector: boost STM chunks reachable from query-token entity nodes."""
        result: dict[str, tuple[float, dict[str, Any]]] = {}
        if not words:
            return result
        entity_chunks = await self.stm.query_entity_chunks(
            session_id=session_id,
            entity_names=words,
            top_k=top_k,
        )
        for rank, ec in enumerate(entity_chunks):
            cid = ec["chunk_id"]
            score = rrf_score(rank)
            info = {
                "text": ec.get("text", ""),
                "entities": ec.get("entity_names", []),
                "sources": {"stm_graph"},
                "doc_id": ec.get("file_id", ""),
                "metadata": ec.get("metadata", {}),
            }
            result[cid] = (result.get(cid, (0.0, info))[0] + score, info)
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

    async def stm_bm25(
        self,
        words: list[str],
        session_id: str,
        top_k: int,
    ) -> dict[str, tuple[float, dict[str, Any]]]:
        """BM25 sparse channel over STM chunks — replaces homegrown two-pass full-text.

        Fetches candidate chunks via the OR pass (any term) from the STM store
        to build a pool, then re-ranks them with BM25 for one clean RRF source.
        Uses ``rank_bm25.BM25Okapi`` when importable; falls back to a normalised
        TF-IDF scorer so the channel is always live.
        """
        result: dict[str, tuple[float, dict[str, Any]]] = {}
        specific = [w for w in words if w not in self._FT_STOPWORDS]
        if not specific:
            return result

        # Fetch a broad candidate pool via the OR pass.
        or_chunks = await self.stm.full_text_search_chunks(
            session_id=session_id,
            terms=specific,
            top_k=top_k * 3,
            require_all=False,
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
                    "sources": {"stm_text"},
                    "doc_id": "",
                    "metadata": {},
                },
            )
        return result

    async def _discriminative_entities(
        self, safe_words: list[str], session_filter: str | None
    ) -> list[tuple[str, float]]:
        """Query-matched entity names with IDF discriminativeness weights.

        Returns ``(name, idf_weight)`` pairs where:
        - Entities in > ``_ENTITY_SATURATION_RATIO`` of the corpus are still
          hard-dropped (traversal seed noise), same as before.
        - Entities with doc_freq / corpus_size in (0, saturation] get a smooth
          IDF weight in (0, 1] from ``idf_weight()``.  Rare entities → 1.0.
          This weight is multiplied into the RRF score at point of insertion
          so ubiquitous-but-sub-saturation entities receive a proportional
          down-weight rather than a cliff.

        Returns ``[]`` when nothing discriminative matched.
        """
        if not safe_words:
            return []
        # Short tokens (< 5 chars) use exact match to avoid false substring hits
        # (e.g. "jan" matching "January", "bob" matching "nobody").
        where_entity = " OR ".join(
            f"toLower(e.name) CONTAINS '{w}'" if len(w) >= 5 else f"toLower(e.name) = '{w}'"
            for w in safe_words
        )
        total_rows = await self.ltm._qry("MATCH (c:CHUNK) RETURN count(c) AS n")
        total = int(total_rows[0]["n"]) if total_rows else 0
        if total <= 0:
            return []
        deg_rows = await self.ltm._qry(
            f"MATCH (e:ENTITY) WHERE {where_entity} "
            f"MATCH (e)<-[:MENTIONS]-(c:CHUNK) "
            f"RETURN e.name AS name, count(DISTINCT c) AS deg"
        )
        cap = max(1, int(_ENTITY_SATURATION_RATIO * total))
        result: list[tuple[str, float]] = []
        for r in deg_rows:
            name = r.get("name")
            deg = int(r.get("deg") or 0)
            if not name or deg > cap:
                continue  # Hard-drop saturated entities (same behaviour as before)
            w = idf_weight(deg, total)
            result.append((name, w))
        return result

    async def ltm_entity_traversal(
        self,
        words: list[str],
        session_id: str,
        top_k: int,
    ) -> dict[str, tuple[float, dict[str, Any]]]:
        """Collector: traverse the LTM entity graph from the query's entities."""
        if not words:
            return {}
        return await self._run_entity_traversal_async(words, session_id or None, top_k)

    async def _run_entity_traversal_async(
        self,
        words: list[str],
        session_filter: str | None,
        top_k: int,
    ) -> dict[str, tuple[float, dict[str, Any]]]:
        """Entity-graph traversal against LTM, seeded on discriminative entities.

        Uses dynamic OR conditions in place of the any() list predicate —
        ArcadeDB's openCypher does not support any(x IN $list WHERE ...) syntax.
        """
        result: dict[str, tuple[float, dict[str, Any]]] = {}
        if not words:
            return result
        safe = [w.replace("'", "") for w in words]
        disc = await self._discriminative_entities(safe, session_filter)
        if not disc:
            return result
        # disc is now list[tuple[name, idf_w]] — build lookup and name list.
        idf_lookup: dict[str, float] = {n.replace("'", ""): w for n, w in disc}
        safe_names = list(idf_lookup.keys())
        # Exact-match the (already query-matched, non-saturated) entity names.
        where_entity = " OR ".join(f"e.name = '{n}'" for n in safe_names)
        if session_filter:
            cypher = (
                f"MATCH (sess:SESSION {{id: $sid}})-[:IN_SESSION]->(f:FILE)"
                f"-[:HAS_CHUNK]->(c:CHUNK) "
                f"MATCH (c)-[:MENTIONS]->(e:ENTITY) "
                f"WHERE {where_entity} "
                f"WITH c, collect(DISTINCT e.name) AS entities, count(DISTINCT e) AS mc, "
                f"     max(coalesce(e.pagerank, 0.0)) AS max_pr "
                f"ORDER BY mc DESC LIMIT $top_k "
                f"RETURN c.id AS chunk_id, c.text AS text, entities, max_pr"
            )
            rows = await self.ltm._qry(cypher, top_k=top_k, sid=session_filter)
        else:
            cypher = (
                f"MATCH (e:ENTITY) WHERE {where_entity} "
                f"MATCH (c:CHUNK)-[:MENTIONS]->(e) "
                f"WITH c, collect(DISTINCT e.name) AS entities, count(DISTINCT e) AS mc, "
                f"     max(coalesce(e.pagerank, 0.0)) AS max_pr "
                f"ORDER BY mc DESC LIMIT $top_k "
                f"RETURN c.id AS chunk_id, c.text AS text, entities, max_pr"
            )
            rows = await self.ltm._qry(cypher, top_k=top_k)

        for rank, row in enumerate(rows):
            # IDF smooth down-weight: multiply by the minimum IDF weight among
            # the matched entities in this chunk.  Rare entities → 1.0 (no change).
            chunk_ents = row.get("entities") or []
            min_idf = min(
                (idf_lookup.get(e, 1.0) for e in chunk_ents if e in idf_lookup),
                default=1.0,
            )
            # Expose max entity pagerank in metadata so the shaping stage can
            # apply pagerank_boost.  Absent in older graphs → defaults to 0.
            pr = float(row.get("max_pr") or 0.0)
            _add_result(
                result,
                row["chunk_id"],
                # boost=2: entity hits are high-precision; outcompete pure-ANN noise
                rrf_score(rank, boost=2.0) * min_idf,
                {
                    "text": row["text"],
                    "entities": row["entities"],
                    "sources": {"ltm_graph"},
                    "doc_id": "",
                    "metadata": {"pagerank": pr},
                },
            )

        await self._run_relation_expansion_async(result, safe_names, session_filter, top_k)
        return result

    async def _run_relation_expansion_async(
        self,
        result: dict[str, tuple[float, dict[str, Any]]],
        seed_names: list[str],
        session_filter: str | None,
        top_k: int,
    ) -> None:
        """2-hop relation expansion from the discriminative seed entities to
        neighbour chunks.

        Replaces any()/NONE() list predicates (unsupported in ArcadeDB Cypher)
        with dynamic OR WHERE conditions. Structural edge types are excluded via
        a NOT IN list embedded in the Cypher string.
        """
        where_entity = " OR ".join(f"e.name = '{n}'" for n in seed_names)
        # Embed skip-types as a literal list in Cypher rather than a $param list,
        # because ArcadeDB does not support IN $list for type() comparisons.
        skip_list = ", ".join(f"'{t}'" for t in sorted(_STRUCTURAL_REL_TYPES))
        not_in_skip = f"NOT type(rel) IN [{skip_list}]"
        if session_filter:
            rel_cypher = (
                f"MATCH (sess:SESSION {{id: $sid}})-[:IN_SESSION]->(f:FILE)"
                f"-[:HAS_CHUNK]->(c0:CHUNK)-[:MENTIONS]->(e:ENTITY) "
                f"WHERE {where_entity} "
                f"MATCH (e)-[rel]-(e2:ENTITY) "
                f"WHERE e2 <> e AND {not_in_skip} "
                f"MATCH (sess)-[:IN_SESSION]->(f2:FILE)-[:HAS_CHUNK]->(c:CHUNK)-[:MENTIONS]->(e2) "
                f"WITH c, collect(DISTINCT e2.name) AS ents, count(DISTINCT e2) AS mc "
                f"ORDER BY mc DESC LIMIT $top_k "
                f"RETURN c.id AS chunk_id, c.text AS text, ents AS entities"
            )
            rows = await self.ltm._qry(rel_cypher, top_k=top_k, sid=session_filter)
        else:
            rel_cypher = (
                f"MATCH (e:ENTITY) WHERE {where_entity} "
                f"MATCH (e)-[rel]-(e2:ENTITY) "
                f"WHERE e2 <> e AND {not_in_skip} "
                f"MATCH (c:CHUNK)-[:MENTIONS]->(e2) "
                f"WITH c, collect(DISTINCT e2.name) AS ents, count(DISTINCT e2) AS mc "
                f"ORDER BY mc DESC LIMIT $top_k "
                f"RETURN c.id AS chunk_id, c.text AS text, ents AS entities"
            )
            rows = await self.ltm._qry(rel_cypher, top_k=top_k)

        for rank, row in enumerate(rows):
            _add_result(
                result,
                row["chunk_id"],
                rrf_score(rank, RRF_RELATION_BOOST),
                {
                    "text": row["text"],
                    "entities": row["entities"],
                    "sources": {"ltm_graph"},
                    "doc_id": "",
                    "metadata": {},
                },
            )

    async def ltm_bm25(
        self,
        words: list[str],
        session_id: str,
        scope: str,
        top_k: int,
    ) -> dict[str, tuple[float, dict[str, Any]]]:
        """BM25 sparse channel over LTM chunks — replaces homegrown two-pass full-text.

        Mirrors ``stm_bm25`` for the LTM scope.  Fetches a broad OR-pool from
        the LTM store's full-text index, then re-ranks with BM25 for one clean
        RRF source.  No-op when scope excludes LTM or the store has no FT method.
        """
        result: dict[str, tuple[float, dict[str, Any]]] = {}
        if scope not in ("ltm", "both"):
            return result
        specific = [w for w in words if w not in self._FT_STOPWORDS]
        if not specific:
            return result

        neo_ft = getattr(self.ltm, "full_text_search_chunks", None)
        if not callable(neo_ft):
            return result

        or_chunks = await neo_ft(
            terms=specific,
            session_id=session_id or None,
            top_k=top_k * 3,
            require_all=False,
        )

        ranked = _bm25_rank(specific, or_chunks)
        for rank, (bm25_score, c) in enumerate(ranked[:top_k]):
            cid = c["chunk_id"]
            rrf = rrf_score(rank, boost=5.0) + FUSION_ALPHA * bm25_score
            result[cid] = (
                rrf,
                {
                    "text": c.get("text", ""),
                    "entities": [],
                    "sources": {"ltm_text"},
                    "doc_id": "",
                    "metadata": {},
                },
            )
        return result

    async def ltm_temporal_traversal(
        self,
        query: str,
        session_id: str,
        scope: str,
        top_k: int,
    ) -> dict[str, tuple[float, dict[str, Any]]]:
        """Match chunks and relation edges by bi-temporal validity window.

        Two passes:
        1. TEMPORAL-node pass — chunks linked to canonical TEMPORAL nodes via
           MENTIONS_DATE edges.  Activates only when the query contains a
           resolvable date.  Unchanged from the original implementation.
        2. Relation-edge validity pass — find entity chunks whose relation edges
           have a ``valid_at`` / ``invalid_at`` window overlapping the queried
           date.  Uses ``filter_by_validity_window`` from ``temporal.py`` to apply
           the as-of filter and attach a ``_temporal_score`` recency boost.

        Both passes are no-ops when no resolvable date is found in the query.
        """
        result: dict[str, tuple[float, dict[str, Any]]] = {}
        if scope not in ("ltm", "both"):
            return result
        candidates = query_date_candidates(query)
        if not candidates:
            return result

        # Dynamic WHERE: exact iso match, or year-prefix match for 4-digit years.
        safe = [c.replace("'", "") for c in candidates]
        conds = []
        for c in safe:
            conds.append(f"d.iso = '{c}'")
            if len(c) == 4 and c.isdigit():
                conds.append(f"d.iso STARTS WITH '{c}-'")
        where_iso = " OR ".join(conds)

        # Pass 1: canonical TEMPORAL node → chunk traversal (existing machinery).
        if session_id:
            cypher = (
                f"MATCH (sess:SESSION {{id: $sid}})-[:IN_SESSION]->(f:FILE)"
                f"-[:HAS_CHUNK]->(c:CHUNK)-[:MENTIONS_DATE]->(d:TEMPORAL) "
                f"WHERE {where_iso} "
                f"RETURN DISTINCT c.id AS chunk_id, c.text AS text LIMIT $top_k"
            )
            rows = await self.ltm._qry(cypher, top_k=top_k, sid=session_id)
        else:
            cypher = (
                f"MATCH (c:CHUNK)-[:MENTIONS_DATE]->(d:TEMPORAL) "
                f"WHERE {where_iso} "
                f"RETURN DISTINCT c.id AS chunk_id, c.text AS text LIMIT $top_k"
            )
            rows = await self.ltm._qry(cypher, top_k=top_k)
        for rank, row in enumerate(rows):
            _add_result(
                result,
                row["chunk_id"],
                rrf_score(rank, RRF_RELATION_BOOST),
                {
                    "text": row["text"],
                    "entities": [],
                    "sources": {"ltm_temporal"},
                    "doc_id": "",
                    "metadata": {},
                },
            )

        # Pass 2: relation-edge validity window — find chunks whose relation edges
        # have a valid_at within the queried date window.  Only activates when
        # relation edges carry bi-temporal metadata (WS-5 forward path).
        # Uses the first candidate date as the as-of anchor.
        as_of_iso = candidates[0] if candidates else None
        rel_rows = await self._ltm_relation_validity_pass(
            as_of_iso, session_id, top_k
        )
        for rank, row in enumerate(rel_rows):
            temporal_score: float = row.get("_temporal_score", 0.5)
            _add_result(
                result,
                row["chunk_id"],
                # Multiply RRF base by a small temporal recency boost (0–1).
                rrf_score(rank, RRF_RELATION_BOOST) * (1.0 + 0.5 * temporal_score),
                {
                    "text": row.get("text", ""),
                    "entities": [],
                    "sources": {"ltm_temporal_rel"},
                    "doc_id": "",
                    "metadata": {},
                },
            )

        return result

    async def _ltm_relation_validity_pass(
        self,
        as_of_iso: str | None,
        session_id: str,
        top_k: int,
    ) -> list[dict[str, Any]]:
        """Fetch relation edges with bi-temporal metadata and apply validity filter.

        Returns chunk dicts enriched with ``_temporal_score`` for use in the
        temporal traversal boost.  Soft-fails to an empty list when the LTM store
        does not yet have bi-temporal relation edges (graceful degradation).
        """
        if not as_of_iso:
            return []
        # Fetch relation edges that have valid_at set.
        if session_id:
            edge_rows = await self.ltm._qry(
                "MATCH (a:ENTITY)-[r]->(b:ENTITY) "
                "WHERE r.is_current IS NOT NULL AND r.valid_at IS NOT NULL "
                "MATCH (c:CHUNK)-[:MENTIONS]->(a) "
                "WHERE c.session_id = $sid "
                "RETURN c.id AS chunk_id, c.text AS text, "
                "       r.valid_at AS valid_at, r.invalid_at AS invalid_at, "
                "       r.created_at AS created_at "
                "LIMIT $top_k",
                sid=session_id,
                top_k=top_k * 2,
            )
        else:
            edge_rows = await self.ltm._qry(
                "MATCH (a:ENTITY)-[r]->(b:ENTITY) "
                "WHERE r.is_current IS NOT NULL AND r.valid_at IS NOT NULL "
                "MATCH (c:CHUNK)-[:MENTIONS]->(a) "
                "RETURN c.id AS chunk_id, c.text AS text, "
                "       r.valid_at AS valid_at, r.invalid_at AS invalid_at, "
                "       r.created_at AS created_at "
                "LIMIT $top_k",
                top_k=top_k * 2,
            )

        if not edge_rows:
            return []

        # Apply bi-temporal window filter from temporal.py.
        filtered = filter_by_validity_window(
            [dict(r) for r in edge_rows],
            as_of=as_of_iso,
            include_open=True,
        )
        return filtered[:top_k]

    async def ltm_topic_match(
        self,
        emb: list[float],
        session_id: str,
        scope: str,
        top_k: int,
    ) -> dict[str, tuple[float, dict[str, Any]]]:
        """Semantic topic match → inject the topic summary + member-entity chunks.

        Topics are pre-aggregated community summaries; matching one surfaces
        high-level multi-entity context (good for synthesis/multi-hop) plus the
        chunks of its member entities. No-op when no topics exist (e.g. llm-free
        without topic creation), so it never regresses the base pipeline.
        """
        result: dict[str, tuple[float, dict[str, Any]]] = {}
        if scope not in ("ltm", "both"):
            return result
        searcher = getattr(self.vec, "search_topics", None)
        if not callable(searcher):
            return result
        topics = await searcher(emb, k=2)
        for t_rank, topic in enumerate(topics):
            summary = (topic.get("summary") or "").strip()
            label = (topic.get("label") or "").strip()
            tid = topic.get("topic_id") or ""
            if summary:
                # Inject the topic summary itself as a synthetic passage.
                _add_result(
                    result,
                    f"topic:{tid}",
                    rrf_score(t_rank, boost=1.5),
                    {
                        "text": f"[Topic: {label}] {summary}" if label else summary,
                        "entities": [],
                        "sources": {"ltm_topic"},
                        "doc_id": "",
                        "metadata": {"is_topic_summary": True},
                    },
                )
            if not tid:
                continue
            rows = await self.ltm._qry(
                "MATCH (t:TOPIC {id: $tid})<-[:IN_TOPIC]-(e:ENTITY) "
                "MATCH (c:CHUNK)-[:MENTIONS]->(e) "
                "RETURN DISTINCT c.id AS chunk_id, c.text AS text LIMIT $k",
                tid=tid,
                k=top_k,
            )
            for rank, row in enumerate(rows):
                _add_result(
                    result,
                    row["chunk_id"],
                    rrf_score(rank, boost=1.0),
                    {
                        "text": row["text"],
                        "entities": [],
                        "sources": {"ltm_topic"},
                        "doc_id": "",
                        "metadata": {},
                    },
                )
        return result

    async def semantic_ann(
        self,
        emb: list[float],
        session_id: str,
        scope: str,
        top_k: int,
        emb2: list[float] | None = None,
    ) -> dict[str, tuple[float, dict[str, Any]]]:
        """Collector: dense vector ANN over chunk embeddings (STM + LTM)."""
        result: dict[str, tuple[float, dict[str, Any]]] = {}
        k = top_k * 2

        def _build_coros(
            embedding: list[float], tag: str
        ) -> tuple[list[Any], list[tuple[str, str]]]:
            coros: list[Any] = []
            tagged: list[tuple[str, str]] = []
            if scope in ("stm", "both") and session_id:
                coros.append(self.vec.search_stm_chunks(session_id, embedding, top_k=k))
                tagged.append(("stm_chunk", tag))
                coros.append(self.vec.search_stm_turns(session_id, embedding, top_k=k))
                tagged.append(("stm_turn", tag))
            if scope in ("ltm", "both"):
                # Pass session_id so the vector store applies session-scoped over-fetching
                # (k_pre = max(top_k * 20, 200)) and post-filters by session.  This prevents
                # cross-session contamination when multiple sessions coexist in LTM.
                ltm_sid = session_id or None
                coros.append(self.vec.search_ltm_chunks(embedding, top_k=k, session_id=ltm_sid))
                tagged.append(("ltm_chunk", tag))
                coros.append(self.vec.search_ltm_turns(embedding, top_k=k, session_id=ltm_sid))
                tagged.append(("ltm_turn", tag))
            return coros, tagged

        coros1, tagged1 = _build_coros(emb, "semantic")
        coros2, tagged2 = _build_coros(emb2, "semantic2") if emb2 is not None else ([], [])

        all_coros = coros1 + coros2
        all_tagged = tagged1 + tagged2

        if not all_coros:
            return result

        all_rows = await asyncio.gather(*all_coros)  # failures raise loudly

        # First pass: collect the best (minimum) rank per chunk across all ANN
        # passes (original query + declarative variant, stm/ltm sub-calls).
        # This prevents a chunk surfaced by both embeddings from receiving 2x
        # RRF mass from a single logical 'semantic' source.
        best_rank: dict[str, int] = {}
        best_row: dict[str, dict] = {}
        for (_label, _source_tag), rows in zip(all_tagged, all_rows, strict=False):
            for rank, row in enumerate(rows):
                cid = row["id"]
                if cid not in best_rank or rank < best_rank[cid]:
                    best_rank[cid] = rank
                    best_row[cid] = row

        # Second pass: emit ONE RRF contribution per chunk for the semantic source.
        for cid, rank in best_rank.items():
            row = best_row[cid]
            metadata = row.get("metadata", {}) or {}
            cosine = float(row.get("cosine_score", 0.0) or 0.0)
            score = rrf_score(rank) + FUSION_ALPHA * cosine
            _add_result(
                result,
                cid,
                score,
                {
                    "text": row.get("text", ""),
                    "entities": metadata.get("entity_names", []),
                    "sources": {"semantic"},
                    "doc_id": row.get("file_id", row.get("doc_id", "")),
                    "metadata": metadata,
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
    try:
        weight = float(metadata.get("community_weight", 1.0))
    except (TypeError, ValueError):
        weight = 1.0
    return max(1.0, min(weight, 3.0))

"""Deterministic entity-traversal + vector hybrid RAG with RRF fusion.

Three collectors run concurrently:
  A. STM structural recall from ArcadeDB STM.
  B. LTM structural recall from ArcadeDB LTM.
  C. Semantic recall from ArcadeDB vector store.

Topic nodes are LLM-assisted enrichment artifacts; deterministic retrieval uses
community metadata as score shaping rather than returning synthetic topic hits.
"""

from __future__ import annotations

import asyncio
import logging
import re
from typing import Any

from graphknows.config.adapters.ports.vector_db import IVectorDb
from graphknows.memory.query._rrf_common import FUSION_ALPHA, RRF_RELATION_BOOST, rrf_score
from graphknows.memory.query._rrf_common import shape_score as _shape_score
from graphknows.memory.query.base import BaseGraphRetriever
from graphknows.memory.query.context import RetrievalContext
from graphknows.memory.stores.arcadedb.stm_store import ArcadeDBSTMStore
from graphknows.memory.stores.embedder import embed
from graphknows.memory.temporal import query_date_candidates

log = logging.getLogger(__name__)

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
        pg: IVectorDb,
        neo: Any,
        mg: ArcadeDBSTMStore,
        embed_model: str = "",
    ) -> None:
        self.pg = pg
        self.neo = neo
        self.mg = mg
        self.embed_model = embed_model

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

        gather_results = await asyncio.gather(
            self._stm_entity_boost(words_graph, session_id, top_k),
            self._ltm_entity_traversal(words_graph, session_id, top_k),
            self._semantic_ann(emb, session_id, scope, top_k, emb2=emb2),
            self._stm_full_text(words, session_id, top_k),
            self._ltm_full_text(words, session_id, scope, top_k),
            self._ltm_temporal_traversal(query, session_id, scope, top_k),
            self._ltm_topic_match(emb, session_id, scope, top_k),
            return_exceptions=True,
        )
        (
            src_stm,
            src_ltm,
            src_semantic,
            src_text,
            src_ltm_text,
            src_temporal,
            src_topic,
        ) = gather_results  # type: ignore[misc]

        combined: dict[str, tuple[float, dict[str, Any]]] = {}
        for src in (
            src_stm,
            src_ltm,
            src_semantic,
            src_text,
            src_ltm_text,
            src_temporal,
            src_topic,
        ):
            if isinstance(src, Exception):
                log.warning("Retrieval source error: %s", src)
                continue
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
        text_map = await self.pg.get_chunks_by_ids(needs_meta) if needs_meta else {}

        # Apply _community_boost and _shape_score to every candidate BEFORE
        # sorting so the final order reflects the shaped score.
        shaped_combined: list[tuple[str, float, dict[str, Any], dict[str, Any]]] = []
        for cid, (base_score, info) in combined.items():
            fc = text_map.get(cid, {})
            meta = fc.get("metadata") or info.get("metadata") or {}
            shaped = _shape_score(base_score * _community_boost(meta), meta.get("memory_class"))
            shaped_combined.append((cid, shaped, info, fc))

        shaped_combined.sort(key=lambda x: -x[1])
        ranked_shaped = shaped_combined[:top_k]

        fused: list[dict[str, Any]] = []
        for rank_i, (cid, shaped, info, fc) in enumerate(ranked_shaped):
            meta = fc.get("metadata") or info.get("metadata") or {}
            ents = info.get("entities", [])
            fused.append(
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

        return RetrievalContext(
            query=query,
            session_id=session_id,
            fused_results=fused,
            search_type="DET",
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
        rows = await self.neo._qry(cypher_template, **(params or {}))
        return [dict(r) for r in rows]

    async def _stm_entity_boost(
        self,
        words: list[str],
        session_id: str,
        top_k: int,
    ) -> dict[str, tuple[float, dict[str, Any]]]:
        result: dict[str, tuple[float, dict[str, Any]]] = {}
        if not words:
            return result
        try:
            entity_chunks = await self.mg.query_entity_chunks(
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
        except Exception as exc:
            log.warning("STM entity boost failed: %s", exc)
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

    async def _stm_full_text(
        self,
        words: list[str],
        session_id: str,
        top_k: int,
    ) -> dict[str, tuple[float, dict[str, Any]]]:
        """Keyword text search — catches entities the extractor missed.

        Runs two passes: AND (all specific terms) for high-precision hits,
        then OR (any term) for broader coverage.  AND results rank higher.
        """
        result: dict[str, tuple[float, dict[str, Any]]] = {}
        specific = [w for w in words if w not in self._FT_STOPWORDS]
        if not specific:
            return result

        def _make_info(text: str) -> dict[str, Any]:
            return {
                "text": text,
                "entities": [],
                "sources": {"stm_text"},
                "doc_id": "",
                "metadata": {},
            }

        try:
            # High-precision pass: all specific terms must appear together.
            # Use boost=3 so an exact multi-term match can outrank a chunk
            # that only has entity+semantic support without the exact phrase.
            and_chunks = await self.mg.full_text_search_chunks(
                session_id=session_id, terms=specific, top_k=top_k, require_all=True
            )
            for rank, c in enumerate(and_chunks):
                cid = c["chunk_id"]
                result[cid] = (rrf_score(rank, boost=3.0), _make_info(c.get("text", "")))

            # Broader pass: any term (lower-ranked, only new chunks)
            or_chunks = await self.mg.full_text_search_chunks(
                session_id=session_id, terms=specific, top_k=top_k, require_all=False
            )
            for rank, c in enumerate(or_chunks):
                cid = c["chunk_id"]
                if cid not in result:
                    result[cid] = (rrf_score(rank + top_k), _make_info(c.get("text", "")))
        except Exception as exc:
            log.warning("STM full-text search failed: %s", exc)
        return result

    async def _discriminative_entities(
        self, safe_words: list[str], session_filter: str | None
    ) -> list[str]:
        """Query-matched entity names, minus saturated (non-discriminative) ones.

        An entity mentioned in more than ``_ENTITY_SATURATION_RATIO`` of the corpus
        (e.g. the two conversation participants) matches nearly every chunk, so
        seeding traversal on it buries the query-specific chunk under the whole
        conversation. Those are dropped so traversal seeds on specific entities
        (adoption, violin, church). Returns ``[]`` when nothing discriminative
        matched — callers then rely on the semantic/full-text channels, which
        already cover those chunks.
        """
        if not safe_words:
            return []
        # Short tokens (< 5 chars) use exact match to avoid false substring hits
        # (e.g. "jan" matching "January", "bob" matching "nobody").
        where_entity = " OR ".join(
            f"toLower(e.name) CONTAINS '{w}'" if len(w) >= 5 else f"toLower(e.name) = '{w}'"
            for w in safe_words
        )
        try:
            total_rows = await self.neo._qry("MATCH (c:CHUNK) RETURN count(c) AS n")
            total = int(total_rows[0]["n"]) if total_rows else 0
            if total <= 0:
                return []
            deg_rows = await self.neo._qry(
                f"MATCH (e:ENTITY) WHERE {where_entity} "
                f"MATCH (e)<-[:MENTIONS]-(c:CHUNK) "
                f"RETURN e.name AS name, count(DISTINCT c) AS deg"
            )
        except Exception as exc:
            log.debug("discriminative-entity filter unavailable (%s); skipping traversal", exc)
            return []
        cap = max(1, int(_ENTITY_SATURATION_RATIO * total))
        return [r["name"] for r in deg_rows if r.get("name") and int(r.get("deg") or 0) <= cap]

    async def _ltm_entity_traversal(
        self,
        words: list[str],
        session_id: str,
        top_k: int,
    ) -> dict[str, tuple[float, dict[str, Any]]]:
        if not words:
            return {}
        try:
            return await self._run_entity_traversal_async(words, session_id or None, top_k)
        except Exception as exc:
            log.warning("LTM entity traversal failed: %s", exc)
            return {}

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
        safe_names = [n.replace("'", "") for n in disc]
        # Exact-match the (already query-matched, non-saturated) entity names.
        where_entity = " OR ".join(f"e.name = '{n}'" for n in safe_names)
        try:
            if session_filter:
                cypher = (
                    f"MATCH (sess:SESSION {{id: $sid}})-[:IN_SESSION]->(f:FILE)"
                    f"-[:HAS_CHUNK]->(c:CHUNK) "
                    f"MATCH (c)-[:MENTIONS]->(e:ENTITY) "
                    f"WHERE {where_entity} "
                    f"WITH c, collect(DISTINCT e.name) AS entities, count(DISTINCT e) AS mc "
                    f"ORDER BY mc DESC LIMIT $top_k "
                    f"RETURN c.id AS chunk_id, c.text AS text, entities"
                )
                rows = await self.neo._qry(cypher, top_k=top_k, sid=session_filter)
            else:
                cypher = (
                    f"MATCH (e:ENTITY) WHERE {where_entity} "
                    f"MATCH (c:CHUNK)-[:MENTIONS]->(e) "
                    f"WITH c, collect(DISTINCT e.name) AS entities, count(DISTINCT e) AS mc "
                    f"ORDER BY mc DESC LIMIT $top_k "
                    f"RETURN c.id AS chunk_id, c.text AS text, entities"
                )
                rows = await self.neo._qry(cypher, top_k=top_k)

            for rank, row in enumerate(rows):
                _add_result(
                    result,
                    row["chunk_id"],
                    # boost=2: entity hits are high-precision; outcompete pure-ANN noise
                    rrf_score(rank, boost=2.0),
                    {
                        "text": row["text"],
                        "entities": row["entities"],
                        "sources": {"ltm_graph"},
                        "doc_id": "",
                        "metadata": {},
                    },
                )

            await self._run_relation_expansion_async(result, safe_names, session_filter, top_k)
        except Exception as exc:
            log.warning("LTM entity traversal error: %s", exc)
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
        try:
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
                rows = await self.neo._qry(rel_cypher, top_k=top_k, sid=session_filter)
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
                rows = await self.neo._qry(rel_cypher, top_k=top_k)

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
        except Exception as exc:
            log.debug("relation-expanded pass skipped: %s", exc)

    async def _ltm_full_text(
        self,
        words: list[str],
        session_id: str,
        scope: str,
        top_k: int,
    ) -> dict[str, tuple[float, dict[str, Any]]]:
        """Keyword text search on LTM chunks — mirrors _stm_full_text for LTM scope.

        Runs only when scope includes 'ltm'. Gives keyword recall for LTM mode
        equivalent to what _stm_full_text provides for STM, catching named entities
        and dates that ANN may rank lower than semantically similar but off-topic chunks.
        """
        result: dict[str, tuple[float, dict[str, Any]]] = {}
        if scope not in ("ltm", "both"):
            return result
        specific = [w for w in words if w not in self._FT_STOPWORDS]
        if not specific:
            return result

        neo_ft = getattr(self.neo, "full_text_search_chunks", None)
        if not callable(neo_ft):
            return result

        def _make_info(text: str) -> dict[str, Any]:
            return {
                "text": text,
                "entities": [],
                "sources": {"ltm_text"},
                "doc_id": "",
                "metadata": {},
            }

        try:
            and_chunks = await neo_ft(
                terms=specific, session_id=session_id or None, top_k=top_k, require_all=True
            )
            for rank, c in enumerate(and_chunks):
                cid = c["chunk_id"]
                result[cid] = (rrf_score(rank, boost=5.0), _make_info(c.get("text", "")))

            or_chunks = await neo_ft(
                terms=specific, session_id=session_id or None, top_k=top_k, require_all=False
            )
            for rank, c in enumerate(or_chunks):
                cid = c["chunk_id"]
                if cid not in result:
                    result[cid] = (rrf_score(rank + top_k), _make_info(c.get("text", "")))
        except Exception as exc:
            log.warning("LTM full-text search failed: %s", exc)
        return result

    async def _ltm_temporal_traversal(
        self,
        query: str,
        session_id: str,
        scope: str,
        top_k: int,
    ) -> dict[str, tuple[float, dict[str, Any]]]:
        """Match chunks linked to canonical TEMPORAL nodes for dates in the query.

        Activates only when the query contains a resolvable date; otherwise it is
        a no-op.  Matches both exact ISO dates and year-prefix dates so a query
        mentioning a year recalls finer-grained dates within it.
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
        try:
            if session_id:
                cypher = (
                    f"MATCH (sess:SESSION {{id: $sid}})-[:IN_SESSION]->(f:FILE)"
                    f"-[:HAS_CHUNK]->(c:CHUNK)-[:MENTIONS_DATE]->(d:TEMPORAL) "
                    f"WHERE {where_iso} "
                    f"RETURN DISTINCT c.id AS chunk_id, c.text AS text LIMIT $top_k"
                )
                rows = await self.neo._qry(cypher, top_k=top_k, sid=session_id)
            else:
                cypher = (
                    f"MATCH (c:CHUNK)-[:MENTIONS_DATE]->(d:TEMPORAL) "
                    f"WHERE {where_iso} "
                    f"RETURN DISTINCT c.id AS chunk_id, c.text AS text LIMIT $top_k"
                )
                rows = await self.neo._qry(cypher, top_k=top_k)
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
        except Exception as exc:
            log.warning("LTM temporal traversal failed: %s", exc)
        return result

    async def _ltm_topic_match(
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
        searcher = getattr(self.pg, "search_topics", None)
        if not callable(searcher):
            return result
        try:
            topics = await searcher(emb, k=2)
        except Exception as exc:
            log.debug("topic match skipped: %s", exc)
            return result
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
            try:
                rows = await self.neo._qry(
                    "MATCH (t:TOPIC {id: $tid})<-[:IN_TOPIC]-(e:ENTITY) "
                    "MATCH (c:CHUNK)-[:MENTIONS]->(e) "
                    "RETURN DISTINCT c.id AS chunk_id, c.text AS text LIMIT $k",
                    tid=tid,
                    k=top_k,
                )
            except Exception as exc:
                log.debug("topic member traversal skipped: %s", exc)
                continue
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

    async def _semantic_ann(
        self,
        emb: list[float],
        session_id: str,
        scope: str,
        top_k: int,
        emb2: list[float] | None = None,
    ) -> dict[str, tuple[float, dict[str, Any]]]:
        result: dict[str, tuple[float, dict[str, Any]]] = {}
        k = top_k * 2

        def _build_coros(
            embedding: list[float], tag: str
        ) -> tuple[list[Any], list[tuple[str, str]]]:
            coros: list[Any] = []
            tagged: list[tuple[str, str]] = []
            if scope in ("stm", "both") and session_id:
                coros.append(self.pg.search_stm_chunks(session_id, embedding, top_k=k))
                tagged.append(("stm_chunk", tag))
                coros.append(self.pg.search_stm_turns(session_id, embedding, top_k=k))
                tagged.append(("stm_turn", tag))
            if scope in ("ltm", "both"):
                # Pass session_id so the vector store applies session-scoped over-fetching
                # (k_pre = max(top_k * 20, 200)) and post-filters by session.  This prevents
                # cross-session contamination when multiple sessions coexist in LTM.
                ltm_sid = session_id or None
                coros.append(self.pg.search_ltm_chunks(embedding, top_k=k, session_id=ltm_sid))
                tagged.append(("ltm_chunk", tag))
                coros.append(self.pg.search_ltm_turns(embedding, top_k=k, session_id=ltm_sid))
                tagged.append(("ltm_turn", tag))
            return coros, tagged

        coros1, tagged1 = _build_coros(emb, "semantic")
        coros2, tagged2 = _build_coros(emb2, "semantic2") if emb2 is not None else ([], [])

        all_coros = coros1 + coros2
        all_tagged = tagged1 + tagged2

        if not all_coros:
            return result

        all_rows = await asyncio.gather(*all_coros, return_exceptions=True)

        # First pass: collect the best (minimum) rank per chunk across all ANN
        # passes (original query + declarative variant, stm/ltm sub-calls).
        # This prevents a chunk surfaced by both embeddings from receiving 2x
        # RRF mass from a single logical 'semantic' source.
        best_rank: dict[str, int] = {}
        best_row: dict[str, dict] = {}
        for (_label, _source_tag), rows in zip(all_tagged, all_rows, strict=False):
            if isinstance(rows, Exception):
                log.warning("ANN %s failed: %s", _label, rows)
                continue
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

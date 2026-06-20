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
from graphknows.memory.query._rrf_common import RRF_RELATION_BOOST, rrf_score
from graphknows.memory.query._rrf_common import shape_score as _shape_score
from graphknows.memory.query.base import BaseGraphRetriever
from graphknows.memory.query.context import RetrievalContext
from graphknows.memory.stores.arcadedb.stm_store import ArcadeDBSTMStore
from graphknows.memory.stores.embedder import embed

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


class DETRetriever(BaseGraphRetriever):
    """Deterministic three-collector hybrid retriever."""

    def __init__(
        self,
        pg: IVectorDb,
        neo: Any,
        mg: ArcadeDBSTMStore,
        embed_model: str = "openai/text-embedding-3-small",
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

        emb_array = await asyncio.to_thread(embed, [query], self.embed_model)
        emb = emb_array[0].tolist()
        # words_graph: include 3-char tokens (month names, short proper nouns) for
        # entity/graph matching where substring-of-name is the lookup key.
        # words: >3-char only for full-text search to reduce noise.
        words_graph = [w for w in re.findall(r"[a-z0-9][a-z0-9']*", query.lower()) if len(w) >= 3]
        words = [w for w in words_graph if len(w) > 3]

        gather_results = await asyncio.gather(
            self._stm_entity_boost(words_graph, session_id, top_k),
            self._ltm_entity_traversal(words_graph, session_id, top_k),
            self._semantic_ann(emb, session_id, scope, top_k),
            self._stm_full_text(words, session_id, top_k),
            self._ltm_full_text(words, session_id, scope, top_k),
            return_exceptions=True,
        )
        src_stm, src_ltm, src_semantic, src_text, src_ltm_text = gather_results  # type: ignore[misc]

        combined: dict[str, tuple[float, dict[str, Any]]] = {}
        for src in (src_stm, src_ltm, src_semantic, src_text, src_ltm_text):
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

        ranked = sorted(combined.items(), key=lambda x: -x[1][0])[:top_k]
        needs_fetch = [cid for cid, (_, info) in ranked if not info.get("metadata")]
        text_map = await self.pg.get_chunks_by_ids(needs_fetch) if needs_fetch else {}

        fused: list[dict[str, Any]] = []
        for rank_i, (cid, (base_score, info)) in enumerate(ranked):
            fc = text_map.get(cid, {})
            meta = fc.get("metadata") or info.get("metadata") or {}
            shaped = _shape_score(base_score * _community_boost(meta), meta.get("memory_class"))
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
        """Entity-graph traversal against LTM.

        Uses dynamic OR conditions in place of the any() list predicate —
        ArcadeDB's openCypher does not support any(x IN $list WHERE ...) syntax.
        """
        result: dict[str, tuple[float, dict[str, Any]]] = {}
        if not words:
            return result
        # Build safe dynamic WHERE clause: toLower(e.name) CONTAINS 'w1' OR ...
        safe = [w.replace("'", "") for w in words]
        where_entity = " OR ".join(f"toLower(e.name) CONTAINS '{w}'" for w in safe)
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
                    rrf_score(rank),
                    {
                        "text": row["text"],
                        "entities": row["entities"],
                        "sources": {"ltm_graph"},
                        "doc_id": "",
                        "metadata": {},
                    },
                )

            await self._run_relation_expansion_async(result, words, safe, session_filter, top_k)
        except Exception as exc:
            log.warning("LTM entity traversal error: %s", exc)
        return result

    async def _run_relation_expansion_async(
        self,
        result: dict[str, tuple[float, dict[str, Any]]],
        words: list[str],
        safe_words: list[str],
        session_filter: str | None,
        top_k: int,
    ) -> None:
        """2-hop relation expansion from seeded entities to neighbour chunks.

        Replaces any()/NONE() list predicates (unsupported in ArcadeDB Cypher)
        with dynamic OR WHERE conditions. Structural edge types are excluded via
        a NOT IN list embedded in the Cypher string.
        """
        try:
            where_entity = " OR ".join(f"toLower(e.name) CONTAINS '{w}'" for w in safe_words)
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

    async def _semantic_ann(
        self,
        emb: list[float],
        session_id: str,
        scope: str,
        top_k: int,
    ) -> dict[str, tuple[float, dict[str, Any]]]:
        result: dict[str, tuple[float, dict[str, Any]]] = {}
        k = top_k * 2

        coros: list[Any] = []
        labels: list[str] = []
        if scope in ("stm", "both") and session_id:
            coros.append(self.pg.search_stm_chunks(session_id, emb, top_k=k))
            labels.append("stm_chunk")
            coros.append(self.pg.search_stm_turns(session_id, emb, top_k=k))
            labels.append("stm_turn")
        if scope in ("ltm", "both"):
            # Pass session_id so the vector store applies session-scoped over-fetching
            # (k_pre = max(top_k * 20, 200)) and post-filters by session.  This prevents
            # cross-session contamination when multiple sessions coexist in LTM.
            ltm_sid = session_id or None
            coros.append(self.pg.search_ltm_chunks(emb, top_k=k, session_id=ltm_sid))
            labels.append("ltm_chunk")
            coros.append(self.pg.search_ltm_turns(emb, top_k=k, session_id=ltm_sid))
            labels.append("ltm_turn")

        if not coros:
            return result

        all_rows = await asyncio.gather(*coros, return_exceptions=True)
        for label, rows in zip(labels, all_rows, strict=False):
            if isinstance(rows, Exception):
                log.warning("ANN %s failed: %s", label, rows)
                continue
            for rank, row in enumerate(rows):
                metadata = row.get("metadata", {}) or {}
                cid = row["id"]
                _add_result(
                    result,
                    cid,
                    rrf_score(rank),
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

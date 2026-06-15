"""DETRetriever — deterministic entity-traversal + pgvector hybrid RAG with RRF fusion.

Four sources run concurrently and are merged via Reciprocal Rank Fusion (RRF):
  A. STM entity boost   — Memgraph entity→chunk lookup for the session
  B. Topic recall       — Neo4j TOPIC node keyword match
  C. LTM entity traversal — Neo4j two-pass entity→chunk (direct + relation-expanded)
  D. Semantic ANN       — pgvector ANN on stm_chunks, stm_turns, ltm_chunks, ltm_turns

All results are RRF-merged into a single ranked ``fused_results`` list inside
a ``RetrievalContext``.  The ``scope`` parameter gates which pgvector tables
Source D queries; the other three sources always run.
"""

from __future__ import annotations

import asyncio
import logging
import re
from collections import defaultdict
from typing import Any

from graphknows.memory.query._rrf_common import RRF_K
from graphknows.memory.query._rrf_common import shape_score as _shape_score
from graphknows.memory.query.base import BaseGraphRetriever
from graphknows.memory.query.context import RetrievalContext
from graphknows.memory.stores.embedder import embed
from graphknows.memory.stores.memgraph import MemgraphStore
from graphknows.memory.stores.neo4j import Neo4jStore
from graphknows.memory.stores.pgvector import PgVectorStore

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

# Chunk-id prefix for topic results — excluded from pgvector bulk-fetch.
_TOPIC_PREFIX = "topic:"


class DETRetriever(BaseGraphRetriever):
    """Deterministic entity-traversal hybrid retriever.

    All four retrieval sources (STM entity boost, topic recall, LTM entity
    traversal, semantic ANN) run concurrently and are merged with RRF.
    """

    def __init__(
        self,
        pg: PgVectorStore,
        neo: Neo4jStore,
        mg: MemgraphStore,
        embed_model: str = "openai/text-embedding-3-small",
    ) -> None:
        self.pg = pg
        self.neo = neo
        self.mg = mg
        self.embed_model = embed_model

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def retrieve(
        self,
        query: str,
        session_id: str,
        top_k: int = 10,
        scope: str = "both",
    ) -> RetrievalContext:
        """Run all four retrieval sources concurrently and return RRF-merged results.

        Args:
            query: Natural-language query.
            session_id: Session namespace for STM sources and pgvector scope.
            top_k: Maximum results in ``fused_results``.
            scope: ``"stm"`` | ``"ltm"`` | ``"both"`` — gates Source D tables.

        Returns:
            RetrievalContext with ``search_type="DET"``.
        """
        log.debug("DET retrieve  query=%r  session=%s  scope=%s  top_k=%d",
                  query, session_id, scope, top_k)

        emb_array = await asyncio.to_thread(embed, [query], self.embed_model)
        emb = emb_array[0].tolist()
        words = [w for w in query.lower().split() if len(w) > 3]

        src_a, src_b, src_c, src_d = await asyncio.gather(
            self._stm_entity_boost(words, session_id, top_k),
            self._topic_recall(words, top_k),
            self._ltm_entity_traversal(words, session_id, top_k),
            self._semantic_ann(emb, session_id, scope, top_k),
            return_exceptions=True,
        )

        # Merge scores across sources; accumulate per unique chunk_id.
        combined: dict[str, tuple[float, dict[str, Any]]] = {}
        for src in (src_a, src_b, src_c, src_d):
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

        # Fetch canonical text from pgvector for chunks that lack inline text.
        needs_fetch = [
            cid
            for cid, (_, info) in ranked
            if not info.get("text") and not cid.startswith(_TOPIC_PREFIX)
        ]
        text_map = await self.pg.get_chunks_by_ids(needs_fetch) if needs_fetch else {}

        fused: list[dict[str, Any]] = []
        for rank_i, (cid, (rrf_score, info)) in enumerate(ranked):
            fc = text_map.get(cid, {})
            meta = fc.get("metadata") or info.get("metadata") or {}
            memory_class = meta.get("memory_class")
            shaped = _shape_score(rrf_score, memory_class)
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

        log.debug("DET retrieve DONE  fused=%d", len(fused))
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
        """Execute a parameterized read-only Cypher query.

        User values must be passed via *params* — never interpolated into the
        template.  Write keywords and f-string markers are rejected.

        Raises:
            ValueError: On ``{var}`` interpolation markers or write keywords.
        """
        if re.search(r"\{[^}]+\}", cypher_template):
            raise ValueError(
                "Cypher template must not contain {var} interpolation — "
                "use $param placeholders and pass values via the params dict"
            )
        forbidden = re.compile(r"\b(CREATE|MERGE|SET|DELETE|DETACH|REMOVE|DROP)\b", re.IGNORECASE)
        if forbidden.search(cypher_template):
            raise ValueError(
                "Only read-only Cypher is permitted. "
                "Detected a write keyword (CREATE/MERGE/SET/DELETE/…)."
            )
        rows = await self.neo._arun(cypher_template, **(params or {}))
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Source A — STM entity boost (Memgraph)
    # ------------------------------------------------------------------

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
                rrf = 1.0 / (RRF_K + rank + 1)
                cid = ec["chunk_id"]
                if cid in result:
                    result[cid] = (result[cid][0] + rrf, result[cid][1])
                else:
                    result[cid] = (
                        rrf,
                        {
                            "text": ec.get("text", ""),
                            "entities": ec.get("entity_names", []),
                            "sources": {"stm_graph"},
                            "doc_id": ec.get("file_id", ""),
                            "metadata": ec.get("metadata", {}),
                        },
                    )
        except Exception as exc:
            log.warning("STM entity boost failed: %s", exc)
        return result

    # ------------------------------------------------------------------
    # Source B — Topic recall (Neo4j TOPIC nodes)
    # ------------------------------------------------------------------

    async def _topic_recall(
        self,
        words: list[str],
        top_k: int,
    ) -> dict[str, tuple[float, dict[str, Any]]]:
        result: dict[str, tuple[float, dict[str, Any]]] = {}
        if not words:
            return result
        try:
            rows = await asyncio.to_thread(self._run_topic_cypher, words, top_k)
            token_set = set(words)
            rank = 0
            for row in rows:
                entity_score = float(row.get("score") or 0)
                summary = row.get("summary") or row.get("label") or ""
                if entity_score == 0:
                    text_bonus = sum(1 for t in token_set if t in summary.lower()) if summary else 0
                    if text_bonus == 0:
                        continue
                rrf = 1.0 / (RRF_K + rank + 1)
                topic_id = f"{_TOPIC_PREFIX}{row.get('topic_id', rank)}"
                result[topic_id] = (
                    rrf,
                    {
                        "text": summary,
                        "entities": list(row.get("entities") or []),
                        "sources": {"topic"},
                        "doc_id": "",
                        "metadata": {},
                    },
                )
                rank += 1
        except Exception as exc:
            log.warning("Topic recall failed: %s", exc)
        return result

    def _run_topic_cypher(self, words: list[str], top_k: int) -> list[dict]:
        cypher = """
        UNWIND $words AS word
        MATCH (e)
        WHERE (e:ENTITY OR e:ENTITY_CONCEPT) AND toLower(e.name) CONTAINS word
        WITH collect(DISTINCT e.name) AS matched_names
        MATCH (t:TOPIC)
        OPTIONAL MATCH (e2)-[:IN_TOPIC]->(t)
        WHERE e2.name IN matched_names
        WITH t,
             collect(DISTINCT e2.name) AS matched_entities,
             count(DISTINCT e2) AS overlap
        ORDER BY overlap DESC
        LIMIT $top_k
        RETURN t.id AS topic_id,
               t.label AS label,
               t.summary AS summary,
               matched_entities AS entities,
               overlap AS score
        """
        driver = getattr(self.neo, "_driver", self.neo)
        _db = getattr(self.neo, "_db", None)
        kwargs: dict[str, Any] = {"database": _db} if _db else {}
        with driver.session(**kwargs) as s:
            return s.run(cypher, words=words, top_k=top_k).data()

    # ------------------------------------------------------------------
    # Source C — LTM entity traversal (Neo4j, two passes)
    # ------------------------------------------------------------------

    async def _ltm_entity_traversal(
        self,
        words: list[str],
        session_id: str,
        top_k: int,
    ) -> dict[str, tuple[float, dict[str, Any]]]:
        if not words:
            return {}
        try:
            return await asyncio.to_thread(
                self._run_entity_traversal_sync,
                words,
                session_id or None,
                top_k,
            )
        except Exception as exc:
            log.warning("LTM entity traversal failed: %s", exc)
            return {}

    def _run_entity_traversal_sync(
        self,
        words: list[str],
        session_filter: str | None,
        top_k: int,
    ) -> dict[str, tuple[float, dict[str, Any]]]:
        result: dict[str, tuple[float, dict[str, Any]]] = {}
        driver = getattr(self.neo, "_driver", self.neo)
        _db = getattr(self.neo, "_db", None)
        kwargs: dict[str, Any] = {"database": _db} if _db else {}
        try:
            with driver.session(**kwargs) as s:
                # Pass 1 — direct entity → chunk
                if session_filter:
                    cypher = """
                    MATCH (sess:SESSION {id: $sid})-[:IN_SESSION]->(f:FILE)-[:HAS_CHUNK]->(c:CHUNK)
                    MATCH (c)-[:MENTIONS]->(e:ENTITY)
                    WHERE any(w IN $words WHERE toLower(e.name) CONTAINS w)
                    WITH c, collect(DISTINCT e.name) AS entities, count(DISTINCT e) AS mc
                    ORDER BY mc DESC LIMIT $top_k
                    RETURN c.id AS chunk_id, c.text AS text, entities
                    """
                    params: dict[str, Any] = {"words": words, "top_k": top_k, "sid": session_filter}
                else:
                    cypher = """
                    MATCH (e:ENTITY)
                    WHERE any(w IN $words WHERE toLower(e.name) CONTAINS w)
                    MATCH (c:CHUNK)-[:MENTIONS]->(e)
                    WITH c, collect(DISTINCT e.name) AS entities, count(DISTINCT e) AS mc
                    ORDER BY mc DESC LIMIT $top_k
                    RETURN c.id AS chunk_id, c.text AS text, entities
                    """
                    params = {"words": words, "top_k": top_k}

                for rank, row in enumerate(s.run(cypher, **params).data()):
                    rrf = 1.0 / (RRF_K + rank + 1)
                    cid = row["chunk_id"]
                    if cid in result:
                        result[cid] = (result[cid][0] + rrf, result[cid][1])
                    else:
                        result[cid] = (
                            rrf,
                            {
                                "text": row["text"],
                                "entities": row["entities"],
                                "sources": {"ltm_graph"},
                                "doc_id": "",
                                "metadata": {},
                            },
                        )

                # Pass 2 — relation-expanded (no-op when no relation edges exist)
                try:
                    skip_types = list(_STRUCTURAL_REL_TYPES)
                    if session_filter:
                        rel_cypher = """
                        MATCH (sess:SESSION {id: $sid})-[:IN_SESSION]->(f:FILE)-[:HAS_CHUNK]->(c0:CHUNK)
                        MATCH (c0)-[:MENTIONS]->(e:ENTITY)
                        WHERE any(w IN $words WHERE toLower(e.name) CONTAINS w)
                        MATCH (e)-[r]-(e2:ENTITY)
                        WHERE NOT type(r) IN $skip_types
                        MATCH (sess)-[:IN_SESSION]->(f2:FILE)-[:HAS_CHUNK]->(c:CHUNK)-[:MENTIONS]->(e2)
                        WITH c, collect(DISTINCT e2.name) AS ents, count(DISTINCT e2) AS mc
                        ORDER BY mc DESC LIMIT $top_k
                        RETURN c.id AS chunk_id, c.text AS text, ents AS entities
                        """
                        rel_params: dict[str, Any] = {
                            "words": words, "top_k": top_k,
                            "sid": session_filter, "skip_types": skip_types,
                        }
                    else:
                        rel_cypher = """
                        MATCH (e:ENTITY)
                        WHERE any(w IN $words WHERE toLower(e.name) CONTAINS w)
                        MATCH (e)-[r]-(e2:ENTITY)
                        WHERE NOT type(r) IN $skip_types
                        MATCH (c:CHUNK)-[:MENTIONS]->(e2)
                        WITH c, collect(DISTINCT e2.name) AS ents, count(DISTINCT e2) AS mc
                        ORDER BY mc DESC LIMIT $top_k
                        RETURN c.id AS chunk_id, c.text AS text, ents AS entities
                        """
                        rel_params = {"words": words, "top_k": top_k, "skip_types": skip_types}

                    for rank, row in enumerate(s.run(rel_cypher, **rel_params).data()):
                        rrf = 1.5 / (RRF_K + rank + 1)  # boosted for relation-expanded results
                        cid = row["chunk_id"]
                        if cid in result:
                            result[cid] = (result[cid][0] + rrf, result[cid][1])
                        else:
                            result[cid] = (
                                rrf,
                                {
                                    "text": row["text"],
                                    "entities": row["entities"],
                                    "sources": {"ltm_graph"},
                                    "doc_id": "",
                                    "metadata": {},
                                },
                            )
                except Exception as rel_exc:
                    log.debug("relation-expanded pass skipped: %s", rel_exc)
        except Exception as exc:
            log.warning("LTM entity traversal error: %s", exc)
        return result

    # ------------------------------------------------------------------
    # Source D — Semantic ANN (pgvector, all four tables by scope)
    # ------------------------------------------------------------------

    async def _semantic_ann(
        self,
        emb: list[float],
        session_id: str,
        scope: str,
        top_k: int,
    ) -> dict[str, tuple[float, dict[str, Any]]]:
        result: dict[str, tuple[float, dict[str, Any]]] = {}
        k = top_k * 2  # more candidates for RRF merge

        coros: list[Any] = []
        labels: list[str] = []
        if scope in ("stm", "both") and session_id:
            coros.append(self.pg.search_stm_chunks(session_id, emb, top_k=k))
            labels.append("stm_chunk")
            coros.append(self.pg.search_stm_turns(session_id, emb, top_k=k))
            labels.append("stm_turn")
        if scope in ("ltm", "both"):
            coros.append(self.pg.search_ltm_chunks(emb, top_k=k, session_id=None))
            labels.append("ltm_chunk")
            coros.append(self.pg.search_ltm_turns(emb, top_k=k, session_id=None))
            labels.append("ltm_turn")

        if not coros:
            return result

        all_rows = await asyncio.gather(*coros, return_exceptions=True)

        for label, rows in zip(labels, all_rows):
            if isinstance(rows, Exception):
                log.warning("ANN %s failed: %s", label, rows)
                continue
            for rank, r in enumerate(rows):
                rrf = 1.0 / (RRF_K + rank + 1)
                cid = r["id"]
                if cid in result:
                    result[cid] = (result[cid][0] + rrf, result[cid][1])
                else:
                    result[cid] = (
                        rrf,
                        {
                            "text": r.get("text", ""),
                            "entities": r.get("metadata", {}).get("entity_names", []),
                            "sources": {"semantic"},
                            "doc_id": r.get("file_id", r.get("doc_id", "")),
                            "metadata": r.get("metadata", {}),
                        },
                    )

        return result


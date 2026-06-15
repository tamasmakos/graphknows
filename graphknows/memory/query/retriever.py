"""Deterministic entity-traversal + pgvector hybrid RAG with RRF fusion.

Three collectors run concurrently:
  A. STM structural recall from Memgraph.
  B. LTM structural recall from Neo4j.
  C. Semantic recall from pgvector.

Topic nodes are LLM-assisted enrichment artifacts; deterministic retrieval uses
community metadata as score shaping rather than returning synthetic topic hits.
"""

from __future__ import annotations

import asyncio
import logging
import re
from typing import Any

from graphknows.memory.query._rrf_common import RRF_RELATION_BOOST
from graphknows.memory.query._rrf_common import rrf_score
from graphknows.memory.query._rrf_common import shape_score as _shape_score
from graphknows.config.adapters.ports.vector_db import IVectorDb
from graphknows.memory.query.base import BaseGraphRetriever
from graphknows.memory.query.context import RetrievalContext
from graphknows.memory.stores.embedder import embed
from graphknows.memory.stores.memgraph import MemgraphStore
from graphknows.memory.stores.neo4j import Neo4jStore

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
        neo: Neo4jStore,
        mg: MemgraphStore,
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
        words = [w for w in query.lower().split() if len(w) > 3]

        src_stm, src_ltm, src_semantic = await asyncio.gather(
            self._stm_entity_boost(words, session_id, top_k),
            self._ltm_entity_traversal(words, session_id, top_k),
            self._semantic_ann(emb, session_id, scope, top_k),
            return_exceptions=True,
        )

        combined: dict[str, tuple[float, dict[str, Any]]] = {}
        for src in (src_stm, src_ltm, src_semantic):
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
        needs_fetch = [cid for cid, (_, info) in ranked if not info.get("text")]
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
        """Execute a parameterized read-only Cypher query."""
        if re.search(r"\{[^}]+\}", cypher_template):
            raise ValueError(
                "Cypher template must not contain {var} interpolation; "
                "use $param placeholders and pass values via params"
            )
        forbidden = re.compile(r"\b(CREATE|MERGE|SET|DELETE|DETACH|REMOVE|DROP)\b", re.IGNORECASE)
        if forbidden.search(cypher_template):
            raise ValueError("Only read-only Cypher is permitted")
        rows = await self.neo._arun(cypher_template, **(params or {}))
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
                if session_filter:
                    cypher = """
                    MATCH (sess:SESSION {id: $sid})-[:IN_SESSION]->(f:FILE)-[:HAS_CHUNK]->(c:CHUNK)
                    MATCH (c)-[:MENTIONS]->(e:ENTITY)
                    WHERE any(w IN $words WHERE toLower(e.name) CONTAINS w)
                    WITH c, collect(DISTINCT e.name) AS entities, count(DISTINCT e) AS mc
                    ORDER BY mc DESC LIMIT $top_k
                    RETURN c.id AS chunk_id, c.text AS text, entities
                    """
                    params: dict[str, Any] = {
                        "words": words,
                        "top_k": top_k,
                        "sid": session_filter,
                    }
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

                self._run_relation_expansion(s, result, words, session_filter, top_k)
        except Exception as exc:
            log.warning("LTM entity traversal error: %s", exc)
        return result

    def _run_relation_expansion(
        self,
        session: Any,
        result: dict[str, tuple[float, dict[str, Any]]],
        words: list[str],
        session_filter: str | None,
        top_k: int,
    ) -> None:
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
                params: dict[str, Any] = {
                    "words": words,
                    "top_k": top_k,
                    "sid": session_filter,
                    "skip_types": skip_types,
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
                params = {"words": words, "top_k": top_k, "skip_types": skip_types}

            for rank, row in enumerate(session.run(rel_cypher, **params).data()):
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


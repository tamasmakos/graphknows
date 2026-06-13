"""HybridRetriever — entity-first graph retrieval + pgvector semantic search + RRF fusion.

Algorithm
---------
1. Embed the query with BGE-small-en-v1.5 (shared stores/embedder singleton).
2. Entity-first graph path:
   a. ANN search entities table → top k_entity entities by cosine similarity.
   b. For each entity, traverse MENTIONS edges in Neo4j → linked Chunk nodes.
   c. Collect (chunk_id, entity_rank) pairs for RRF scoring.
3. Semantic path:
   a. ANN search chunks table filtered to session_id → top k_semantic chunks.
4. RRF fusion (K=60):
   - Graph path contributes: 1 / (60 + entity_rank + 1) per chunk.
   - Semantic path contributes: 1 / (60 + semantic_rank + 1) per chunk.
   - Sum scores per unique chunk_id, sort descending, take top k_final.
5. Fetch full text + metadata for fused chunk IDs from pgvector.
6. Return a RetrievalContext with all three views: graph_matches, semantic_matches, fused_results.
"""

from __future__ import annotations

import asyncio
import logging
import re
from collections import defaultdict

from graphknows.memory.query.base import BaseGraphRetriever
from graphknows.memory.query.context import ChunkMatch, EntityMatch, RetrievalContext
from graphknows.memory.stores.embedder import embed
from graphknows.memory.stores.neo4j import Neo4jStore
from graphknows.memory.stores.pgvector import PgVectorStore

log = logging.getLogger(__name__)

RRF_K: int = 60
DEFAULT_K_ENTITY: int = 10
DEFAULT_K_SEMANTIC: int = 20
DEFAULT_K_FINAL: int = 10
MAX_CHUNKS_PER_ENTITY: int = 5


class HybridRetriever(BaseGraphRetriever):
    """Stateless retriever: entity-graph traversal + dense vector search + RRF fusion.

    Attributes:
        pg: Connected PgVectorStore.
        neo: Connected Neo4jStore.
        embed_model: Name of the sentence-transformer model.
    """

    def __init__(
        self,
        pg: PgVectorStore,
        neo: Neo4jStore,
        embed_model: str = "BAAI/bge-small-en-v1.5",
    ) -> None:
        self.pg = pg
        self.neo = neo
        self.embed_model = embed_model

    async def retrieve(  # noqa: C901  # NOSONAR
        self,
        query: str,
        session_id: str,
        top_k: int = DEFAULT_K_FINAL,
        scope: str = "both",
        k_entity: int = DEFAULT_K_ENTITY,
        k_semantic: int = DEFAULT_K_SEMANTIC,
    ) -> RetrievalContext:
        """Run the hybrid retrieval pipeline.

        Embeds the query once, then concurrently:
        - Runs an ANN entity search (graph path, degrades gracefully if no entities table).
        - Runs ANN chunk search on the tier(s) selected by *scope*.

        Both result sets are merged via RRF and returned as a ranked flat list.

        Args:
            query: Natural-language query string.
            session_id: Session namespace used when scope includes ``"stm"``.
            top_k: Maximum fused results to return.
            scope: Which memory tier(s) to search — ``"stm"`` (session chunks),
                ``"ltm"`` (promoted chunks), or ``"both"`` (merge both, default).
            k_entity: Candidate entity count for the graph path.
            k_semantic: Candidate chunk count per tier for the semantic path.

        Returns:
            RetrievalContext with ranked fused_results.
        """
        log.debug(
            "retrieve START  query=%r  session=%s  scope=%s  k_entity=%d  k_semantic=%d  top_k=%d",
            query, session_id, scope, k_entity, k_semantic, top_k,
        )

        # ── Step 1: embed query once ───────────────────────────────────────
        emb_array = await asyncio.to_thread(embed, [query], self.embed_model)
        emb = emb_array[0].tolist()

        # ── Step 2: run entity + chunk searches concurrently ──────────────
        # Entity search degrades to [] when the entities table is absent.
        # Chunk search routes to stm_chunks / ltm_chunks / both based on scope.
        entity_rows_task = asyncio.create_task(self.pg.search_entities(emb, k=k_entity))
        sem_rows_task = asyncio.create_task(
            self._search_chunks_by_scope(emb, session_id, scope, k_semantic)
        )
        entity_rows, sem_rows = await asyncio.gather(entity_rows_task, sem_rows_task)
        log.debug("ANN  entities=%d  chunks=%d", len(entity_rows), len(sem_rows))

        # ── Step 3: entity → chunks via Neo4j MENTIONS (graph path) ───────
        graph_chunk_info: dict[str, dict] = {}
        entity_matches: list[EntityMatch] = []

        if entity_rows:
            neo_tasks = [
                self.neo.get_chunks_for_entity(ent["name"], top_k=MAX_CHUNKS_PER_ENTITY)
                for ent in entity_rows
            ]
            neo_results = await asyncio.gather(*neo_tasks, return_exceptions=True)

            for i, (ent, neo_chunks) in enumerate(zip(entity_rows, neo_results, strict=False)):
                if isinstance(neo_chunks, Exception):
                    log.warning("Neo4j lookup failed for %r: %s", ent["name"], neo_chunks)
                    neo_chunks = []
                entity_similarity = ent["score"]
                chunk_matches: list[ChunkMatch] = []
                for j, nc in enumerate(neo_chunks):
                    mention_weight = entity_similarity * (1.0 / (j + 1))
                    cid = nc["chunk_id"]
                    if cid not in graph_chunk_info or mention_weight > graph_chunk_info[cid]["mention_weight"]:
                        graph_chunk_info[cid] = {
                            "entity_rank": i,
                            "mention_weight": mention_weight,
                            "doc_id": nc.get("doc_id", ""),
                            "text": nc.get("text", ""),
                        }
                    chunk_matches.append(ChunkMatch(
                        chunk_id=cid,
                        doc_id=nc.get("doc_id", ""),
                        text=nc.get("text", ""),
                        metadata={"mention_weight": mention_weight},
                        entities=[{"name": ent["name"], "label": ent["label"]}],
                    ))
                entity_matches.append(EntityMatch(
                    name=ent["name"], label=ent["label"],
                    similarity=entity_similarity, chunks=chunk_matches,
                ))

        # ── Step 4: semantic matches list ──────────────────────────────────
        semantic_matches: list[ChunkMatch] = [
            ChunkMatch(
                chunk_id=r["chunk_id"],
                doc_id=r.get("doc_id", ""),
                text=r.get("text", ""),
                metadata={"cosine_score": r.get("cosine_score", 0.0), **r.get("metadata", {})},
                entities=[],
            )
            for r in sem_rows
        ]

        # ── Step 5: RRF fusion ─────────────────────────────────────────────
        rrf_scores: dict[str, float] = defaultdict(float)
        sem_ids: set[str] = set()

        for cid, info in graph_chunk_info.items():
            rrf_scores[cid] += 1.0 / (RRF_K + info["entity_rank"] + 1)

        for rank, r in enumerate(sem_rows):
            cid = r["chunk_id"]
            rrf_scores[cid] += 1.0 / (RRF_K + rank + 1)
            sem_ids.add(cid)

        ranked = sorted(rrf_scores.items(), key=lambda x: -x[1])[:top_k]
        log.debug("RRF  unique=%d  returning=%d", len(rrf_scores), len(ranked))

        # ── Step 6: bulk-fetch full text for fused chunk IDs ──────────────
        full_text_map = await self.pg.get_chunks_by_ids([cid for cid, _ in ranked])

        chunk_entities: dict[str, list[dict]] = defaultdict(list)
        for em in entity_matches:
            for cm in em.chunks:
                chunk_entities[cm.chunk_id].append({"name": em.name, "label": em.label})

        fused: list[dict] = []
        for rank, (cid, rrf_score) in enumerate(ranked):
            fc = full_text_map.get(cid, {})
            sources = []
            if cid in graph_chunk_info:
                sources.append("graph")
            if cid in sem_ids:
                sources.append("semantic")
            fused.append({
                "chunk_id": cid,
                "doc_id": fc.get("doc_id", graph_chunk_info.get(cid, {}).get("doc_id", "")),
                "text": fc.get("text", graph_chunk_info.get(cid, {}).get("text", "")),
                "metadata": fc.get("metadata", {}),
                "rrf_score": round(rrf_score, 6),
                "sources": ",".join(sources),
                "rank": rank + 1,
                "entities": chunk_entities.get(cid, []),
            })

        log.debug("retrieve DONE  fused=%d", len(fused))
        return RetrievalContext(
            query=query,
            session_id=session_id,
            graph_matches=entity_matches,
            semantic_matches=semantic_matches,
            fused_results=fused,
            search_type="HYBRID",
        )

    async def _search_chunks_by_scope(
        self,
        emb: list[float],
        session_id: str,
        scope: str,
        k: int,
    ) -> list[dict]:
        """Run ANN chunk search against the tier(s) selected by *scope*.

        - ``"stm"``: stm_chunks scoped to session_id.
        - ``"ltm"``: ltm_chunks (all sessions).
        - ``"both"``: merge stm + ltm, deduplicate by chunk_id, sort by score.
        """
        if scope == "stm":
            return await self.pg.search_chunks(emb, session_id=session_id, k=k)
        if scope == "ltm":
            return await self.pg.search_chunks(emb, session_id="", k=k)
        # "both": query both tiers concurrently, merge by score
        stm_task = asyncio.create_task(
            self.pg.search_chunks(emb, session_id=session_id, k=k)
        ) if session_id else None
        ltm_task = asyncio.create_task(
            self.pg.search_chunks(emb, session_id="", k=k)
        )
        tasks = [t for t in (stm_task, ltm_task) if t is not None]
        results = await asyncio.gather(*tasks)
        stm_rows = results[0] if stm_task else []
        ltm_rows = results[-1]

        merged: dict[str, dict] = {}
        for r in stm_rows:
            merged[r["chunk_id"]] = r
        for r in ltm_rows:
            if r["chunk_id"] not in merged:
                merged[r["chunk_id"]] = r
        return sorted(merged.values(), key=lambda x: -x.get("cosine_score", 0))[:k]

    # ------------------------------------------------------------------
    # SEMANTIC mode — pure pgvector ANN on session-scoped chunks
    # ------------------------------------------------------------------

    async def retrieve_semantic(
        self,
        query: str,
        session_id: str,
        k: int = 20,
    ) -> RetrievalContext:
        """Pure vector search on chunks — no graph traversal.

        Args:
            query: Natural-language query.
            session_id: Session namespace for chunk search.
            k: Number of chunks to return.

        Returns:
            RetrievalContext with semantic_matches and fused_results only.
        """
        emb_array = await asyncio.to_thread(embed, [query], self.embed_model)
        emb = emb_array[0].tolist()

        sem_rows = await self.pg.search_chunks(emb, session_id=session_id, k=k)
        semantic_matches = [
            ChunkMatch(
                chunk_id=r["chunk_id"],
                doc_id=r.get("doc_id", ""),
                text=r.get("text", ""),
                metadata={"cosine_score": r.get("cosine_score", 0.0), **r.get("metadata", {})},
                entities=[],
            )
            for r in sem_rows
        ]
        fused = [
            {
                "chunk_id": cm.chunk_id,
                "doc_id": cm.doc_id,
                "text": cm.text,
                "metadata": cm.metadata,
                "rrf_score": cm.metadata.get("cosine_score", 0.0),
                "sources": "semantic",
                "rank": i + 1,
                "entities": [],
            }
            for i, cm in enumerate(semantic_matches)
        ]
        return RetrievalContext(
            query=query,
            session_id=session_id,
            semantic_matches=semantic_matches,
            fused_results=fused,
            search_type="SEMANTIC",
        )

    # ------------------------------------------------------------------
    # GRAPH_TRAVERSAL mode — entity match → multi-hop neighborhood → chunks
    # ------------------------------------------------------------------

    async def retrieve_graph_traversal(  # noqa: C901  # NOSONAR
        self,
        query: str,
        session_id: str,
        k_entity: int = 5,
        depth: int = 2,
        k_final: int = 10,
    ) -> RetrievalContext:
        """Entity-first graph traversal with multi-hop neighbor expansion.

        Args:
            query: Natural-language query.
            session_id: Session namespace (passed through to context).
            k_entity: Number of seed entities to retrieve.
            depth: Graph traversal depth (capped at 3).
            k_final: Maximum fused results to return.

        Returns:
            RetrievalContext with deep graph-traversal results.
        """
        emb_array = await asyncio.to_thread(embed, [query], self.embed_model)
        emb = emb_array[0].tolist()

        entity_rows = await self.pg.search_entities(emb, k=k_entity)
        entity_matches: list[EntityMatch] = []
        seen_chunks: dict[str, dict] = {}

        for ent in entity_rows:
            # Direct MENTIONS chunks
            direct_chunks = await self.neo.get_chunks_for_entity(
                ent["name"], top_k=MAX_CHUNKS_PER_ENTITY
            )
            # Neighbor entities and their chunks
            neighbor_info = await self.neo.get_entity_neighbors(ent["name"], depth=depth)
            neighbor_names = [n["name"] for n in neighbor_info.get("neighbors", [])]
            neighbor_chunk_tasks = [
                self.neo.get_chunks_for_entity(n, top_k=2)
                for n in neighbor_names[:10]  # limit fan-out
            ]
            neighbor_chunk_results = await asyncio.gather(
                *neighbor_chunk_tasks, return_exceptions=True
            )

            all_raw = list(direct_chunks)
            for result in neighbor_chunk_results:
                if not isinstance(result, Exception):
                    all_raw.extend(result)

            chunk_matches: list[ChunkMatch] = []
            for _j, nc in enumerate(all_raw):
                cid = nc["chunk_id"]
                if cid not in seen_chunks:
                    seen_chunks[cid] = {
                        "entity_rank": entity_rows.index(ent),
                        "doc_id": nc.get("doc_id", ""),
                        "text": nc.get("text", ""),
                    }
                chunk_matches.append(
                    ChunkMatch(
                        chunk_id=cid,
                        doc_id=nc.get("doc_id", ""),
                        text=nc.get("text", ""),
                        metadata={"traversal_depth": depth},
                        entities=[{"name": ent["name"], "label": ent["label"]}],
                    )
                )

            entity_matches.append(
                EntityMatch(
                    name=ent["name"],
                    label=ent["label"],
                    similarity=ent["score"],
                    chunks=chunk_matches,
                )
            )

        # Build fused results deduped by chunk_id
        fused: list[dict] = []
        seen: set[str] = set()
        for em in entity_matches:
            for cm in em.chunks:
                if cm.chunk_id not in seen:
                    seen.add(cm.chunk_id)
                    fused.append(
                        {
                            "chunk_id": cm.chunk_id,
                            "doc_id": cm.doc_id,
                            "text": cm.text,
                            "metadata": cm.metadata,
                            "rrf_score": 0.0,
                            "sources": "graph",
                            "rank": len(fused) + 1,
                            "entities": cm.entities,
                        }
                    )
                    if len(fused) >= k_final:
                        break
            if len(fused) >= k_final:
                break

        return RetrievalContext(
            query=query,
            session_id=session_id,
            graph_matches=entity_matches,
            fused_results=fused,
            search_type="GRAPH_TRAVERSAL",
        )

    # ------------------------------------------------------------------
    # SUMMARIES mode — entity match → community Topic summaries
    # ------------------------------------------------------------------

    async def retrieve_summaries(
        self,
        query: str,
        session_id: str,
        k_entity: int = 5,
    ) -> RetrievalContext:
        """Return community/topic summaries for entities matching the query.

        Args:
            query: Natural-language query.
            session_id: Passed through to context.
            k_entity: Number of entities to look up.

        Returns:
            RetrievalContext whose fused_results contain topic summary dicts.
        """
        emb_array = await asyncio.to_thread(embed, [query], self.embed_model)
        emb = emb_array[0].tolist()

        entity_rows = await self.pg.search_entities(emb, k=k_entity)

        community_tasks = [self.neo.get_entity_community(ent["name"]) for ent in entity_rows]
        community_results = await asyncio.gather(*community_tasks, return_exceptions=True)

        fused: list[dict] = []
        for ent, topic in zip(entity_rows, community_results, strict=False):
            if isinstance(topic, Exception) or topic is None:
                continue
            fused.append(
                {
                    "chunk_id": f"topic:{topic.get('id', '')}",
                    "doc_id": "",
                    "text": f"{topic.get('label', '')}\n\n{topic.get('summary', '')}",
                    "metadata": {"community_label": topic.get("label", "")},
                    "rrf_score": round(ent["score"], 6),
                    "sources": "summaries",
                    "rank": len(fused) + 1,
                    "entities": [{"name": ent["name"], "label": ent["label"]}],
                }
            )

        return RetrievalContext(
            query=query,
            session_id=session_id,
            fused_results=fused,
            search_type="SUMMARIES",
        )

    # ------------------------------------------------------------------
    # CYPHER mode — parameterized read-only Cypher execution
    # ------------------------------------------------------------------

    async def execute_cypher(
        self,
        cypher_template: str,
        params: dict | None = None,
    ) -> list[dict]:
        """Execute a parameterized read-only Cypher query.

        User values are passed only via *params* — never interpolated into the
        template string. The template is validated to reject any f-string style
        ``{variable}`` interpolation markers.

        Args:
            cypher_template: Cypher query with ``$param`` placeholders.
            params: Dict of parameter values bound to ``$param`` references.

        Returns:
            List of row dicts.

        Raises:
            ValueError: If the template contains ``{...}`` interpolation markers
                or forbidden write keywords.
        """
        # Guard: reject f-string style interpolation markers
        if re.search(r"\{[^}]+\}", cypher_template):
            raise ValueError(
                "Cypher template must not contain {var} interpolation — "
                "use $param placeholders and pass values via the params dict"
            )
        # Guard: reject write operations
        forbidden = re.compile(r"\b(CREATE|MERGE|SET|DELETE|DETACH|REMOVE|DROP)\b", re.IGNORECASE)
        if forbidden.search(cypher_template):
            raise ValueError(
                "Only read-only Cypher is permitted. "
                "Detected a write keyword (CREATE/MERGE/SET/DELETE/…)."
            )
        rows = await self.neo._arun(cypher_template, **(params or {}))
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # FEELING_LUCKY mode — heuristic search-type routing
    # ------------------------------------------------------------------

    async def retrieve_feeling_lucky(
        self,
        query: str,
        session_id: str,
        k_entity: int = DEFAULT_K_ENTITY,
        k_semantic: int = DEFAULT_K_SEMANTIC,
        k_final: int = DEFAULT_K_FINAL,
    ) -> RetrievalContext:
        """Auto-route to the most appropriate search mode based on query heuristics.

        Routing rules (evaluated in order):
        1. Query contains Cypher keywords → raise ValueError (force explicit CYPHER).
        2. Short query (< 5 tokens) → GRAPH_TRAVERSAL (likely an entity/concept lookup).
        3. Contains summary triggers (summarize, overview, what is, etc.) → SUMMARIES.
        4. Default → HYBRID.

        Args:
            query: Natural-language query.
            session_id: Session namespace.
            k_entity: Max entity candidates to retrieve per mode.
            k_semantic: Max semantic chunk candidates to retrieve.
            k_final: Max results to return after fusion/reranking.

        Returns:
            RetrievalContext from the selected mode, with search_type set.
        """
        cypher_pat = re.compile(r"\b(MATCH|RETURN|WHERE|WITH|UNWIND)\b", re.IGNORECASE)
        if cypher_pat.search(query):
            raise ValueError(
                "Query looks like Cypher syntax — use search_type='CYPHER' explicitly "
                "and pass your query as cypher_template with a params dict."
            )

        tokens = query.strip().split()
        query_lower = query.lower()

        summary_triggers = {
            "summarize",
            "summary",
            "overview",
            "what is",
            "describe",
            "explain",
            "tell me about",
            "who is",
        }

        if len(tokens) < 5:
            ctx = await self.retrieve_graph_traversal(
                query, session_id, k_entity=k_entity, depth=2, k_final=k_final
            )
            ctx.search_type = "FEELING_LUCKY→GRAPH_TRAVERSAL"
            return ctx

        if any(trigger in query_lower for trigger in summary_triggers):
            ctx = await self.retrieve_summaries(query, session_id, k_entity=k_entity)
            ctx.search_type = "FEELING_LUCKY→SUMMARIES"
            return ctx

        ctx = await self.retrieve(
            query,
            session_id,
            top_k=k_final,
            k_entity=k_entity,
            k_semantic=k_semantic,
        )
        ctx.search_type = "FEELING_LUCKY→HYBRID"
        return ctx


def _assert_base() -> None:
    assert issubclass(HybridRetriever, BaseGraphRetriever)


_assert_base()

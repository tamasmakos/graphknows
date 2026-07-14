"""Personalized-PageRank retrieval channel.

Seeds a restart vector on the query's entities and propagates it over the LTM
entity relation graph, then surfaces the chunks of the highest-mass entities.

This reaches answers that are a *different* entity connected to a query entity
through a relation (``Caroline -LIVES_IN-> Sweden``) — exactly the graph-shaped
questions that plain entity traversal misses. Because PPR concentrates its mass
on genuinely-connected nodes and restarts to the seeds, ubiquitous nodes do not
flood the result (unlike seeding traversal directly on the ever-present speakers).

The per-conversation graph is tiny (~150 entities), so PPR is a few milliseconds
of numpy per query — no native graph-algorithm call needed. Read-only: it uses
the existing ENTITY / relation / MENTIONS structure and adds no schema.
"""

from __future__ import annotations

import logging
from typing import Any

from graphknows.channels.base import Channel, ChannelContext, CollectResult
from graphknows.ranking.rrf import rrf_score

log = logging.getLogger(__name__)

_DAMPING = 0.85
_ITERATIONS = 40
_TOP_ENTITIES = 12  # entities whose chunks are surfaced
_MIN_GRAPH_NODES = 3

# Graph-embedding seed expansion (WS4): pulls in entities structurally close to
# a token-matched seed (via ENTITY.graph_embedding cosine) but missed by name
# overlap — e.g. a query mentioning "Caroline" also seeding her tightly-linked
# but differently-named collaborators. Capped so a broad query (many token
# matches) can't turn into an unbounded number of embedding scans.
_EMBED_SEED_MIN_SIM = 0.7
_EMBED_SEED_EXPAND_TOP_K = 3
_EMBED_SEED_MAX_QUERIES = 5


class GraphPPRChannel(Channel):
    """Personalized PageRank over the entity relation graph, seeded on the query."""

    name = "ppr"

    async def collect(self, ctx: ChannelContext, rt: Any, top_k: int) -> CollectResult:
        """Seed PPR on the query's entities and return the top entities' chunks."""
        words = {w for w in ctx.words_graph if w}
        if not words:
            return {}
        names, edges = await self._relation_graph(rt)
        if len(names) < _MIN_GRAPH_NODES:
            return {}
        seeds = _seed_indices(names, words)
        if not seeds:
            return {}
        seeds = await self._expand_seeds_by_embedding(rt, names, seeds)
        scores = _personalized_pagerank(len(names), edges, seeds)
        ranked = sorted(range(len(names)), key=lambda i: scores[i], reverse=True)
        top = {names[i]: scores[i] for i in ranked[:_TOP_ENTITIES] if scores[i] > 0.0}
        if not top:
            return {}
        return await self._chunks_for_entities(rt, ctx.session_id, ctx.state, top, top_k)

    @staticmethod
    async def _relation_graph(rt: Any) -> tuple[list[str], list[tuple[int, int, float]]]:
        """Directed weighted ENTITY->ENTITY graph from semantic (non-structural) edges."""
        rows = await rt.store.entity_relation_graph()
        idx: dict[str, int] = {}
        edges: list[tuple[int, int, float]] = []
        for row in rows:
            s, t = row.get("src"), row.get("dst")
            if not s or not t:
                continue
            si = idx.setdefault(s, len(idx))
            ti = idx.setdefault(t, len(idx))
            edges.append((si, ti, float(row.get("weight") or 1.0)))
        names = [""] * len(idx)
        for name, i in idx.items():
            names[i] = name
        return names, edges

    @staticmethod
    async def _expand_seeds_by_embedding(rt: Any, names: list[str], seeds: list[int]) -> list[int]:
        """Add graph-embedding-similar entities to the seed set (best-effort, WS4).

        ``ENTITY.graph_embedding`` (native node2vec, written at consolidation)
        surfaces entities structurally close to a token-matched seed but missed
        by name overlap. Soft-fails to the original seeds — no embeddings yet,
        or the similarity lookup erroring, is a clean no-op.
        """
        name_idx = {n: i for i, n in enumerate(names)}
        expanded = set(seeds)
        for i in seeds[:_EMBED_SEED_MAX_QUERIES]:
            try:
                similar = await rt.store.similar_entities_by_embedding(
                    names[i], top_k=_EMBED_SEED_EXPAND_TOP_K
                )
            except Exception:
                continue
            for row in similar:
                if float(row.get("similarity") or 0.0) < _EMBED_SEED_MIN_SIM:
                    continue
                j = name_idx.get(row.get("name"))
                if j is not None:
                    expanded.add(j)
        return list(expanded)

    @staticmethod
    async def _chunks_for_entities(
        rt: Any, session_id: str, state: str | None, entity_scores: dict[str, float], top_k: int
    ) -> CollectResult:
        """Chunks mentioning the top-PPR entities, ranked by summed PPR mass.

        One entity lookup per top entity (the per-conversation graph is tiny);
        each chunk accumulates the PPR mass of every top entity it mentions.
        """
        mass: dict[str, float] = {}
        texts: dict[str, str] = {}
        for name, score in entity_scores.items():
            rows = await rt.store.query_entity_chunks([name], session_id or None, top_k, state)
            for r in rows:
                cid = r["chunk_id"]
                mass[cid] = mass.get(cid, 0.0) + score
                texts.setdefault(cid, r.get("text", ""))
        ranked = sorted(mass.items(), key=lambda x: -x[1])[:top_k]
        result: CollectResult = {}
        for rank, (cid, m) in enumerate(ranked):
            result[cid] = (
                rrf_score(rank, boost=2.0),
                {
                    "text": texts.get(cid, ""),
                    "entities": [],
                    "sources": {"ppr"},
                    "doc_id": "",
                    "metadata": {"ppr_mass": round(m, 6)},
                },
            )
        return result


def _seed_indices(names: list[str], words: set[str]) -> list[int]:
    """Graph nodes whose name overlaps a query token (case-insensitive)."""
    seeds: list[int] = []
    for i, name in enumerate(names):
        toks = {t for t in name.lower().replace("'", " ").split() if len(t) >= 3}
        if (toks & words) or name.lower() in words:
            seeds.append(i)
    return seeds


def _personalized_pagerank(
    n: int, edges: list[tuple[int, int, float]], seeds: list[int]
) -> list[float]:
    """Power-iteration PPR with restart to the seed set. Pure numpy, tiny graphs."""
    import numpy as np

    # Column-stochastic transition: trans[j, i] = P(step from i to j).
    trans = np.zeros((n, n), dtype=float)
    for si, ti, w in edges:
        trans[ti, si] += w
    col = trans.sum(axis=0)
    nonzero = col > 0
    trans[:, nonzero] /= col[nonzero]
    dangling_mask = ~nonzero

    restart = np.zeros(n, dtype=float)
    for s in seeds:
        restart[s] = 1.0
    restart /= restart.sum()

    rank = restart.copy()
    for _ in range(_ITERATIONS):
        # Mass on out-edgeless (dangling) nodes would vanish; redistribute it to
        # the restart set — the personalized analogue of PageRank's dangling term.
        dangling = float(rank[dangling_mask].sum()) if dangling_mask.any() else 0.0
        rank = _DAMPING * (trans @ rank + dangling * restart) + (1.0 - _DAMPING) * restart
    return rank.tolist()


__all__ = ["GraphPPRChannel"]

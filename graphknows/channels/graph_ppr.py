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

# Edge types carrying no semantic entity->entity meaning (structure/plumbing).
_STRUCTURAL_TYPES = frozenset(
    {
        "MENTIONS", "HAS_CHUNK", "HAS_ENTITY", "HAS_FILE", "HAS_TURN", "IN_SESSION",
        "IN_TOPIC", "IN_COMMUNITY", "IN_DIALOG", "PART_OF", "NEXT", "APPEARED_IN",
        "CONTAINS_DOC", "MENTIONS_DATE", "HAPPENED_ON",
    }
)


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
        scores = _personalized_pagerank(len(names), edges, seeds)
        ranked = sorted(range(len(names)), key=lambda i: scores[i], reverse=True)
        top = {names[i]: scores[i] for i in ranked[:_TOP_ENTITIES] if scores[i] > 0.0}
        if not top:
            return {}
        return await self._chunks_for_entities(rt, ctx.session_id, top, top_k)

    @staticmethod
    async def _relation_graph(rt: Any) -> tuple[list[str], list[tuple[int, int, float]]]:
        """Directed weighted ENTITY->ENTITY graph from semantic (non-structural) edges."""
        skip = ", ".join(f"'{t}'" for t in sorted(_STRUCTURAL_TYPES))
        rows = await rt.ltm._qry(
            "MATCH (a:ENTITY)-[r]->(b:ENTITY) "
            "WHERE a.name IS NOT NULL AND b.name IS NOT NULL AND a.name <> b.name "
            f"AND NOT type(r) IN [{skip}] "
            "AND (r.is_current = true OR r.is_current IS NULL) "
            "RETURN a.name AS src, b.name AS tgt, coalesce(r.weight, 1.0) AS w"
        )
        idx: dict[str, int] = {}
        edges: list[tuple[int, int, float]] = []
        for row in rows:
            s, t = row.get("src"), row.get("tgt")
            if not s or not t:
                continue
            si = idx.setdefault(s, len(idx))
            ti = idx.setdefault(t, len(idx))
            edges.append((si, ti, float(row.get("w") or 1.0)))
        names = [""] * len(idx)
        for name, i in idx.items():
            names[i] = name
        return names, edges

    @staticmethod
    async def _chunks_for_entities(
        rt: Any, session_id: str, entity_scores: dict[str, float], top_k: int
    ) -> CollectResult:
        """Chunks mentioning the top-PPR entities, ranked by summed PPR mass."""
        safe = {n.replace("'", ""): sc for n, sc in entity_scores.items()}
        where = " OR ".join(f"e.name = '{n}'" for n in safe)
        if not where:
            return {}
        if session_id:
            rows = await rt.ltm._qry(
                "MATCH (sess:SESSION {id: $sid})-[:IN_SESSION]->(:FILE)"
                "-[:HAS_CHUNK]->(c:CHUNK)-[:MENTIONS]->(e:ENTITY) "
                f"WHERE {where} "
                "WITH c, collect(DISTINCT e.name) AS entities "
                "RETURN c.id AS chunk_id, c.text AS text, entities",
                sid=session_id,
            )
        else:
            rows = await rt.ltm._qry(
                "MATCH (c:CHUNK)-[:MENTIONS]->(e:ENTITY) "
                f"WHERE {where} "
                "WITH c, collect(DISTINCT e.name) AS entities "
                "RETURN c.id AS chunk_id, c.text AS text, entities"
            )
        scored: list[tuple[float, dict[str, Any]]] = []
        for row in rows:
            ents = row.get("entities") or []
            mass = sum(safe.get(e, 0.0) for e in ents)
            if mass > 0.0:
                scored.append((mass, row))
        scored.sort(key=lambda x: x[0], reverse=True)

        result: CollectResult = {}
        for rank, (mass, row) in enumerate(scored[:top_k]):
            result[row["chunk_id"]] = (
                rrf_score(rank, boost=2.0),
                {
                    "text": row.get("text", ""),
                    "entities": row.get("entities", []),
                    "sources": {"ppr"},
                    "doc_id": "",
                    "metadata": {"ppr_mass": round(mass, 6)},
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

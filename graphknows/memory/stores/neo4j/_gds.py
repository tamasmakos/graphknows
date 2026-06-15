"""GDS helper functions: silhouette scoring and pgvector topic writes."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

logger = logging.getLogger(__name__)


def _compute_silhouette(
    names: list[str],
    name_idx: dict[str, int],
    communities: dict[int, list[str]],
    rows: list[dict[str, Any]],
    weights: list[float],
) -> float | None:
    """Compute silhouette score from weighted edge co-occurrence data.

    Uses a distance matrix: d(i,j) = 1 - weight/max_weight. Returns None on
    any failure (missing optional deps, single community, etc.).
    """
    try:
        import numpy as np
        from sklearn.metrics import silhouette_score

        labels = [0] * len(names)
        for comm_id, members in communities.items():
            for m in members:
                labels[name_idx[m]] = comm_id
        if len(set(labels)) < 2:
            return None
        max_w = max(weights) if weights else 1.0
        adj = [[1.0] * len(names) for _ in range(len(names))]
        for edge_data, w in zip(rows, weights, strict=False):
            i, j = name_idx[edge_data["a"]], name_idx[edge_data["b"]]
            d = 1.0 - w / max_w
            adj[i][j] = d
            adj[j][i] = d
            adj[i][i] = 0.0
            adj[j][j] = 0.0
        return float(silhouette_score(np.array(adj), labels, metric="precomputed", random_state=42))
    except Exception as exc:
        logger.warning("Silhouette computation skipped: %s", exc)
        return None


async def _write_pgvector_topic(
    pg_store: Any,
    topic: dict[str, Any],
    *,
    title: str,
    summary: str,
    comm_id: int,
) -> None:
    """Write a topic row to pgvector, embedding summary/title text when available."""
    embedding_text = (summary or "").strip() or (title or "").strip()
    if embedding_text:
        try:
            from graphknows.memory.stores.embedder import embed_one

            topic["embedding"] = await asyncio.to_thread(embed_one, embedding_text)
        except Exception as exc:
            logger.warning("Topic embedding failed for community %d: %s", comm_id, exc)

    await pg_store.write_ltm_topic(topic)

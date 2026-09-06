"""MCP tool: memory_topics — rebuild the topic layer without re-ingesting."""

from __future__ import annotations

from typing import Any

from graphknows.server.mcp._app import app
from graphknows.server.mcp._state import get_runtime


@app.tool()
async def memory_topics(
    mode: str = "",
    min_size: int = 2,
    persist: bool = True,
    export_path: str = "",
) -> dict[str, Any]:
    """Rebuild the topic layer over the already-ingested graph.

    Re-runs clustering, description and grading only — no ingestion, entity
    resolution, PageRank or consolidation — so topics can be swept at different
    settings over an unchanged graph.

    Topics are the Louvain partition, which covers near-every entity but offers
    no granularity knob: ArcadeDB's Louvain ignores its parameters. ``min_size``
    is therefore the only setting that changes the shape of the output.

    Args:
        mode: ``extractive`` (deterministic, no LLM), ``llm``, or ``none``.
            Empty uses ``GRAPHKNOWS_TOPICS``.
        min_size: Clusters smaller than this are dropped as stragglers.
        persist: Write the TOPIC layer, replacing the previous build.
        export_path: Optional path for a read-only JSON artifact.

    Returns:
        Build metrics — ``topics``, ``newman_modularity``, ``mean_coherence``,
        ``chunk_coverage``, ``entity_coverage``, ``largest_topic`` — plus
        ``persisted``. Tune against ``newman_modularity`` and
        ``mean_coherence``; the reported ``arcadedb_modularity_untrustworthy``
        rewards coarser partitions and must not be optimised.
    """
    from graphknows.settings import TopicMode
    from graphknows.topics import TopicConfig

    runtime = await get_runtime()
    resolved = TopicMode(mode) if mode else runtime.settings.topic_mode
    config = TopicConfig(min_size=max(1, min_size), mode=resolved)
    return await runtime.build_topics(  # type: ignore[no-any-return]
        config=config, persist=persist, export_path=export_path
    )

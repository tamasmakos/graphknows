"""LTMWriter — pure Neo4j graph write (MERGE nodes and edges).

Delegates to the Neo4jStore.write_session_subgraph for actual graph write
semantics.  This wrapper provides the canonical LTM write interface so that
flush.py and other callers are decoupled from the store implementation.

# TODO(Wave 3): replace neo4j store direct usage with IGraphDb adapter once #37 merges.
"""

from __future__ import annotations

import logging
from typing import Any

log = logging.getLogger("graphknows.memory.ltm.writer")


class LTMWriter:
    """Write STM subgraph data into Neo4j LTM with MERGE semantics.

    All methods are static or class-level to keep this class stateless;
    callers pass the store (or driver) explicitly.
    """

    @staticmethod
    async def write(
        session_id: str,
        stm_data: dict[str, Any],
        neo_store: Any,
        errors: list[str] | None = None,
    ) -> dict[str, int]:
        """MERGE all STM nodes/edges into Neo4j LTM.

        Args:
            session_id: Session being promoted.
            stm_data: STM subgraph dict with keys: entities, files, chunks, dialogs, turns.
            neo_store: Neo4jStore (or compatible) — must expose write_session_subgraph
                       and extract_and_write_relations coroutines.
            errors: Optional mutable list; non-fatal errors are appended here.

        Returns:
            dict with ``nodes`` and ``edges`` counts.
        """
        nodes = 0
        edges = 0
        try:
            result = await neo_store.write_session_subgraph(session_id, stm_data)
            nodes = result.get("nodes", 0)
            edges = result.get("edges", 0)
        except Exception as exc:
            log.warning("LTMWriter.write %s: %s", session_id, exc)
            if errors is not None:
                errors.append(f"write_ltm_graph: {exc}")

        # Relation extraction (fail-hard per spec — no try/except here)
        rel_edges = await neo_store.extract_and_write_relations(
            stm_data=stm_data,
            openrouter_key="",
        )
        edges += rel_edges
        return {"nodes": nodes, "edges": edges}

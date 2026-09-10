"""MCP tool: memory_recall — facts with their evidence, never bare segments."""

from __future__ import annotations

from typing import Any

from graphknows.models.report import RecallBudget
from graphknows.server.mcp._app import app
from graphknows.server.mcp._bounded import bounded
from graphknows.server.mcp._state import get_runtime


@app.tool()
@bounded
async def memory_recall(query: str, budget: dict[str, Any] | None = None) -> dict[str, Any]:
    """Recall the facts *query*'s symbols activate, with the evidence asserting them.

    A thin adapter over :meth:`graphknows.Memory.recall` — the same call an
    in-process caller makes, so this transport cannot drift from it.

    Args:
        query: Natural-language question or search phrase.
        budget: Optional explicit bound on the slice (FR-010) — ``max_facts``
            and ``max_evidence_per_fact``.

    Returns:
        The wire form of :class:`~graphknows.models.report.RecallResult`:
        ``facts`` (each with its ``source_uri`` / ``byte_range`` / ``text``
        evidence), ``no_evidence``, ``budget``, ``truncated_by`` and
        ``counters``. An empty slice says ``no_evidence`` rather than
        presenting itself as a successful empty answer (FR-015).
    """
    runtime = await get_runtime()
    result = await runtime.recall(query, RecallBudget(**budget) if budget else None)
    return result.model_dump(mode="json")

"""EU source router — derives source IDs from the archive registry.

Rather than maintaining a parallel keyword table here, we read query_strengths
directly from the canonical ArchiveRegistry.  Adding a new connector only
requires a new ArchiveEntry in archive_registry.py — no changes here.

Routing logic (in priority order):
  1. For each registry entry, check whether any query token appears in any of
     its query_strengths phrases.
  2. Score each entry by the number of matching strength phrases.
  3. Return the top-scored entry IDs, falling back to defaults if none match.
"""

from __future__ import annotations

import sys
from pathlib import Path

# The archive_registry lives in the search service package.  We import it
# directly (same process on the agent side); the agent Dockerfile includes
# the search package on PYTHONPATH.
_SEARCH_PKG = Path(__file__).resolve().parents[3] / "search"
if str(_SEARCH_PKG.parent) not in sys.path:
    sys.path.insert(0, str(_SEARCH_PKG.parent))

from search.archive_registry import REGISTRY

_DEFAULT_SOURCES = ["eprs_briefings", "ep_resolutions", "ep_debates"]


def route_sources(query: str, max_sources: int = 3) -> list[str]:
    """Return up to *max_sources* adapter IDs best suited to *query*.

    Scoring: number of query_strengths phrases from the registry entry that
    have at least one token appearing in the query.  Ties are broken by
    registry order (higher entries are preferred).
    """
    if not query or not query.strip():
        return _DEFAULT_SOURCES[:max_sources]

    q_lower = query.lower()

    scores: list[tuple[int, str]] = []
    for entry in REGISTRY:
        score = sum(
            1 for phrase in entry.query_strengths
            if any(tok in q_lower for tok in phrase.lower().split())
        )
        if score > 0:
            scores.append((score, entry.id))

    # Sort highest-score first; stable sort preserves registry order on ties.
    scores.sort(key=lambda t: -t[0])

    selected = [entry_id for _, entry_id in scores]
    if not selected:
        selected = list(_DEFAULT_SOURCES)

    return selected[:max_sources]

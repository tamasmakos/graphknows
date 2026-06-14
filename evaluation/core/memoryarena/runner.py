"""MemoryArena research-memory helper functions."""

from __future__ import annotations

import re
from typing import Any


def _compact_answer(answer: Any) -> str:
    """Render a MemoryArena answer into compact text."""
    if isinstance(answer, dict):
        return "; ".join(f"{key}: {value}" for key, value in sorted(answer.items()))
    if isinstance(answer, list):
        return "; ".join(str(item) for item in answer)
    return str(answer)


def _expected_prior_terms(answer: Any) -> tuple[str, ...]:
    """Extract stable scoring terms from a prior MemoryArena answer."""
    text = _compact_answer(answer)
    terms = re.findall(r"[A-Za-z0-9][A-Za-z0-9_-]{2,}", text)
    return tuple(dict.fromkeys(terms))


async def fetch_memoryarena_rows(
    domain: str,
    offset: int,
    limit: int,
) -> tuple[list[dict[str, Any]], int]:
    """Return MemoryArena rows from local data when available.

    CI unit tests patch higher-level case construction, so the default smoke
    implementation returns no rows rather than performing network access.
    """
    _ = domain
    _ = offset
    _ = limit
    return [], 0

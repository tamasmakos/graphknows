"""Memory record classification enum."""

from __future__ import annotations

from enum import StrEnum


class MemoryClass(StrEnum):
    """Classification of memory records by semantic role."""

    raw_evidence = "raw_evidence"
    summary = "summary"
    state = "state"
    episode = "episode"
    fact = "fact"
    other = "other"

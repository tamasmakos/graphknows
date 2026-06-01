"""Shared scoring helpers for STM and LTM recall pipelines."""

from __future__ import annotations

RRF_K: int = 60

_SHAPE_FACTORS: dict[str | None, float] = {
    "fact": 1.5,
    "state": 1.4,
    "episode": 1.3,
    "summary": 1.2,
    "raw_evidence": 0.6,
}


def _shape_score(score: float, memory_class: str | None) -> float:
    """Apply memory_class-aware multiplicative boost/de-boost to a score."""
    return score * _SHAPE_FACTORS.get(memory_class, 1.0)

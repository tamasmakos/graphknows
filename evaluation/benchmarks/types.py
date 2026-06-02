"""Shared benchmark result types."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class BenchmarkCaseResult:
    """One benchmark evaluation case."""

    case_id: str
    question: str
    gold: Any
    memory_context: list[dict[str, Any]]
    metadata: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    score: float | None = None

    def to_record(self) -> dict[str, Any]:
        """Serialize into a JSON-friendly dictionary."""
        return asdict(self)

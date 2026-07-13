"""Promotion step contract + shared context.

STM->LTM promotion is an ordered list of :class:`PromotionStep` objects run over
one mutable :class:`PromotionContext`. To insert a stage between STM and LTM,
write a step and add it to the list in ``build_promotion_pipeline`` — no existing
step changes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from graphknows.ingestion.models.promotion import (
    PromotionPolicy,
    PromotionRecord,
    PromotionRequest,
)


@dataclass
class PromotionContext:
    """Mutable state threaded through every promotion step."""

    request: PromotionRequest
    policy: PromotionPolicy
    record: PromotionRecord
    stm: Any
    vec: Any
    ltm: Any
    channels: list[Any] = field(default_factory=list)
    llm_api_key: str = ""
    llm_api_base: str = ""
    llm_model: str = ""
    # Filled by the read/graph steps and consumed downstream.
    stm_data: dict[str, Any] = field(default_factory=dict)
    gds_result: Any = None


@runtime_checkable
class PromotionStep(Protocol):
    """One promotion stage. ``name`` labels it in logs and error records."""

    name: str

    async def run(self, ctx: PromotionContext) -> None:
        """Advance the promotion using and mutating ``ctx``."""
        ...


__all__ = ["PromotionContext", "PromotionStep"]

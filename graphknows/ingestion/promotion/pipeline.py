"""LTMPromotionPipeline — run an ordered list of promotion steps STM->LTM."""

from __future__ import annotations

import contextlib
import logging
import time
from typing import Any

from graphknows.ingestion.models.promotion import PromotionRecord, PromotionRequest
from graphknows.ingestion.promotion.base import PromotionContext, PromotionStep
from graphknows.ingestion.promotion.steps import (
    ChannelPromotionsStep,
    ClearStmStep,
    CommunityDetectionStep,
    CommunityMetadataStep,
    CopyVectorStep,
    ExtractRelationsStep,
    GraphPruningStep,
    IndexEntitiesStep,
    PagerankStep,
    ReadStmStep,
    ResolveEntitiesStep,
    TopicSummaryStep,
    WriteLtmGraphStep,
)

log = logging.getLogger("graphknows.ingestion.promotion")


def default_promotion_steps(extraction_uses_llm: bool) -> list[PromotionStep]:
    """Assemble the default promotion step list for a mode.

    llm_assisted adds relation extraction, community detection, topic summary, and
    community-metadata write-back; llm_free omits them (no retrieved context in
    that mode, only wasted ingest time). Callers can build their own list and pass
    it to :class:`LTMPromotionPipeline` to insert or reorder stages.
    """
    steps: list[PromotionStep] = [ReadStmStep(), ResolveEntitiesStep(), WriteLtmGraphStep()]
    if extraction_uses_llm:
        steps.append(ExtractRelationsStep())
    steps += [CopyVectorStep(), IndexEntitiesStep(), ChannelPromotionsStep(), PagerankStep(), GraphPruningStep()]
    if extraction_uses_llm:
        steps += [CommunityDetectionStep(), TopicSummaryStep(), CommunityMetadataStep()]
    steps.append(ClearStmStep())
    return steps


class LTMPromotionPipeline:
    """Promote a session's STM to LTM by running ``steps`` in order.

    When ``steps`` is not given, the default mode-appropriate list is assembled
    from ``extraction_uses_llm``. Inserting a stage between STM and LTM is a list
    edit — build your own via :func:`default_promotion_steps`, splice, and pass it.
    Each step encapsulates its own soft-fail behavior and records non-fatal errors
    onto the ``PromotionRecord``.
    """

    def __init__(
        self,
        stm_store: Any,
        vec_store: Any,
        ltm_store: Any,
        steps: list[PromotionStep] | None = None,
        *,
        extraction_uses_llm: bool = False,
        llm_api_key: str = "",
        llm_api_base: str = "",
        llm_model: str = "",
        channels: list[Any] | None = None,
    ) -> None:
        self._stm = stm_store
        self._vec = vec_store
        self._ltm = ltm_store
        self._steps = steps if steps is not None else default_promotion_steps(extraction_uses_llm)
        self._extraction_uses_llm = extraction_uses_llm
        self._llm_api_key = llm_api_key
        self._llm_api_base = llm_api_base
        self._llm_model = llm_model
        self._channels = channels or []

    async def close(self) -> None:
        """Release all store connections."""
        for store in (self._vec, self._stm, self._ltm):
            with contextlib.suppress(Exception):
                await store.close()

    async def promote(self, request: PromotionRequest) -> PromotionRecord:
        """Run every step over a shared context and return the record."""
        t0 = time.monotonic()
        record = PromotionRecord(session_id=request.session_id)
        policy = request.resolve_policy(self._extraction_uses_llm)
        record.topic_mode = policy.topic_mode

        for store in (self._vec, self._stm, self._ltm):
            await store.connect()

        ctx = PromotionContext(
            request=request,
            policy=policy,
            record=record,
            stm=self._stm,
            vec=self._vec,
            ltm=self._ltm,
            channels=self._channels,
            llm_api_key=self._llm_api_key,
            llm_api_base=self._llm_api_base,
            llm_model=self._llm_model,
        )

        for step in self._steps:
            try:
                await step.run(ctx)
            except Exception as exc:  # last-resort guard; steps handle their own errors
                log.warning("promotion step %s failed (non-fatal): %s", step.name, exc)
                record.errors.append(f"{step.name}: {exc}")

        record.elapsed_s = time.monotonic() - t0
        log.info(
            "promote session=%s nodes=%d edges=%d chunks=%d turns=%d topics=%d elapsed=%.1fs",
            request.session_id,
            record.nodes_written,
            record.edges_written,
            record.chunks_promoted,
            record.turns_promoted,
            record.topics_created,
            record.elapsed_s,
        )
        return record


__all__ = ["LTMPromotionPipeline", "default_promotion_steps"]

"""Extensibility proofs for the two plug-in axes of the 2.0 architecture.

1. A promotion step spliced between two defaults runs in order.
2. A channel registered once is usable on BOTH the write (promote) and read
   (collect) paths — the symmetric-channel guarantee.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from graphknows.channels.base import Channel, ChannelContext, ChannelStores
from graphknows.channels.registry import CHANNEL_REGISTRY, default_channels, register
from graphknows.ingestion.models.promotion import PromotionRequest
from graphknows.ingestion.promotion.base import PromotionContext
from graphknows.ingestion.promotion.pipeline import (
    LTMPromotionPipeline,
    default_promotion_steps,
)
from graphknows.settings import GraphKnowsSettings, MemoryMode


@pytest.mark.asyncio
async def test_promotion_step_can_be_inserted_between_defaults() -> None:
    """A custom step spliced into the default list runs at its position."""
    executed: list[str] = []

    class RecordingStep:
        name = "recording"

        async def run(self, ctx: PromotionContext) -> None:
            executed.append(self.name)

    # Wrap every default step so we can observe ordering, then splice ours in
    # right after read_stm (i.e. "a step between STM and LTM").
    base = default_promotion_steps(extraction_uses_llm=False)

    class Traced:
        def __init__(self, inner: object) -> None:
            self._inner = inner
            self.name = inner.name

        async def run(self, ctx: PromotionContext) -> None:
            executed.append(self.name)
            await self._inner.run(ctx)

    steps = [Traced(base[0]), RecordingStep(), *[Traced(s) for s in base[1:]]]

    stores = AsyncMock()
    stores.read_session_subgraph.return_value = {
        "entities": [], "files": [], "chunks": [], "dialogs": [], "turns": []
    }
    stores.write_session_subgraph.return_value = {"nodes": 0, "edges": 0}
    stores.promote_to_ltm.return_value = (0, 0)
    stores.compute_and_persist_pagerank.return_value = 0
    stores.prune_tombstoned_edges.return_value = 0
    stores.prune_duplicate_edges.return_value = 0

    pipeline = LTMPromotionPipeline(stores, stores, stores, steps=steps)
    await pipeline.promote(PromotionRequest(session_id="s1"))

    assert "recording" in executed
    # recording runs after read_stm and before write_ltm_graph
    assert executed.index("read_stm") < executed.index("recording")
    assert executed.index("recording") < executed.index("write_ltm_graph")


@pytest.mark.asyncio
async def test_registered_channel_serves_both_write_and_read() -> None:
    """Registering one channel makes it active on ingestion AND retrieval."""
    calls: list[str] = []

    class ToyChannel(Channel):
        name = "toy_metadata"

        async def promote(self, rt, session_id, record) -> None:
            calls.append("promote")

        async def collect(self, ctx, rt, top_k):
            calls.append("collect")
            return {}

    try:
        register("toy_metadata", ToyChannel)
        assert "toy_metadata" in CHANNEL_REGISTRY

        settings = GraphKnowsSettings(mode=MemoryMode.llm_free)
        active = default_channels(settings)
        toy = next(c for c in active if c.name == "toy_metadata")

        # Same instance's write hook (flush) and read hook (search) both fire.
        await toy.promote(ChannelStores(), "s1", record=None)
        ctx = ChannelContext("q", "s1", "both", [], None, [], [])
        await toy.collect(ctx, rt=None, top_k=5)

        assert calls == ["promote", "collect"]
    finally:
        CHANNEL_REGISTRY.pop("toy_metadata", None)

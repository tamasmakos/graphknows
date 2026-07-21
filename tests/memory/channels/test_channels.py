"""Unit tests for the modular channel architecture + FrameChannel."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from graphknows.settings import GraphKnowsSettings, MemoryMode
from graphknows.channels.registry import default_channels as build_channels
from graphknows.channels.base import Channel, ChannelContext
from graphknows.channels.frames import FrameChannel


def _ctx(**parse):
    ctx = ChannelContext(
        query="q",
        session_id="s",
        scope="both",
        emb=[0.0],
        emb2=None,
        words=["a"],
        words_graph=["a"],
    )
    if parse:
        ctx._frame_parse = parse  # pre-seed to avoid loading spaCy/FrameNet
    return ctx


class TestBuildChannels:
    def test_default_channels_no_frames(self):
        s = GraphKnowsSettings(GRAPHKNOWS_ENABLE_FRAMES=False, GRAPHKNOWS_MODE=MemoryMode.llm_free)
        names = [c.name for c in build_channels(s)]
        assert names == [
            "entity",
            "vector",
            "bm25",
            "temporal",
            "topic",
            "ppr",
        ]
        assert "frame" not in names

    def test_frames_appended_when_enabled(self):
        s = GraphKnowsSettings(GRAPHKNOWS_ENABLE_FRAMES=True, GRAPHKNOWS_MODE=MemoryMode.llm_free)
        assert [c.name for c in build_channels(s)][-1] == "frame"

    def test_enabled_selects_named_subset(self):
        s = GraphKnowsSettings(GRAPHKNOWS_ENABLE_FRAMES=False, GRAPHKNOWS_MODE=MemoryMode.llm_free)
        names = [c.name for c in build_channels(s, enabled=["vector", "bm25"])]
        assert names == ["vector", "bm25"]

    def test_enabled_unknown_name_raises(self):
        s = GraphKnowsSettings(GRAPHKNOWS_ENABLE_FRAMES=False, GRAPHKNOWS_MODE=MemoryMode.llm_free)
        with pytest.raises(KeyError):
            build_channels(s, enabled=["does_not_exist"])


class TestChannelBaseDefaults:
    async def test_defaults_are_noops(self):
        ch = Channel()
        assert await ch.collect(_ctx(), MagicMock(), 5) == {}
        assert await ch.ingest(MagicMock(), "s", "c", "t", None) is None


class TestFrameChannelCollect:
    async def test_no_matched_frames_returns_empty(self):
        """No frame vector-matches the query -> return {}."""
        rt = MagicMock()
        rt.store.query_frames_by_embedding = AsyncMock(return_value=[])
        out = await FrameChannel().collect(_ctx(), rt, 5)
        assert out == {}
        # the query side vector-searches the namespace-local FRAME embeddings
        rt.store.query_frames_by_embedding.assert_awaited_once()

    async def test_stronger_query_frame_outranks(self):
        rt = MagicMock()
        rt.store.query = AsyncMock(return_value=[{"c": 10}])  # 10 session chunks
        rt.store.query_frames_by_embedding = AsyncMock(
            return_value=[
                {"chunk_id": "c_strong", "frame_type": "Process_start", "score": 0.42},
                {"chunk_id": "c_weak", "frame_type": "Activity_start", "score": 0.30},
            ]
        )
        out = await FrameChannel().collect(_ctx(), rt, 5)
        assert set(out) == {"c_strong", "c_weak"}
        # chunk evoking the higher-scoring frame ranks first
        assert out["c_strong"][0] > out["c_weak"][0]
        assert out["c_strong"][1]["sources"] == {"frame"}
        # query side vector-searched FRAME.embedding with the query embedding
        args = rt.store.query_frames_by_embedding.await_args.args
        assert args[0] == [0.0]  # ctx.emb

    async def test_saturated_frame_dropped(self):
        """A frame evoked in >50% of session chunks is dropped as non-discriminative."""
        rt = MagicMock()
        rt.store.query = AsyncMock(return_value=[{"c": 4}])  # 4 session chunks
        rt.store.query_frames_by_embedding = AsyncMock(
            return_value=[
                # Activity_ongoing floods 3/4 chunks -> saturated, dropped.
                {"chunk_id": "c1", "frame_type": "Activity_ongoing", "score": 0.35},
                {"chunk_id": "c2", "frame_type": "Activity_ongoing", "score": 0.35},
                {"chunk_id": "c3", "frame_type": "Activity_ongoing", "score": 0.35},
                {"chunk_id": "c4", "frame_type": "Process_start", "score": 0.34},
            ]
        )
        out = await FrameChannel().collect(_ctx(), rt, 5)
        assert set(out) == {"c4"}  # only the discriminative frame's chunk survives

    async def test_empty_store_result_returns_empty(self):
        rt = MagicMock()
        rt.store.query_frames_by_embedding = AsyncMock(return_value=[])
        out = await FrameChannel().collect(_ctx(), rt, 5)
        assert out == {}


pytestmark = pytest.mark.asyncio

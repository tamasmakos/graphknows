"""Unit tests for the modular channel architecture + FrameChannel."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from graphknows.config import GraphKnowsSettings, MemoryMode
from graphknows.config.builders.channels import build_channels
from graphknows.memory.channels.base import Channel, ChannelContext, ChannelStores
from graphknows.memory.channels.frame import FrameChannel


def _ctx(**parse):
    ctx = ChannelContext(
        query="q", session_id="s", scope="both",
        emb=[0.0], emb2=None, words=["a"], words_graph=["a"],
    )
    if parse:
        ctx._frame_parse = parse  # pre-seed to avoid loading spaCy/FrameNet
    return ctx


class TestBuildChannels:
    def test_default_seven_no_frames(self):
        s = GraphKnowsSettings(GRAPHKNOWS_ENABLE_FRAMES=False, GRAPHKNOWS_MODE=MemoryMode.llm_free)
        names = [c.name for c in build_channels(s)]
        assert names == ["stm_entity", "ltm_entity", "vector", "stm_bm25",
                         "ltm_bm25", "temporal", "topic"]
        assert "frame" not in names

    def test_frames_appended_when_enabled(self):
        s = GraphKnowsSettings(GRAPHKNOWS_ENABLE_FRAMES=True, GRAPHKNOWS_MODE=MemoryMode.llm_free)
        assert [c.name for c in build_channels(s)][-1] == "frame"

    def test_mock_settings_do_not_enable_frames_or_empty_list(self):
        """A partial/mock settings must yield the 7 defaults, never [] or frames."""
        mock = MagicMock()  # enable_frames + mode.value are truthy mocks
        names = [c.name for c in build_channels(mock)]
        assert names == ["stm_entity", "ltm_entity", "vector", "stm_bm25",
                         "ltm_bm25", "temporal", "topic"]


class TestChannelBaseDefaults:
    async def test_defaults_are_noops(self):
        ch = Channel()
        assert await ch.collect(_ctx(), MagicMock(), 5) == {}
        assert await ch.ingest(MagicMock(), "s", "c", "t", None) is None
        assert await ch.promote(MagicMock(), "s", None) is None


class TestFrameChannelCollect:
    async def test_abstract_gate_returns_empty_without_touching_store(self):
        """No concrete query frame -> return {} and never query the store."""
        rt = MagicMock()
        rt.neo.query_frame_chunks = AsyncMock()
        out = await FrameChannel().collect(_ctx(q_frames=set(), expanded=set(), anchors=set()), rt, 5)
        assert out == {}
        rt.neo.query_frame_chunks.assert_not_awaited()

    async def test_exact_frame_and_anchor_outrank(self):
        rt = MagicMock()
        rt.neo.query_frame_chunks = AsyncMock(return_value=[
            {"chunk_id": "c_exact_anchor", "frame_type": "Residence", "speaker": "melanie", "fillers": ["beach"]},
            {"chunk_id": "c_expanded", "frame_type": "Temporary_stay", "speaker": "caroline", "fillers": ["x"]},
        ])
        ctx = _ctx(q_frames={"Residence"}, expanded={"Residence", "Temporary_stay"}, anchors={"melanie"})
        out = await FrameChannel().collect(ctx, rt, 5)
        assert set(out) == {"c_exact_anchor", "c_expanded"}
        # exact frame (2.0) x anchor bonus (1.6) must rank above expanded (1.0)
        assert out["c_exact_anchor"][0] > out["c_expanded"][0]
        assert out["c_exact_anchor"][1]["sources"] == {"frame"}

    async def test_empty_store_result_returns_empty(self):
        rt = MagicMock()
        rt.neo.query_frame_chunks = AsyncMock(return_value=[])
        out = await FrameChannel().collect(_ctx(q_frames={"F"}, expanded={"F"}, anchors=set()), rt, 5)
        assert out == {}


class TestFrameChannelPromote:
    async def test_promote_extracts_and_writes(self, monkeypatch):
        rt = ChannelStores(neo=MagicMock())
        rt.neo._qry = AsyncMock(return_value=[{"id": "c1", "text": "2pm | Mel: I went camping in the mountains."}])
        rt.neo.write_frames = AsyncMock(return_value=1)
        # stub extraction so the test stays offline (no spaCy/FrameNet load)
        monkeypatch.setattr(
            "graphknows.extraction.frames.extract_frames_from_chunk",
            lambda text, speakers: [{"frame": "Residence", "lu": "camp.v",
                                     "speaker": "Mel", "fes": [("in", "mountains")]}],
        )
        await FrameChannel().promote(rt, "sess", record=None)
        rt.neo.write_frames.assert_awaited_once()
        session_id, rows = rt.neo.write_frames.await_args.args
        assert session_id == "sess"
        assert rows[0]["frame_type"] == "Residence"
        assert rows[0]["chunk_id"] == "c1"
        assert "mountains" in rows[0]["fillers"] and "Mel" in rows[0]["fillers"]

    async def test_promote_no_chunks_is_noop(self):
        rt = ChannelStores(neo=MagicMock())
        rt.neo._qry = AsyncMock(return_value=[])
        rt.neo.write_frames = AsyncMock()
        await FrameChannel().promote(rt, "sess", record=None)
        rt.neo.write_frames.assert_not_awaited()


pytestmark = pytest.mark.asyncio

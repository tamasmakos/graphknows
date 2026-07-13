"""Unit tests for the modular channel architecture + FrameChannel."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from graphknows.settings import GraphKnowsSettings, MemoryMode
from graphknows.channels.registry import default_channels as build_channels
from graphknows.channels.base import Channel, ChannelContext, ChannelStores
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
            "stm_entity",
            "ltm_entity",
            "vector",
            "stm_bm25",
            "ltm_bm25",
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
        names = [c.name for c in build_channels(s, enabled=["vector", "stm_bm25"])]
        assert names == ["vector", "stm_bm25"]

    def test_enabled_unknown_name_raises(self):
        s = GraphKnowsSettings(GRAPHKNOWS_ENABLE_FRAMES=False, GRAPHKNOWS_MODE=MemoryMode.llm_free)
        with pytest.raises(KeyError):
            build_channels(s, enabled=["does_not_exist"])


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
        rt.ltm.query_frame_chunks = AsyncMock()
        out = await FrameChannel().collect(
            _ctx(q_frames=set(), expanded=set(), anchors=set()), rt, 5
        )
        assert out == {}
        rt.ltm.query_frame_chunks.assert_not_awaited()

    async def test_exact_frame_and_anchor_outrank(self):
        rt = MagicMock()
        rt.ltm.query_frame_chunks = AsyncMock(
            return_value=[
                {
                    "chunk_id": "c_exact_anchor",
                    "frame_type": "Residence",
                    "speaker": "melanie",
                    "fillers": ["beach"],
                },
                {
                    "chunk_id": "c_expanded",
                    "frame_type": "Temporary_stay",
                    "speaker": "caroline",
                    "fillers": ["x"],
                },
            ]
        )
        ctx = _ctx(
            q_frames={"Residence"}, expanded={"Residence", "Temporary_stay"}, anchors={"melanie"}
        )
        out = await FrameChannel().collect(ctx, rt, 5)
        assert set(out) == {"c_exact_anchor", "c_expanded"}
        # exact frame (2.0) x anchor bonus (1.6) must rank above expanded (1.0)
        assert out["c_exact_anchor"][0] > out["c_expanded"][0]
        assert out["c_exact_anchor"][1]["sources"] == {"frame"}

    async def test_empty_store_result_returns_empty(self):
        rt = MagicMock()
        rt.ltm.query_frame_chunks = AsyncMock(return_value=[])
        out = await FrameChannel().collect(
            _ctx(q_frames={"F"}, expanded={"F"}, anchors=set()), rt, 5
        )
        assert out == {}


class TestFrameChannelPromote:
    async def test_promote_extracts_and_writes(self, monkeypatch):
        rt = ChannelStores(ltm=MagicMock())
        rt.ltm._qry = AsyncMock(
            return_value=[{"id": "c1", "text": "2pm | Mel: I went camping in the mountains."}]
        )
        rt.ltm.write_frames = AsyncMock(return_value=1)
        # stub extraction so the test stays offline (no spaCy/FrameNet load)
        monkeypatch.setattr(
            "graphknows.channels._framenet.extract_frames_from_chunk",
            lambda text, speakers: [
                {
                    "frame": "Residence",
                    "lu": "camp.v",
                    "speaker": "Mel",
                    "fes": [("in", "mountains")],
                }
            ],
        )
        await FrameChannel().promote(rt, "sess", record=None)
        rt.ltm.write_frames.assert_awaited_once()
        session_id, rows = rt.ltm.write_frames.await_args.args
        assert session_id == "sess"
        assert rows[0]["frame_type"] == "Residence"
        assert rows[0]["chunk_id"] == "c1"
        assert "mountains" in rows[0]["fillers"] and "Mel" in rows[0]["fillers"]

    async def test_promote_no_chunks_is_noop(self):
        rt = ChannelStores(ltm=MagicMock())
        rt.ltm._qry = AsyncMock(return_value=[])
        rt.ltm.write_frames = AsyncMock()
        await FrameChannel().promote(rt, "sess", record=None)
        rt.ltm.write_frames.assert_not_awaited()


pytestmark = pytest.mark.asyncio

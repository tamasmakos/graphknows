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
        assert await ch.promote(MagicMock(), "s", None) is None


class TestFrameChannelCollect:
    async def test_no_matched_frames_returns_empty_without_touching_store(self, monkeypatch):
        """Query embedding matches no frame -> return {} and never query the store."""
        monkeypatch.setattr(
            "graphknows.channels._frame_index.match_frames", lambda emb, top_k=5: []
        )
        rt = MagicMock()
        rt.store.query_frame_chunks = AsyncMock()
        out = await FrameChannel().collect(_ctx(), rt, 5)
        assert out == {}
        rt.store.query_frame_chunks.assert_not_awaited()

    async def test_stronger_query_frame_outranks(self, monkeypatch):
        monkeypatch.setattr(
            "graphknows.channels._frame_index.match_frames",
            lambda emb, top_k=5: [("Process_start", 0.42), ("Activity_start", 0.30)],
        )
        rt = MagicMock()
        rt.store._qry = AsyncMock(return_value=[{"c": 10}])  # 10 session chunks
        rt.store.query_frame_chunks = AsyncMock(
            return_value=[
                {"chunk_id": "c_strong", "frame_type": "Process_start",
                 "speaker": "mel", "fillers": []},
                {"chunk_id": "c_weak", "frame_type": "Activity_start",
                 "speaker": "car", "fillers": []},
            ]
        )
        out = await FrameChannel().collect(_ctx(), rt, 5)
        assert set(out) == {"c_strong", "c_weak"}
        # chunk evoking the higher-cosine query frame ranks first
        assert out["c_strong"][0] > out["c_weak"][0]
        assert out["c_strong"][1]["sources"] == {"frame"}
        # store queried with exactly the matched frame names
        args = rt.store.query_frame_chunks.await_args.args
        assert set(args[0]) == {"Process_start", "Activity_start"}

    async def test_saturated_frame_dropped(self, monkeypatch):
        """A frame evoked in >50% of session chunks is dropped as non-discriminative."""
        monkeypatch.setattr(
            "graphknows.channels._frame_index.match_frames",
            lambda emb, top_k=5: [("Activity_ongoing", 0.35), ("Process_start", 0.34)],
        )
        rt = MagicMock()
        rt.store._qry = AsyncMock(return_value=[{"c": 4}])  # 4 session chunks
        rt.store.query_frame_chunks = AsyncMock(
            return_value=[
                # Activity_ongoing floods 3/4 chunks -> saturated, dropped.
                {"chunk_id": "c1", "frame_type": "Activity_ongoing", "speaker": "", "fillers": []},
                {"chunk_id": "c2", "frame_type": "Activity_ongoing", "speaker": "", "fillers": []},
                {"chunk_id": "c3", "frame_type": "Activity_ongoing", "speaker": "", "fillers": []},
                {"chunk_id": "c4", "frame_type": "Process_start", "speaker": "", "fillers": []},
            ]
        )
        out = await FrameChannel().collect(_ctx(), rt, 5)
        assert set(out) == {"c4"}  # only the discriminative frame's chunk survives

    async def test_empty_store_result_returns_empty(self, monkeypatch):
        monkeypatch.setattr(
            "graphknows.channels._frame_index.match_frames",
            lambda emb, top_k=5: [("F", 0.4)],
        )
        rt = MagicMock()
        rt.store.query_frame_chunks = AsyncMock(return_value=[])
        out = await FrameChannel().collect(_ctx(), rt, 5)
        assert out == {}


class TestFrameChannelPromote:
    async def test_promote_extracts_and_writes(self, monkeypatch):
        rt = MagicMock()
        rt.store._qry = AsyncMock(
            return_value=[{"id": "c1", "text": "2pm | Mel: I went camping in the mountains."}]
        )
        rt.store.write_frames = AsyncMock(return_value=1)
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
        rt.store.write_frames.assert_awaited_once()
        session_id, rows = rt.store.write_frames.await_args.args
        assert session_id == "sess"
        assert rows[0]["frame_type"] == "Residence"
        assert rows[0]["chunk_id"] == "c1"
        assert "mountains" in rows[0]["fillers"] and "Mel" in rows[0]["fillers"]

    async def test_promote_no_chunks_is_noop(self):
        rt = MagicMock()
        rt.store._qry = AsyncMock(return_value=[])
        rt.store.write_frames = AsyncMock()
        await FrameChannel().promote(rt, "sess", record=None)
        rt.store.write_frames.assert_not_awaited()


pytestmark = pytest.mark.asyncio

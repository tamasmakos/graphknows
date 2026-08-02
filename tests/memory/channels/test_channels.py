"""Unit tests for the modular channel architecture + FrameChannel."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from graphknows.channels.base import Channel, ChannelContext
from graphknows.channels.frames import FrameChannel
from graphknows.channels.registry import default_channels
from graphknows.settings import GraphKnowsSettings, MemoryMode


def _ctx(**parse):
    ctx = ChannelContext(
        query="q",
        session_id="s",
        emb=[0.0],
        emb2=None,
        words=["a"],
        words_graph=["a"],
    )
    if parse:
        ctx._frame_parse = parse  # pre-seed to avoid loading spaCy/FrameNet
    return ctx


class TestBuildChannels:
    """``default_channels`` returns OPTIONAL channels only.

    The four baseline collectors (entity, vector, bm25, temporal) are methods on
    the retriever spine and always run — they used to be `Channel` subclasses
    whose whole body was ``return await rt.<method>(...)``, listed in a
    name-keyed registry behind a profile filter that never excluded anything.
    """

    def test_ontology_channel_is_off_unless_asked_for(self):
        """It fuses into RRF unmeasured; see ``registry.default_channels``."""
        s = GraphKnowsSettings(
            GRAPHKNOWS_ENABLE_FRAMES=False,
            GRAPHKNOWS_ONTOLOGY="",
            GRAPHKNOWS_MODE=MemoryMode.llm_free,
        )
        assert [c.name for c in default_channels(s)] == []

    def test_ontology_channel_is_opt_in(self):
        s = GraphKnowsSettings(
            GRAPHKNOWS_ENABLE_FRAMES=False,
            GRAPHKNOWS_ENABLE_ONTOLOGY_CHANNEL=True,
            GRAPHKNOWS_ONTOLOGY="",
            GRAPHKNOWS_MODE=MemoryMode.llm_free,
        )
        assert [c.name for c in default_channels(s)] == ["ontology"]

    def test_frames_channel_added_when_enabled(self):
        s = GraphKnowsSettings(
            GRAPHKNOWS_ENABLE_FRAMES=True,
            GRAPHKNOWS_ENABLE_ONTOLOGY_CHANNEL=True,
            GRAPHKNOWS_ONTOLOGY="",
            GRAPHKNOWS_MODE=MemoryMode.llm_free,
        )
        assert [c.name for c in default_channels(s)] == ["frame", "ontology"]

    def test_frame_channel_runs_in_both_modes(self):
        """The profile filter is gone; every channel declared both modes anyway."""
        for mode in (MemoryMode.llm_free, MemoryMode.llm_assisted):
            s = GraphKnowsSettings(GRAPHKNOWS_ENABLE_FRAMES=True, GRAPHKNOWS_MODE=mode)
            assert "frame" in [c.name for c in default_channels(s)]


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
        rt.store.query = AsyncMock(
            side_effect=[
                [{"c": 4}],  # 4 session chunks
                [  # corpus coverage: Activity_ongoing in 3/4, Process_start in 1/4
                    {"ft": "Activity_ongoing", "n": 3},
                    {"ft": "Process_start", "n": 1},
                ],
            ]
        )
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

    async def test_saturation_measured_on_corpus_not_on_the_sample(self):
        """A frame saturated in the SESSION is dropped even when the sample is sparse.

        The regression this pins: ``_drop_saturated`` used to count coverage from
        the rows it was handed. Those rows come from an ANN capped at ``top_k``
        FRAME nodes -- and FRAME is one node per (chunk, frame) pair -- so the
        numerator was bounded by the sample while the denominator was the whole
        session. The ratio could not reach 0.5 and the guard never fired.

        Measured on conv-30/lgdemo before the fix: ``Activity_done_state`` is
        evoked in 19 of session s1's 28 chunks (68%), yet appeared on only 8 of
        any query's 25 returned rows (29%) -- under the threshold, so it was kept
        on every query. It saturated 18 of 19 sessions, to as high as 86%.

        Here the sample shows it on 2 of 28 chunks (7%) while the corpus says
        19/28 (68%). Sample-based counting keeps it; corpus-based drops it.
        """
        rt = MagicMock()
        rt.store.query = AsyncMock(
            side_effect=[
                [{"c": 28}],  # 28 chunks in the session
                [
                    {"ft": "Activity_done_state", "n": 19},  # 68% -> saturated
                    {"ft": "Process_start", "n": 3},  # 11% -> discriminative
                ],
            ]
        )
        rt.store.query_frames_by_embedding = AsyncMock(
            return_value=[
                {"chunk_id": "c1", "frame_type": "Activity_done_state", "score": 0.42},
                {"chunk_id": "c2", "frame_type": "Activity_done_state", "score": 0.41},
                {"chunk_id": "c3", "frame_type": "Process_start", "score": 0.30},
            ]
        )
        out = await FrameChannel().collect(_ctx(), rt, 5)
        assert set(out) == {"c3"}

    async def test_sample_dense_but_corpus_sparse_frame_survives(self):
        """The inverse: dominating one query's results is not saturation.

        A frame the query genuinely matches will fill the sample. That is the
        channel working, not flooding -- so it must survive as long as its
        coverage across the session stays low.
        """
        rt = MagicMock()
        rt.store.query = AsyncMock(
            side_effect=[
                [{"c": 40}],
                [{"ft": "Process_start", "n": 4}],  # 10% of the session
            ]
        )
        rt.store.query_frames_by_embedding = AsyncMock(
            return_value=[
                {"chunk_id": "c1", "frame_type": "Process_start", "score": 0.42},
                {"chunk_id": "c2", "frame_type": "Process_start", "score": 0.41},
                {"chunk_id": "c3", "frame_type": "Process_start", "score": 0.40},
            ]
        )
        out = await FrameChannel().collect(_ctx(), rt, 5)
        assert set(out) == {"c1", "c2", "c3"}
        assert out["c1"][0] > out["c2"][0] > out["c3"][0]

    async def test_empty_store_result_returns_empty(self):
        rt = MagicMock()
        rt.store.query_frames_by_embedding = AsyncMock(return_value=[])
        out = await FrameChannel().collect(_ctx(), rt, 5)
        assert out == {}


pytestmark = pytest.mark.asyncio

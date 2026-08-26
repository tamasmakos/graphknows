"""FrameFactChannel — the fact layer's route back to a chunk.

The point under test is the route, not the ranking: a FRAME_INSTANCE knows the
chunk it was read from, so a query that matches a frame or a role filler can
name its own evidence. ``REL`` cannot, which is why the relation layer has never
moved retrieval.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from graphknows.channels.base import ChannelContext
from graphknows.channels.frame_facts import FrameFactChannel
from graphknows.channels.registry import default_channels
from graphknows.settings import GraphKnowsSettings


def _ctx(**parse: Any) -> ChannelContext:
    ctx = ChannelContext(
        query="what did Melanie paint?",
        session_id="s1",
        emb=[0.1],
        emb2=None,
        words=["melanie", "paint"],
        words_graph=["melanie", "paint"],
    )
    # Pre-seed the lazy parse so no spaCy/FrameNet load happens in a unit test.
    ctx._frame_parse = {"q_frames": set(), "expanded": set(), "anchors": set(), **parse}
    return ctx


def _rt(
    rows: list[dict[str, Any]],
    resolved: list[str] | None = None,
    total: int = 100,
    coverage: dict[str, int] | None = None,
) -> MagicMock:
    rt = MagicMock()
    # `resolved or [...]` would swallow an explicitly empty list — the exact case
    # the "skips the read" test needs.
    rt.store.resolve_query_entities = AsyncMock(
        return_value=["Melanie"] if resolved is None else resolved
    )
    rt.store.query_chunks_by_frame_facts = AsyncMock(return_value=rows)
    # The saturation guard's two reads, in call order: session chunk count, then
    # per-frame chunk coverage.
    rt.store.query = AsyncMock(
        side_effect=[
            [{"c": total}],
            [{"ft": ft, "n": n} for ft, n in (coverage or {}).items()],
        ]
    )
    return rt


@pytest.mark.asyncio
async def test_matching_instances_yield_their_chunks_in_rank_order() -> None:
    rt = _rt([{"chunk_id": "c1", "score": 3.0}, {"chunk_id": "c2", "score": 1.0}])

    out = await FrameFactChannel().collect(_ctx(expanded={"Create_physical_artwork"}), rt, 5)

    assert list(out) == ["c1", "c2"]
    assert out["c1"][0] > out["c2"][0]  # rank 0 outranks rank 1
    assert out["c1"][1]["sources"] == {"frame_fact"}


@pytest.mark.asyncio
async def test_expanded_frames_are_queried_not_the_bare_query_frames() -> None:
    # Question and statement rarely evoke the same frame, so the 1-hop expansion
    # is what makes the join land at all.
    rt = _rt([])
    await FrameFactChannel().collect(
        _ctx(q_frames={"Giving"}, expanded={"Giving", "Transfer"}), rt, 5
    )

    frames = rt.store.query_chunks_by_frame_facts.await_args.args[0]
    assert frames == ["Giving", "Transfer"]


@pytest.mark.asyncio
async def test_no_frames_and_no_entities_skips_the_read_entirely() -> None:
    rt = _rt([], resolved=[])
    assert await FrameFactChannel().collect(_ctx(), rt, 5) == {}
    rt.store.query_chunks_by_frame_facts.assert_not_awaited()


@pytest.mark.asyncio
async def test_entities_alone_still_route(monkeypatch: pytest.MonkeyPatch) -> None:
    rt = _rt([{"chunk_id": "c9", "score": 2.0}], resolved=["Melanie"])
    out = await FrameFactChannel().collect(_ctx(), rt, 5)
    assert list(out) == ["c9"]


@pytest.mark.asyncio
async def test_a_frame_covering_most_of_the_session_is_dropped_before_the_read() -> None:
    """A frame in most chunks names no evidence — it floods the fusion pool.

    Measured on conv-30: SRL raises FRAME_INSTANCE to ~10 per chunk, so the
    unguarded join returned 100 of the session's 110 chunks and displaced
    genuine hits (evidence_recall_in_context 0.9255 -> 0.7646). Both sibling
    channels already drop saturated matches; this one did not.
    """
    rt = _rt([], total=100, coverage={"Motion": 80, "Giving": 5})

    await FrameFactChannel().collect(_ctx(expanded={"Motion", "Giving"}), rt, 5)

    frames = rt.store.query_chunks_by_frame_facts.await_args.args[0]
    assert frames == ["Giving"]


@pytest.mark.asyncio
async def test_dropping_every_frame_still_routes_by_role_filler() -> None:
    # The entity side is independent evidence; a saturated frame must not take
    # it down with it.
    rt = _rt(
        [{"chunk_id": "c4", "score": 1.0}], resolved=["Melanie"], total=100, coverage={"Motion": 90}
    )

    out = await FrameFactChannel().collect(_ctx(expanded={"Motion"}), rt, 5)

    assert list(out) == ["c4"]
    assert rt.store.query_chunks_by_frame_facts.await_args.args[0] == []


@pytest.mark.asyncio
async def test_every_frame_saturated_and_no_entities_skips_the_read() -> None:
    rt = _rt([], resolved=[], total=100, coverage={"Motion": 90})
    assert await FrameFactChannel().collect(_ctx(expanded={"Motion"}), rt, 5) == {}
    rt.store.query_chunks_by_frame_facts.assert_not_awaited()


@pytest.mark.asyncio
async def test_without_a_session_there_is_no_corpus_to_measure_saturation_against() -> None:
    rt = _rt([{"chunk_id": "c1", "score": 1.0}])
    ctx = _ctx(expanded={"Motion"})
    ctx.session_id = ""

    await FrameFactChannel().collect(ctx, rt, 5)

    rt.store.query.assert_not_awaited()
    assert rt.store.query_chunks_by_frame_facts.await_args.args[0] == ["Motion"]


def test_channel_is_off_by_default_and_opt_in() -> None:
    # Adding a leg to the fusion is the move that has regressed this project;
    # it must stay an A/B until measured.
    names = [
        c.name for c in default_channels(GraphKnowsSettings(GRAPHKNOWS_ENABLE_FRAME_FACTS=False))
    ]
    assert "frame_fact" not in names
    on = [c.name for c in default_channels(GraphKnowsSettings(GRAPHKNOWS_ENABLE_FRAME_FACTS=True))]
    assert "frame_fact" in on

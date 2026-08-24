"""OntologyChannel — registration gating and the saturation guard."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from graphknows.channels.base import ChannelContext
from graphknows.channels.ontology import OntologyChannel
from graphknows.channels.registry import default_channels
from graphknows.settings import GraphKnowsSettings

ONT = "graphknows/symbolic/ontology/assets/personal/personal.ttl"


def _ctx(session_id: str = "s1") -> ChannelContext:
    return ChannelContext(
        query="where did they go",
        session_id=session_id,
        emb=[0.1] * 384,
        emb2=None,
        words=[],
        words_graph=[],
        state=None,
    )


def _rt(rows: list[dict], chunk_total: int, coverage: dict[str, int] | None = None) -> MagicMock:
    """A retriever whose graph reports *coverage* chunks per class URI.

    Coverage is a SECOND query now, not a count over ``rows``. The guard used to
    measure saturation from the returned page — numerator bounded by the page
    size, denominator the whole session — so it could never reach the threshold
    and never dropped anything. A test that supplies no coverage is a test of a
    graph where nothing is saturated.
    """
    rt = MagicMock()
    rt.store.search_ontology_classes = AsyncMock(return_value=rows)
    rt.store.query = AsyncMock(
        side_effect=[
            [{"c": chunk_total}],
            [{"k": uri, "n": n} for uri, n in (coverage or {}).items()],
        ]
    )
    return rt


class TestRegistration:
    def test_absent_by_default(self) -> None:
        """Opt-in: an unmeasured leg does not get to sit in the fusion."""
        names = [c.name for c in default_channels(GraphKnowsSettings(GRAPHKNOWS_ONTOLOGY=""))]
        assert "ontology" not in names

    def test_configuring_an_ontology_does_not_switch_it_on(self) -> None:
        """``ontology_source`` always resolves, so it can never be the gate."""
        names = [c.name for c in default_channels(GraphKnowsSettings(GRAPHKNOWS_ONTOLOGY=ONT))]
        assert "ontology" not in names

    def test_present_when_asked_for(self) -> None:
        names = [
            c.name
            for c in default_channels(
                GraphKnowsSettings(GRAPHKNOWS_ONTOLOGY=ONT, GRAPHKNOWS_ENABLE_ONTOLOGY_CHANNEL=True)
            )
        ]
        assert "ontology" in names


class TestCollect:
    @pytest.mark.asyncio
    async def test_no_rows_yields_nothing(self) -> None:
        assert await OntologyChannel().collect(_ctx(), _rt([], 10), 5) == {}

    @pytest.mark.asyncio
    async def test_ranks_chunks_by_summed_class_score(self) -> None:
        rows = [
            {"chunk_id": "a", "uri": "u1", "label": "Place", "score": 0.1},
            {"chunk_id": "b", "uri": "u1", "label": "Place", "score": 0.9},
            {"chunk_id": "a", "uri": "u2", "label": "Event", "score": 0.3},
        ]
        out = await OntologyChannel().collect(_ctx(), _rt(rows, 20), 5)
        assert set(out) == {"a", "b"}
        # b (0.9) outranks a (0.1 + 0.3), so it gets the better RRF score.
        assert out["b"][0] > out["a"][0]
        assert out["b"][1]["sources"] == {"ontology"}

    @pytest.mark.asyncio
    async def test_saturated_class_is_dropped(self) -> None:
        """A class tagging >50% of the session is noise, not signal.

        In a personal corpus almost every chunk is "about" a Person; returning
        its chunks floods recall with the whole conversation.
        """
        rows = [
            {"chunk_id": f"c{i}", "uri": "person", "label": "Person", "score": 0.5}
            for i in range(6)
        ]
        # Person tags 6 of the session's 10 chunks in the GRAPH — the sample
        # happens to agree here, but it is the graph figure the guard reads.
        rt = _rt(rows, 10, coverage={"person": 6})
        assert await OntologyChannel().collect(_ctx(), rt, 5) == {}

    @pytest.mark.asyncio
    async def test_saturation_is_measured_on_the_corpus_not_on_the_sample(self) -> None:
        """The defect the frame channel found first, fixed here too.

        Person tags 8 of 10 session chunks, but this query's page shows it on
        only 2. Counting the page gives 2/10 = 20%, under the 50% threshold, so
        the guard never fired. Counting the graph gives 80% and drops it.
        """
        rows = [
            {"chunk_id": "c1", "uri": "person", "label": "Person", "score": 0.5},
            {"chunk_id": "c2", "uri": "person", "label": "Person", "score": 0.5},
            {"chunk_id": "c3", "uri": "place", "label": "Place", "score": 0.4},
        ]
        rt = _rt(rows, 10, coverage={"person": 8, "place": 1})
        out = await OntologyChannel().collect(_ctx(), rt, 5)
        assert set(out) == {"c3"}, "saturated class survived because coverage came from the page"

    @pytest.mark.asyncio
    async def test_discriminative_class_survives_alongside_a_saturated_one(self) -> None:
        rows = [
            *[
                {"chunk_id": f"c{i}", "uri": "person", "label": "Person", "score": 0.5}
                for i in range(6)
            ],
            {"chunk_id": "c9", "uri": "place", "label": "Place", "score": 0.8},
        ]
        rt = _rt(rows, 10, coverage={"person": 6, "place": 1})
        out = await OntologyChannel().collect(_ctx(), rt, 5)
        assert set(out) == {"c9"}

    @pytest.mark.asyncio
    async def test_without_a_session_saturation_is_not_measurable(self) -> None:
        """No session means no corpus to measure against, so rows pass through."""
        rows = [
            {"chunk_id": f"c{i}", "uri": "person", "label": "Person", "score": 0.5}
            for i in range(6)
        ]
        out = await OntologyChannel().collect(_ctx(session_id=""), _rt(rows, 10), 6)
        assert len(out) == 6, "rows were dropped despite there being no corpus to measure"

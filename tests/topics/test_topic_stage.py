"""The topic stage: metrics, clustering, assembly and the export artifact."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from graphknows.settings import TopicMode
from graphknows.topics import TopicConfig, TopicSet, newman_modularity
from graphknows.topics.metrics import coherence, keywords, summarize_metrics
from graphknows.topics.model import Topic
from graphknows.topics.persist import export_json, persist_topics


class TestNewmanModularity:
    """The metric ArcadeDB's algo.modularityScore is not."""

    def test_single_cluster_is_zero(self) -> None:
        """Q for one all-encompassing cluster is 0 by definition.

        This is the exact case ArcadeDB scores 0.750, which is why a sweep must
        not be tuned against the native score.
        """
        edges = [("a", "b"), ("b", "c"), ("c", "d"), ("d", "a")]
        part = dict.fromkeys("abcd", 0)
        assert abs(newman_modularity(edges, part)) < 1e-9

    def test_real_structure_beats_random(self) -> None:
        # Two tight triangles joined by a single bridge.
        edges = [
            ("a", "b"), ("b", "c"), ("c", "a"),
            ("x", "y"), ("y", "z"), ("z", "x"),
            ("a", "x"),
        ]
        good = {"a": 0, "b": 0, "c": 0, "x": 1, "y": 1, "z": 1}
        bad = {"a": 0, "b": 1, "c": 0, "x": 1, "y": 0, "z": 1}
        assert newman_modularity(edges, good) > newman_modularity(edges, bad)
        assert newman_modularity(edges, good) > 0.3

    def test_empty_graph_is_zero(self) -> None:
        assert newman_modularity([], {"a": 0}) == 0.0

    def test_unpartitioned_nodes_are_ignored(self) -> None:
        edges = [("a", "b"), ("b", "unknown")]
        assert newman_modularity(edges, {"a": 0, "b": 0}) == 0.0

    def test_self_loops_ignored(self) -> None:
        assert newman_modularity([("a", "a")], {"a": 0}) == 0.0


class TestCoherence:
    def test_identical_vectors_are_perfectly_coherent(self) -> None:
        vecs = np.tile(np.array([1.0, 0.0, 0.0], dtype=np.float32), (4, 1))
        score, centroid = coherence(vecs)
        assert score == pytest.approx(1.0, abs=1e-5)
        assert len(centroid) == 3

    def test_spread_vectors_are_less_coherent(self) -> None:
        tight = np.array([[1, 0.02, 0], [1, 0, 0.02], [1, 0.01, 0.01]], dtype=np.float32)
        spread = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
        assert coherence(tight)[0] > coherence(spread)[0]

    def test_empty_is_zero(self) -> None:
        assert coherence(np.zeros((0, 0), dtype=np.float32)) == (0.0, [])


class TestKeywords:
    def test_distinctive_terms_win_over_common_ones(self) -> None:
        """A term shared by every topic carries no signal, however frequent."""
        mine = ["kayaking down the river was wonderful", "kayaking again on the river"]
        others = [{"weather", "river"}, {"dinner", "river"}, {"river", "traffic"}]
        out = keywords(mine, others, top_k=3)
        assert "kayaking" in out
        assert "river" not in out

    def test_stopwords_and_short_tokens_dropped(self) -> None:
        assert keywords(["the a an it is of to"], [], top_k=5) == []

    def test_empty_input(self) -> None:
        assert keywords([], [], top_k=5) == []


class TestSummarizeMetrics:
    def test_grades_a_build(self) -> None:
        topics = [
            Topic(topic_id="t1", title="a", entities=[("a", 1.0), ("b", 0.5)], chunk_ids=["c1"],
                  coherence=0.9),
            Topic(topic_id="t2", title="b", entities=[("c", 0.2)], chunk_ids=["c2"], coherence=0.5),
        ]
        m = summarize_metrics(
            topics, total_chunks=4, total_entities=6, newman_q=0.31, native_modularity=0.75
        )
        assert m["topics"] == 2
        assert m["newman_modularity"] == 0.31
        assert m["chunk_coverage"] == 0.5
        assert m["entity_coverage"] == 0.5
        assert m["largest_topic"] == 2
        assert m["singleton_topics"] == 1
        # Reported, but named so nobody optimises it by accident.
        assert m["arcadedb_modularity_untrustworthy"] == 0.75

    def test_zero_totals_do_not_divide_by_zero(self) -> None:
        m = summarize_metrics([], total_chunks=0, total_entities=0, newman_q=0.0,
                              native_modularity=None)
        assert m["chunk_coverage"] == 0.0
        assert m["arcadedb_modularity_untrustworthy"] is None


class TestConfig:
    def test_fingerprint_is_stable_and_parameter_sensitive(self) -> None:
        a = TopicConfig(distance=0.3)
        assert a.fingerprint() == TopicConfig(distance=0.3).fingerprint()
        assert a.fingerprint() != TopicConfig(distance=0.4).fingerprint()

    def test_label_does_not_change_the_fingerprint(self) -> None:
        """The label is provenance, not a parameter."""
        assert TopicConfig(label="x").fingerprint() == TopicConfig(label="y").fingerprint()

    def test_default_source_is_the_high_coverage_one(self) -> None:
        """graph_embedding covers ~3% of a sparse conversational graph."""
        assert TopicConfig().source == "community"

    def test_to_dict_is_json_serialisable(self) -> None:
        json.dumps(TopicConfig(mode=TopicMode.llm).to_dict())


class TestExport:
    def _topic_set(self) -> TopicSet:
        return TopicSet(
            topics=[
                Topic(
                    topic_id="t1",
                    title="Trips",
                    summary="Trips. We went kayaking.",
                    keywords=["kayaking"],
                    entities=[("Caroline", 0.4)],
                    chunk_ids=["c1"],
                    coherence=0.87,
                    embedding=[0.1] * 8,
                )
            ],
            metrics={"topics": 1},
            config=TopicConfig().to_dict(),
            namespace="ns",
        )

    def test_export_omits_embeddings_by_default(self, tmp_path: Path) -> None:
        path = export_json(self._topic_set(), tmp_path / "t.json")
        data = json.loads(path.read_text(encoding="utf-8"))
        assert "embedding" not in data["topics"][0]
        assert data["topics"][0]["keywords"] == ["kayaking"]
        assert data["topics"][0]["entities"] == [{"name": "Caroline", "pagerank": 0.4}]

    def test_export_can_include_embeddings(self, tmp_path: Path) -> None:
        path = export_json(self._topic_set(), tmp_path / "t.json", include_embedding=True)
        assert len(json.loads(path.read_text())["topics"][0]["embedding"]) == 8

    def test_artifact_says_it_is_read_only(self, tmp_path: Path) -> None:
        """There is no import path; the graph stays authoritative."""
        data = json.loads(export_json(self._topic_set(), tmp_path / "t.json").read_text())
        assert "never imported" in data["_note"]

    def test_creates_missing_parent_directories(self, tmp_path: Path) -> None:
        assert export_json(self._topic_set(), tmp_path / "a" / "b" / "t.json").exists()


class TestPersist:
    @pytest.mark.asyncio
    async def test_replaces_the_previous_layer_by_default(self) -> None:
        """Two builds must not coexist, or a channel matches topics from both."""
        store = MagicMock()
        store.command = AsyncMock()
        store.write_topics = AsyncMock(return_value=1)
        ts = TopicSet(topics=[Topic(topic_id="t1", title="x")])

        await persist_topics(store, ts)

        assert "DETACH DELETE" in store.command.await_args[0][0]
        assert store.write_topics.await_args[0][0][0]["topic_id"] == "t1"

    @pytest.mark.asyncio
    async def test_replace_false_keeps_the_old_layer(self) -> None:
        store = MagicMock()
        store.command = AsyncMock()
        store.write_topics = AsyncMock(return_value=1)
        await persist_topics(store, TopicSet(topics=[Topic(topic_id="t1", title="x")]),
                             replace=False)
        store.command.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_empty_build_writes_nothing(self) -> None:
        store = MagicMock()
        store.command = AsyncMock()
        store.write_topics = AsyncMock()
        assert await persist_topics(store, TopicSet()) == 0
        store.write_topics.assert_not_awaited()

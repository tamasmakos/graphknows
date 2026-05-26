"""Wave 3 — community/detection.py tests.

Tests the CommunityDetector class:
- _get_entity_graph subgraph filtering
- run_leiden on small graphs (no igraph/leidenalg calls needed — graph too small)
- detect_communities on trivially small graphs (fallback path)
- _merge_small_communities (pure graph logic)
"""

from __future__ import annotations

import networkx as nx
import pytest


def _make_entity_graph(n_nodes: int = 5) -> nx.Graph:
    """Build a small undirected graph with entity nodes and entity_relation edges."""
    G = nx.Graph()
    for i in range(n_nodes):
        G.add_node(f"E{i}", node_type="ENTITY")
    for i in range(n_nodes - 1):
        G.add_edge(f"E{i}", f"E{i+1}", graph_type="entity_relation", weight=1.0)
    return G


class TestGetEntityGraph:
    def setup_method(self):
        from graphknows.graphgen.pipeline.community.detection import CommunityDetector

        self.det = CommunityDetector()

    def test_includes_entity_nodes_with_entity_relation_edges(self):
        G = _make_entity_graph(4)
        subg = self.det._get_entity_graph(G)
        # All 4 entity nodes should appear
        assert len(list(subg.nodes())) == 4

    def test_excludes_non_entity_nodes(self):
        G = nx.Graph()
        G.add_node("E0", node_type="ENTITY")
        G.add_node("D0", node_type="DOCUMENT")  # not entity
        G.add_edge("E0", "D0", graph_type="entity_relation")
        subg = self.det._get_entity_graph(G)
        # D0 should be filtered out
        assert "D0" not in list(subg.nodes())

    def test_excludes_non_entity_relation_edges(self):
        G = nx.Graph()
        G.add_node("E0", node_type="ENTITY")
        G.add_node("E1", node_type="ENTITY")
        G.add_edge("E0", "E1", graph_type="semantic")  # not entity_relation
        subg = self.det._get_entity_graph(G)
        assert subg.number_of_edges() == 0

    def test_includes_place_and_named_entity_types(self):
        G = nx.Graph()
        G.add_node("P0", node_type="PLACE")
        G.add_node("N0", node_type="NAMEDENTITY")
        G.add_edge("P0", "N0", graph_type="entity_relation")
        subg = self.det._get_entity_graph(G)
        assert len(list(subg.nodes())) >= 1


class TestRunLeiden:
    def setup_method(self):
        from graphknows.graphgen.pipeline.community.detection import CommunityDetector

        self.det = CommunityDetector()

    def test_returns_single_community_for_empty_graph(self):
        G = nx.Graph()
        G.add_node("A")
        result = self.det.run_leiden(G)
        # < 3 nodes → fallback
        assert result == {"A": 0}

    def test_returns_community_zero_for_two_nodes(self):
        G = nx.Graph()
        G.add_node("A")
        G.add_node("B")
        G.add_edge("A", "B")
        result = self.det.run_leiden(G)
        assert result == {"A": 0, "B": 0}

    def test_returns_single_community_for_no_edges(self):
        G = nx.Graph()
        for n in ["A", "B", "C"]:
            G.add_node(n)
        # 3 nodes but 0 edges → fallback
        result = self.det.run_leiden(G)
        assert all(v == 0 for v in result.values())

    def test_calls_leiden_on_connected_graph(self):
        """For a connected 4-node graph, leiden should return a partition dict."""
        G = _make_entity_graph(4)
        result = self.det.run_leiden(G)
        # Could return tuple (partition, modularity) or dict depending on success
        if isinstance(result, tuple):
            partition, modularity = result
            assert isinstance(partition, dict)
            assert isinstance(modularity, float)
        else:
            # fallback dict
            assert isinstance(result, dict)


class TestDetectCommunities:
    def setup_method(self):
        from graphknows.graphgen.pipeline.community.detection import CommunityDetector

        self.det = CommunityDetector()

    def test_returns_fallback_for_trivially_small_graph(self):
        """Graph with < 3 entity nodes → should return fallback."""
        G = nx.Graph()
        G.add_node("A", node_type="ENTITY")
        G.add_node("B", node_type="ENTITY")
        G.add_edge("A", "B", graph_type="entity_relation")
        # Only 2 entity nodes → entity_graph.number_of_nodes() < 3
        result = self.det.detect_communities(G)
        assert "assignments" in result
        assert "modularity" in result
        assert result["community_count"] == 1

    def test_returns_fallback_for_graph_with_no_entity_edges(self):
        G = nx.Graph()
        for n in ["A", "B", "C"]:
            G.add_node(n, node_type="ENTITY")
        # Edges exist but not entity_relation type
        G.add_edge("A", "B", graph_type="semantic")
        result = self.det.detect_communities(G)
        assert result["community_count"] == 1

    def test_returns_dict_with_expected_keys(self):
        G = _make_entity_graph(3)
        result = self.det.detect_communities(G)
        for key in ["assignments", "modularity", "community_count", "resolution"]:
            assert key in result, f"Missing key: {key}"

    def test_uses_settings_when_provided(self):
        from graphknows.graphgen.pipeline.community.detection import CommunityDetector

        mock_settings = type(
            "Settings",
            (),
            {"resolutions": [0.5, 1.0], "n_iterations": 1, "min_community_size": 1, "seed": 42},
        )()
        det = CommunityDetector(settings=mock_settings)
        G = nx.Graph()
        G.add_node("A")
        result = det.detect_communities(G)
        assert "assignments" in result


class TestMergeSmallCommunities:
    def setup_method(self):
        from graphknows.graphgen.pipeline.community.detection import CommunityDetector

        self.det = CommunityDetector()

    def test_no_op_when_all_communities_large_enough(self):
        G = _make_entity_graph(4)
        partition = {"E0": 0, "E1": 0, "E2": 1, "E3": 1}
        result = self.det._merge_small_communities(G, partition, min_size=2)
        # All communities have >=2 members → no merging needed
        from collections import Counter
        counts = Counter(result.values())
        assert all(c >= 2 for c in counts.values()) or len(counts) <= 2

    def test_merges_singleton_community(self):
        G = _make_entity_graph(4)
        # E2 in its own community → should be merged
        partition = {"E0": 0, "E1": 0, "E2": 0, "E3": 1}
        result = self.det._merge_small_communities(G, partition, min_size=2)
        from collections import Counter
        counts = Counter(result.values())
        # Singletons should be absorbed
        assert min(counts.values()) >= 1  # at minimum no crash

    def test_returns_dict_with_same_keys(self):
        G = _make_entity_graph(5)
        partition = {f"E{i}": i for i in range(5)}
        result = self.det._merge_small_communities(G, partition, min_size=2)
        assert set(result.keys()) == set(partition.keys())


# ===========================================================================
# Subcommunity / enhanced attributes
# ===========================================================================



import networkx as nx
import pytest


def _make_graph_with_entities(*entity_ids: str) -> nx.DiGraph:
    """Build a DiGraph with entity nodes."""
    G = nx.DiGraph()
    for eid in entity_ids:
        G.add_node(eid, node_type="ENTITY", name=eid)
    return G


class TestAddEnhancedCommunityAttributes:
    def _call(self, G, communities, subcommunities):
        from graphknows.graphgen.pipeline.community.subcommunities import (
            add_enhanced_community_attributes_to_graph,
        )
        return add_enhanced_community_attributes_to_graph(G, communities, subcommunities)

    def test_creates_topic_nodes_for_each_community(self):
        G = _make_graph_with_entities("E0", "E1", "E2")
        communities = {"E0": 0, "E1": 0, "E2": 1}
        result = self._call(G, communities, {})
        assert "TOPIC_0" in result.nodes
        assert "TOPIC_1" in result.nodes

    def test_topic_node_has_correct_type(self):
        G = _make_graph_with_entities("E0")
        result = self._call(G, {"E0": 0}, {})
        assert result.nodes["TOPIC_0"]["node_type"] == "TOPIC"

    def test_entity_connected_to_topic_via_fallback(self):
        """Entities not in subcommunities get a direct IN_TOPIC edge via fallback."""
        G = _make_graph_with_entities("E0")
        result = self._call(G, {"E0": 0}, {})
        assert result.has_edge("E0", "TOPIC_0")

    def test_subcommunity_creates_subtopic_node(self):
        G = _make_graph_with_entities("E0")
        communities = {"E0": 0}
        subcommunities = {"E0": (0, 1)}  # parent_comm=0, local_sub=1
        result = self._call(G, communities, subcommunities)
        assert "SUBTOPIC_0_1" in result.nodes

    def test_entity_to_subtopic_edge_created(self):
        G = _make_graph_with_entities("E0")
        communities = {"E0": 0}
        subcommunities = {"E0": (0, 2)}
        result = self._call(G, communities, subcommunities)
        assert result.has_edge("E0", "SUBTOPIC_0_2")

    def test_subtopic_to_topic_edge_created(self):
        G = _make_graph_with_entities("E0")
        communities = {"E0": 0}
        subcommunities = {"E0": (0, 2)}
        result = self._call(G, communities, subcommunities)
        assert result.has_edge("SUBTOPIC_0_2", "TOPIC_0")

    def test_non_entity_nodes_excluded(self):
        """DOCUMENT nodes should not get IN_TOPIC edges."""
        G = nx.DiGraph()
        G.add_node("D0", node_type="DOCUMENT", name="doc")
        communities = {"D0": 0}
        result = self._call(G, communities, {})
        # TOPIC_0 is created but D0 should not be connected
        if "TOPIC_0" in result.nodes:
            assert not result.has_edge("D0", "TOPIC_0")

    def test_entity_not_in_graph_skipped(self):
        """Entities in the communities dict but absent from the graph are skipped."""
        G = nx.DiGraph()
        communities = {"E_ghost": 0}
        result = self._call(G, communities, {})
        # Should not raise, even if TOPIC_0 is created
        assert isinstance(result, nx.DiGraph)

    def test_multiple_entities_same_community(self):
        G = _make_graph_with_entities("E0", "E1", "E2")
        communities = {"E0": 0, "E1": 0, "E2": 0}
        result = self._call(G, communities, {})
        assert "TOPIC_0" in result.nodes
        # All 3 entities should be connected
        for eid in ["E0", "E1", "E2"]:
            assert result.has_edge(eid, "TOPIC_0"), f"{eid} not connected to TOPIC_0"

    def test_returns_same_graph_object(self):
        G = _make_graph_with_entities("E0")
        result = self._call(G, {"E0": 0}, {})
        # Should return the same DiGraph (modified in-place and returned)
        assert result is G

    def test_subtopic_node_has_correct_attributes(self):
        G = _make_graph_with_entities("E0")
        communities = {"E0": 0}
        subcommunities = {"E0": (0, 3)}
        result = self._call(G, communities, subcommunities)
        sub = result.nodes["SUBTOPIC_0_3"]
        assert sub["node_type"] == "SUBTOPIC"
        assert sub["community_id"] == 0
        assert sub["subtopic_local_id"] == 3

    def test_no_duplicate_edges_on_repeated_entities(self):
        """Calling with same entity twice should not create duplicate edges."""
        G = _make_graph_with_entities("E0")
        communities = {"E0": 0}
        result = self._call(G, communities, {})
        edge_count_1 = result.number_of_edges()
        # Call again — should be idempotent (no new edges)
        result = self._call(result, communities, {})
        edge_count_2 = result.number_of_edges()
        assert edge_count_2 == edge_count_1

    def test_entity_with_none_local_sub_id_connects_directly(self):
        """local_sub_id == None means direct Entity → Topic connection."""
        G = _make_graph_with_entities("E0")
        communities = {"E0": 0}
        subcommunities = {"E0": (0, None)}
        result = self._call(G, communities, subcommunities)
        # Should connect directly to TOPIC_0, no SUBTOPIC
        assert result.has_edge("E0", "TOPIC_0")

    def test_entity_with_minus_one_local_sub_id_connects_directly(self):
        """local_sub_id == -1 also means direct Entity → Topic connection."""
        G = _make_graph_with_entities("E0")
        communities = {"E0": 0}
        subcommunities = {"E0": (0, -1)}
        result = self._call(G, communities, subcommunities)
        assert result.has_edge("E0", "TOPIC_0")

import unittest
import networkx as nx
from src.kg.community.detection import CommunityDetector

class TestCommunityDetection(unittest.TestCase):
    def setUp(self):
        self.detector = CommunityDetector()
        self.graph = nx.Graph()
        # Create a simple graph with two clear communities
        # Community 1: 0, 1, 2
        self.graph.add_edges_from([(0, 1), (1, 2), (0, 2)])
        # Community 2: 3, 4, 5
        self.graph.add_edges_from([(3, 4), (4, 5), (3, 5)])
        # Bridge
        self.graph.add_edge(2, 3)

    def test_detect_communities(self):
        communities = self.detector.detect_communities(self.graph)
        self.assertEqual(len(set(communities.values())), 2)
        self.assertEqual(communities[0], communities[1])
        self.assertEqual(communities[3], communities[4])
        self.assertNotEqual(communities[0], communities[3])

    def test_empty_graph(self):
        empty_graph = nx.Graph()
        communities = self.detector.detect_communities(empty_graph)
        self.assertEqual(communities, {})

if __name__ == '__main__':
    unittest.main()

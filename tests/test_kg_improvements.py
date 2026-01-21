
import unittest
import networkx as nx
import logging
from src.kg.graph.text_resolution import resolve_similar_entities
from src.kg.graph.pruning import prune_graph

# Configure logging to see output
logging.basicConfig(level=logging.INFO)

class TestKGImprovements(unittest.TestCase):
    
    def test_entity_resolution_hybrid(self):
        """Test that "New York City" and "City New York" are merged."""
        g = nx.DiGraph()
        
        # Add nodes with reordered names
        g.add_node("n1", node_type="ENTITY_CONCEPT", name="New York City")
        g.add_node("n2", node_type="ENTITY_CONCEPT", name="City New York") # Should match via Token Sort
        g.add_node("n3", node_type="ENTITY_CONCEPT", name="New York") # Might match if score > threshold
        g.add_node("n4", node_type="ENTITY_CONCEPT", name="Different Place")
        
        # Run resolution
        merges = resolve_similar_entities(g, similarity_threshold=0.90)
        
        print("\nMerges performed:", merges)
        
        # Check if n1 and n2 merged
        # The function modifies the graph and returns merges.
        # Check logic: n2 should be merged into n1 (or vice versa)
        
        # We expect at least one merge between n1 and n2
        merged_ids = [(m[0], m[1]) for m in merges]
        
        # Flatten
        all_ids = set()
        for s, t in merged_ids:
            all_ids.add(s)
            all_ids.add(t)
            
        self.assertTrue("n1" in all_ids and "n2" in all_ids, "n1 and n2 should be involved in a merge")
        
    def test_pruning_disconnected(self):
        """Test that small disconnected components are pruned."""
        g = nx.DiGraph()
        
        # Component 1: Main Backbone (Size 3, has SEGMENT)
        g.add_node("s1", node_type="SEGMENT")
        g.add_node("c1", node_type="CHUNK")
        g.add_node("e1", node_type="ENTITY_CONCEPT")
        g.add_edge("s1", "c1")
        g.add_edge("c1", "e1")
        
        # Component 2: Isolated Noise (Size 2, no vital types)
        g.add_node("noise1", node_type="ENTITY_CONCEPT")
        g.add_node("noise2", node_type="ENTITY_CONCEPT")
        g.add_edge("noise1", "noise2")
        
        # Component 3: Isolated but Vital (Size 1, TOPIC) -> Should KEEP
        g.add_node("t1", node_type="TOPIC")
        
        initial_nodes = g.number_of_nodes()
        print(f"\nInitial nodes: {initial_nodes}")
        
        config = {'min_component_size': 3, 'enable_pruning': True}
        stats = prune_graph(g, config)
        
        print("Pruning stats:", stats)
        
        final_nodes = g.number_of_nodes()
        
        self.assertIn("s1", g.nodes, "Backbone segment should be kept")
        self.assertIn("t1", g.nodes, "Isolated Topic should be kept")
        self.assertNotIn("noise1", g.nodes, "Noise component should be pruned")
        self.assertEqual(final_nodes, 4, f"Should have 4 nodes left, got {final_nodes}")

if __name__ == '__main__':
    unittest.main()

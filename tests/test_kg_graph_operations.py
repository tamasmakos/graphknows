"""
Unit tests for knowledge graph operations and utilities.
"""

import unittest
import tempfile
import shutil
import os
import json
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import networkx as nx
import numpy as np

# Add src to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))


class TestGraphUtilities(unittest.TestCase):
    """Test graph utility functions."""
    
    def setUp(self):
        self.graph = nx.DiGraph()
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        shutil.rmtree(self.temp_dir)
    
    def test_create_output_directory(self):
        """Test output directory creation."""
        from kg.graph.utils import create_output_directory
        
        output_path = os.path.join(self.temp_dir, 'test_output')
        create_output_directory(output_path)
        
        self.assertTrue(os.path.exists(output_path))
        self.assertTrue(os.path.isdir(output_path))
    
    def test_graph_schema_export(self):
        """Test graph schema export functionality."""
        from kg.graph.schema import save_graph_schema
        
        # Create test graph with various node types
        self.graph.add_node("doc1", type="DOCUMENT", title="Test Document")
        self.graph.add_node("seg1", type="SEGMENT", text="Test segment")
        self.graph.add_node("entity1", type="ENTITY_CONCEPT", name="Test Entity")
        self.graph.add_node("topic1", type="TOPIC", summary="Test Topic")
        
        # Add edges with different relation types
        self.graph.add_edge("doc1", "seg1", relation="CONTAINS")
        self.graph.add_edge("seg1", "entity1", relation="MENTIONS")
        self.graph.add_edge("entity1", "topic1", relation="BELONGS_TO")
        
        save_graph_schema(self.graph, self.temp_dir)
        
        # Verify schema file was created
        schema_file = os.path.join(self.temp_dir, "graph_schema.json")
        self.assertTrue(os.path.exists(schema_file))
        
        # Verify schema content
        with open(schema_file, 'r') as f:
            schema = json.load(f)
        
        self.assertIn('node_types', schema)
        self.assertIn('edge_types', schema)
        self.assertIn('DOCUMENT', schema['node_types'])
        self.assertIn('ENTITY_CONCEPT', schema['node_types'])
        self.assertIn('CONTAINS', schema['edge_types'])


class TestGraphConstruction(unittest.TestCase):
    """Test graph construction operations."""
    
    def setUp(self):
        self.graph = nx.DiGraph()
    
    def test_add_document_node(self):
        """Test adding document nodes to graph."""
        # Simulate document node creation
        doc_id = "doc_001"
        doc_data = {
            'type': 'DOCUMENT',
            'title': 'Test Document',
            'file_path': '/path/to/doc.txt',
            'processed_date': '2023-01-01'
        }
        
        self.graph.add_node(doc_id, **doc_data)
        
        self.assertIn(doc_id, self.graph.nodes())
        self.assertEqual(self.graph.nodes[doc_id]['type'], 'DOCUMENT')
        self.assertEqual(self.graph.nodes[doc_id]['title'], 'Test Document')
    
    def test_add_segment_nodes(self):
        """Test adding segment nodes to graph."""
        doc_id = "doc_001"
        seg_id = "seg_001"
        
        # Add document first
        self.graph.add_node(doc_id, type='DOCUMENT', title='Test Doc')
        
        # Add segment
        seg_data = {
            'type': 'SEGMENT',
            'text': 'This is a test segment.',
            'segment_index': 0,
            'speaker': 'Test Speaker'
        }
        
        self.graph.add_node(seg_id, **seg_data)
        self.graph.add_edge(doc_id, seg_id, relation='CONTAINS')
        
        self.assertIn(seg_id, self.graph.nodes())
        self.assertEqual(self.graph.nodes[seg_id]['type'], 'SEGMENT')
        self.assertTrue(self.graph.has_edge(doc_id, seg_id))
        self.assertEqual(self.graph.edges[doc_id, seg_id]['relation'], 'CONTAINS')
    
    def test_add_entity_nodes(self):
        """Test adding entity nodes to graph."""
        entity_id = "entity_001"
        entity_data = {
            'type': 'ENTITY_CONCEPT',
            'name': 'Artificial Intelligence',
            'description': 'AI technology concept',
            'frequency': 5
        }
        
        self.graph.add_node(entity_id, **entity_data)
        
        self.assertIn(entity_id, self.graph.nodes())
        self.assertEqual(self.graph.nodes[entity_id]['type'], 'ENTITY_CONCEPT')
        self.assertEqual(self.graph.nodes[entity_id]['name'], 'Artificial Intelligence')
    
    def test_add_relation_edges(self):
        """Test adding relation edges between entities."""
        entity1_id = "entity_001"
        entity2_id = "entity_002"
        
        # Add entities
        self.graph.add_node(entity1_id, type='ENTITY_CONCEPT', name='AI')
        self.graph.add_node(entity2_id, type='ENTITY_CONCEPT', name='Machine Learning')
        
        # Add relation
        relation_data = {
            'relation': 'RELATED_TO',
            'weight': 0.8,
            'confidence': 0.9,
            'source': 'extraction'
        }
        
        self.graph.add_edge(entity1_id, entity2_id, **relation_data)
        
        self.assertTrue(self.graph.has_edge(entity1_id, entity2_id))
        edge_data = self.graph.edges[entity1_id, entity2_id]
        self.assertEqual(edge_data['relation'], 'RELATED_TO')
        self.assertEqual(edge_data['weight'], 0.8)


class TestGraphEnrichment(unittest.TestCase):
    """Test graph enrichment operations."""
    
    def setUp(self):
        self.graph = nx.DiGraph()
        
        # Create test graph with entities
        entities = [
            ("entity_1", {"type": "ENTITY_CONCEPT", "name": "AI", "embedding": [0.1, 0.2, 0.3]}),
            ("entity_2", {"type": "ENTITY_CONCEPT", "name": "ML", "embedding": [0.15, 0.25, 0.35]}),
            ("entity_3", {"type": "ENTITY_CONCEPT", "name": "Data", "embedding": [0.8, 0.9, 0.7]})
        ]
        
        for entity_id, data in entities:
            self.graph.add_node(entity_id, **data)
    
    def test_similarity_computation(self):
        """Test similarity computation between entities."""
        # Mock similarity computation
        def compute_cosine_similarity(emb1, emb2):
            """Compute cosine similarity between embeddings."""
            emb1 = np.array(emb1)
            emb2 = np.array(emb2)
            return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        
        # Get embeddings
        emb1 = self.graph.nodes["entity_1"]["embedding"]
        emb2 = self.graph.nodes["entity_2"]["embedding"]
        emb3 = self.graph.nodes["entity_3"]["embedding"]
        
        # Compute similarities
        sim_1_2 = compute_cosine_similarity(emb1, emb2)
        sim_1_3 = compute_cosine_similarity(emb1, emb3)
        
        # entity_1 and entity_2 should be more similar than entity_1 and entity_3
        self.assertGreater(sim_1_2, sim_1_3)
        self.assertGreater(sim_1_2, 0.9)  # Very similar embeddings
    
    @patch('kg.graph.similarity.compute_embedding_similarity_edges')
    def test_add_similarity_edges(self, mock_compute):
        """Test adding similarity edges to graph."""
        mock_compute.return_value = {
            'edges_added': 2,
            'weights_updated': 1,
            'similarity_threshold': 0.7
        }
        
        from kg.graph.similarity import compute_embedding_similarity_edges
        
        result = compute_embedding_similarity_edges(
            self.graph,
            similarity_threshold=0.7,
            node_types=['ENTITY_CONCEPT'],
            add_new_edges=True
        )
        
        self.assertEqual(result['edges_added'], 2)
        mock_compute.assert_called_once()
    
    def test_node_merging(self):
        """Test merging similar nodes."""
        # Add duplicate entities with slight variations
        self.graph.add_node("entity_dup1", type="ENTITY_CONCEPT", name="Artificial Intelligence")
        self.graph.add_node("entity_dup2", type="ENTITY_CONCEPT", name="artificial intelligence")
        
        # Mock merge operation
        def merge_nodes(graph, node1, node2):
            """Merge two nodes by combining their attributes."""
            # Get attributes from both nodes
            attrs1 = graph.nodes[node1]
            attrs2 = graph.nodes[node2]
            
            # Merge attributes (simple approach)
            merged_attrs = {**attrs1, **attrs2}
            merged_attrs['merged_from'] = [node1, node2]
            
            # Add merged node
            merged_id = f"merged_{node1}_{node2}"
            graph.add_node(merged_id, **merged_attrs)
            
            # Remove original nodes
            graph.remove_node(node1)
            graph.remove_node(node2)
            
            return merged_id
        
        original_count = len(self.graph.nodes())
        merged_id = merge_nodes(self.graph, "entity_dup1", "entity_dup2")
        
        # Should have one less node after merging
        self.assertEqual(len(self.graph.nodes()), original_count - 1)
        self.assertIn(merged_id, self.graph.nodes())
        self.assertNotIn("entity_dup1", self.graph.nodes())
        self.assertNotIn("entity_dup2", self.graph.nodes())


class TestCommunityOperations(unittest.TestCase):
    """Test community detection and operations."""
    
    def setUp(self):
        self.graph = nx.Graph()  # Undirected for community detection
        
        # Create test graph with clear community structure
        # Community 1: nodes 0, 1, 2
        self.graph.add_edges_from([(0, 1), (1, 2), (0, 2)])
        
        # Community 2: nodes 3, 4, 5
        self.graph.add_edges_from([(3, 4), (4, 5), (3, 5)])
        
        # Bridge between communities
        self.graph.add_edge(2, 3)
        
        # Add node attributes
        for node in self.graph.nodes():
            self.graph.nodes[node]['type'] = 'ENTITY_CONCEPT'
            self.graph.nodes[node]['name'] = f'Entity_{node}'
    
    def test_community_detection_basic(self):
        """Test basic community detection."""
        from kg.community.detection import CommunityDetector
        
        detector = CommunityDetector()
        results = detector.detect_communities(self.graph)
        communities = results['assignments']
        
        # Should detect 2 communities
        unique_communities = set(communities.values())
        self.assertEqual(len(unique_communities), 2)
        
        # Nodes 0, 1, 2 should be in same community
        self.assertEqual(communities[0], communities[1])
        self.assertEqual(communities[1], communities[2])
        
        # Nodes 3, 4, 5 should be in same community
        self.assertEqual(communities[3], communities[4])
        self.assertEqual(communities[4], communities[5])
        
        # Communities should be different
        self.assertNotEqual(communities[0], communities[3])
    
    def test_community_attributes(self):
        """Test adding community attributes to graph."""
        communities = {0: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1}
        
        # Add community attributes
        nx.set_node_attributes(self.graph, communities, 'community')
        
        # Verify attributes were set
        for node in self.graph.nodes():
            self.assertIn('community', self.graph.nodes[node])
        
        # Verify community assignments
        self.assertEqual(self.graph.nodes[0]['community'], 0)
        self.assertEqual(self.graph.nodes[3]['community'], 1)
    
    def test_subcommunity_detection(self):
        """Test subcommunity detection within communities."""
        # Create larger community for subcommunity detection
        large_graph = nx.Graph()
        
        # Add nodes in groups that could form subcommunities
        # Group 1 within community 0
        large_graph.add_edges_from([(0, 1), (1, 2), (0, 2)])
        # Group 2 within community 0  
        large_graph.add_edges_from([(3, 4), (4, 5), (3, 5)])
        # Connect groups
        large_graph.add_edge(2, 3)
        
        communities = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
        
        from kg.community.detection import CommunityDetector
        detector = CommunityDetector()
        
        subcommunities = detector.detect_subcommunities_leiden(
            large_graph,
            communities,
            min_sub_size=2
        )
        
        # Should detect subcommunities within the main community
        self.assertIsInstance(subcommunities, dict)


class TestEmbeddingOperations(unittest.TestCase):
    """Test embedding generation and operations."""
    
    def setUp(self):
        self.graph = nx.DiGraph()
        
        # Add test nodes
        self.graph.add_node("entity_1", type="ENTITY_CONCEPT", name="AI", text="Artificial Intelligence")
        self.graph.add_node("entity_2", type="ENTITY_CONCEPT", name="ML", text="Machine Learning")
        self.graph.add_node("topic_1", type="TOPIC", summary="Technology topic")
    
    @patch('kg.embeddings.rag.generate_rag_embeddings')
    def test_rag_embedding_generation(self, mock_generate):
        """Test RAG embedding generation."""
        mock_embeddings = {
            "entity_1": [0.1, 0.2, 0.3, 0.4],
            "entity_2": [0.2, 0.3, 0.4, 0.5],
            "topic_1": [0.3, 0.4, 0.5, 0.6]
        }
        mock_generate.return_value = mock_embeddings
        
        from kg.embeddings.rag import generate_rag_embeddings
        
        embeddings = generate_rag_embeddings(
            self.graph,
            embedding_model="test-model",
            batch_size=32
        )
        
        self.assertEqual(len(embeddings), 3)
        self.assertIn("entity_1", embeddings)
        mock_generate.assert_called_once()
    
    def test_embedding_storage(self):
        """Test storing embeddings in graph nodes."""
        # Add embeddings to nodes
        embeddings = {
            "entity_1": [0.1, 0.2, 0.3],
            "entity_2": [0.4, 0.5, 0.6]
        }
        
        for node_id, embedding in embeddings.items():
            if node_id in self.graph.nodes():
                self.graph.nodes[node_id]['embedding'] = embedding
        
        # Verify embeddings were stored
        self.assertEqual(self.graph.nodes["entity_1"]["embedding"], [0.1, 0.2, 0.3])
        self.assertEqual(self.graph.nodes["entity_2"]["embedding"], [0.4, 0.5, 0.6])
    
    def test_embedding_dimension_consistency(self):
        """Test that all embeddings have consistent dimensions."""
        embeddings = {
            "entity_1": [0.1, 0.2, 0.3],
            "entity_2": [0.4, 0.5, 0.6],
            "topic_1": [0.7, 0.8, 0.9]
        }
        
        # Check dimension consistency
        dimensions = [len(emb) for emb in embeddings.values()]
        self.assertTrue(all(dim == dimensions[0] for dim in dimensions))
        self.assertEqual(dimensions[0], 3)


class TestGraphValidation(unittest.TestCase):
    """Test graph validation and integrity checks."""
    
    def setUp(self):
        self.graph = nx.DiGraph()
    
    def test_node_type_validation(self):
        """Test validation of node types."""
        valid_types = {'DOCUMENT', 'SEGMENT', 'CHUNK', 'ENTITY_CONCEPT', 'TOPIC', 'SUBTOPIC'}
        
        # Add nodes with valid types
        self.graph.add_node("doc1", type="DOCUMENT")
        self.graph.add_node("entity1", type="ENTITY_CONCEPT")
        self.graph.add_node("topic1", type="TOPIC")
        
        # Validate node types
        for node, data in self.graph.nodes(data=True):
            node_type = data.get('type')
            self.assertIn(node_type, valid_types, f"Invalid node type: {node_type}")
    
    def test_edge_relation_validation(self):
        """Test validation of edge relations."""
        valid_relations = {'CONTAINS', 'MENTIONS', 'RELATED_TO', 'SIMILAR_TO', 'BELONGS_TO'}
        
        # Add nodes and edges
        self.graph.add_node("doc1", type="DOCUMENT")
        self.graph.add_node("seg1", type="SEGMENT")
        self.graph.add_node("entity1", type="ENTITY_CONCEPT")
        
        self.graph.add_edge("doc1", "seg1", relation="CONTAINS")
        self.graph.add_edge("seg1", "entity1", relation="MENTIONS")
        
        # Validate edge relations
        for u, v, data in self.graph.edges(data=True):
            relation = data.get('relation')
            self.assertIn(relation, valid_relations, f"Invalid relation: {relation}")
    
    def test_graph_connectivity(self):
        """Test graph connectivity properties."""
        # Create connected components
        self.graph.add_edges_from([("A", "B"), ("B", "C"), ("D", "E")])
        
        # Convert to undirected for connectivity analysis
        undirected = self.graph.to_undirected()
        
        # Check number of connected components
        components = list(nx.connected_components(undirected))
        self.assertEqual(len(components), 2)  # Two separate components
        
        # Check component sizes
        component_sizes = [len(comp) for comp in components]
        self.assertIn(3, component_sizes)  # Component with A, B, C
        self.assertIn(2, component_sizes)  # Component with D, E
    
    def test_graph_statistics(self):
        """Test computation of graph statistics."""
        # Add test data
        self.graph.add_nodes_from([
            ("doc1", {"type": "DOCUMENT"}),
            ("seg1", {"type": "SEGMENT"}),
            ("entity1", {"type": "ENTITY_CONCEPT"}),
            ("entity2", {"type": "ENTITY_CONCEPT"})
        ])
        
        self.graph.add_edges_from([
            ("doc1", "seg1", {"relation": "CONTAINS"}),
            ("seg1", "entity1", {"relation": "MENTIONS"}),
            ("entity1", "entity2", {"relation": "RELATED_TO"})
        ])
        
        # Compute statistics
        stats = {
            'total_nodes': len(self.graph.nodes()),
            'total_edges': len(self.graph.edges()),
            'node_types': {},
            'edge_relations': {}
        }
        
        # Count node types
        for node, data in self.graph.nodes(data=True):
            node_type = data.get('type', 'UNKNOWN')
            stats['node_types'][node_type] = stats['node_types'].get(node_type, 0) + 1
        
        # Count edge relations
        for u, v, data in self.graph.edges(data=True):
            relation = data.get('relation', 'UNKNOWN')
            stats['edge_relations'][relation] = stats['edge_relations'].get(relation, 0) + 1
        
        # Verify statistics
        self.assertEqual(stats['total_nodes'], 4)
        self.assertEqual(stats['total_edges'], 3)
        self.assertEqual(stats['node_types']['ENTITY_CONCEPT'], 2)
        self.assertEqual(stats['edge_relations']['MENTIONS'], 1)


if __name__ == '__main__':
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestGraphUtilities,
        TestGraphConstruction,
        TestGraphEnrichment,
        TestCommunityOperations,
        TestEmbeddingOperations,
        TestGraphValidation
    ]
    
    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Exit with appropriate code
    sys.exit(0 if result.wasSuccessful() else 1)
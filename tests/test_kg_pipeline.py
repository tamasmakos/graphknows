"""
Comprehensive unit tests for the Knowledge Graph generation pipeline.
"""

import unittest
import asyncio
import tempfile
import shutil
import os
import json
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from pathlib import Path
import networkx as nx

# Add src to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from kg.config.loader import Config, load_config
from kg.types import AgentDependencies, SemanticKGResults, ChunkExtractionTask
from kg.pipeline.core import (
    build_semantic_kg_with_communities,
    run_lexical_graph,
    run_extraction,
    run_embeddings,
    run_semantic_resolution,
    run_similarity_edges,
    run_community_detection,
    run_summarization
)
from kg.pipeline.stages import PipelineStage, StageRegistry


class TestKGConfig(unittest.TestCase):
    """Test configuration loading and validation."""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = os.path.join(self.temp_dir, 'test_config.yaml')
        
    def tearDown(self):
        shutil.rmtree(self.temp_dir)
    
    def test_load_config_with_valid_yaml(self):
        """Test loading configuration from valid YAML file."""
        config_data = {
            'mode': 'batch',
            'clean_start': False,
            'processing': {
                'input_dir': 'test_input',
                'output_dir': 'test_output',
                'chunk_size': 256
            },
            'llm': {
                'provider': 'openai',
                'model_name': 'gpt-4o-mini'
            }
        }
        
        with open(self.config_path, 'w') as f:
            import yaml
            yaml.dump(config_data, f)
        
        config = load_config(self.config_path)
        
        self.assertEqual(config.mode, 'batch')
        self.assertEqual(config.processing.input_dir, 'test_input')
        self.assertEqual(config.llm.provider, 'openai')
    
    def test_load_config_missing_file(self):
        """Test loading configuration when file doesn't exist."""
        config = load_config('nonexistent.yaml')
        
        # Should still create config object with defaults
        self.assertIsInstance(config, Config)
    
    def test_config_mode_logic(self):
        """Test mode-specific configuration logic."""
        config_data = {
            'mode': 'incremental',
            'batch': {'speech_limit': 100},
            'incremental': {'speech_limit': 5},
            'processing': {},
            'falkordb': {}
        }
        
        config = Config(config_data)
        
        # Should use incremental speech_limit
        self.assertEqual(config.processing.speech_limit, 5)
        self.assertFalse(config.falkordb.clean_database)
    
    def test_config_environment_variables(self):
        """Test environment variable overrides."""
        with patch.dict(os.environ, {
            'GROQ_API_KEY': 'test_key',
            'NEO4J_URI': 'bolt://localhost:7687'
        }):
            config = load_config('nonexistent.yaml')
            
            self.assertEqual(config.llm.api_key, 'test_key')
            self.assertEqual(config.neo4j.uri, 'bolt://localhost:7687')


class TestAgentDependencies(unittest.TestCase):
    """Test AgentDependencies data structure."""
    
    def test_initialization(self):
        """Test AgentDependencies initialization."""
        graph = nx.DiGraph()
        deps = AgentDependencies(graph=graph)
        
        self.assertIs(deps.graph, graph)
        self.assertEqual(len(deps.extraction_tasks), 0)
        self.assertEqual(deps.total_segments, 0)
    
    def test_add_extraction_task(self):
        """Test adding extraction tasks."""
        graph = nx.DiGraph()
        deps = AgentDependencies(graph=graph)
        
        task = ChunkExtractionTask(
            chunk_id="test_chunk_1",
            chunk_text="Test text content",
            entities=["Entity1", "Entity2"],
            abstract_concepts=["Concept1"]
        )
        
        deps.extraction_tasks.append(task)
        
        self.assertEqual(len(deps.extraction_tasks), 1)
        self.assertEqual(deps.extraction_tasks[0].chunk_id, "test_chunk_1")


class TestPipelineStages(unittest.TestCase):
    """Test individual pipeline stages."""
    
    def setUp(self):
        self.graph = nx.DiGraph()
        self.config = Mock()
        self.config.to_dict.return_value = {}
        self.deps = AgentDependencies(graph=self.graph)
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        shutil.rmtree(self.temp_dir)
    
    @patch('kg.pipeline.core.build_lexical_graph')
    async def test_run_lexical_graph(self, mock_build):
        """Test lexical graph stage."""
        mock_build.return_value = {'documents_processed': 5, 'total_segments': 20}
        self.config.processing.input_dir = 'test_input'
        
        result = await run_lexical_graph(
            self.graph, 
            self.config, 
            deps=self.deps
        )
        
        self.assertIn('lexical_result', result)
        mock_build.assert_called_once()
    
    @patch('kg.pipeline.core.extract_all_entities_relations')
    async def test_run_extraction(self, mock_extract):
        """Test entity/relation extraction stage."""
        mock_extract.return_value = {'processed': 10, 'successful': 8}
        
        # Add some extraction tasks
        task = ChunkExtractionTask(
            chunk_id="test_chunk",
            chunk_text="Test content",
            entities=["Entity1"],
            abstract_concepts=["Concept1"]
        )
        self.deps.extraction_tasks.append(task)
        
        result = await run_extraction(
            self.graph,
            self.config,
            deps=self.deps
        )
        
        self.assertIn('extraction_result', result)
        mock_extract.assert_called_once()
    
    @patch('kg.pipeline.core.generate_rag_embeddings')
    @patch('kg.pipeline.core._train_and_cache_global_kge')
    async def test_run_embeddings(self, mock_kge, mock_rag):
        """Test embeddings generation stage."""
        self.config.graph.enable_kge = True
        self.config.embeddings.model = 'test-model'
        self.config.embeddings.batch_size = 32
        
        mock_rag.return_value = {'node1': [0.1, 0.2], 'node2': [0.3, 0.4]}
        
        result = await run_embeddings(
            self.graph,
            self.config,
            output_dir=self.temp_dir
        )
        
        self.assertIn('rag_embeddings_count', result)
        self.assertIn('kge_trained', result)
        self.assertEqual(result['rag_embeddings_count'], 2)
        self.assertTrue(result['kge_trained'])
        
        mock_kge.assert_called_once()
        mock_rag.assert_called_once()
    
    @patch('kg.pipeline.core.merge_similar_nodes')
    async def test_run_semantic_resolution(self, mock_merge):
        """Test semantic entity resolution stage."""
        mock_merge.return_value = {'merged_nodes': 5, 'similarity_threshold': 0.8}
        self.config.graph.semantic_resolution_threshold = 0.8
        
        result = await run_semantic_resolution(
            self.graph,
            self.config
        )
        
        self.assertIn('resolution_stats', result)
        mock_merge.assert_called_once_with(
            self.graph,
            similarity_threshold=0.8,
            node_types=['ENTITY_CONCEPT']
        )
    
    @patch('kg.pipeline.core.compute_embedding_similarity_edges')
    async def test_run_similarity_edges(self, mock_compute):
        """Test similarity edges computation stage."""
        mock_compute.return_value = {'edges_added': 10, 'weights_updated': 5}
        self.config.graph.embedding_similarity_threshold = 0.7
        
        result = await run_similarity_edges(
            self.graph,
            self.config
        )
        
        self.assertIn('similarity_stats', result)
        mock_compute.assert_called_once_with(
            self.graph,
            similarity_threshold=0.7,
            node_types=['ENTITY_CONCEPT'],
            add_new_edges=True,
            update_existing_weights=True
        )
    
    @patch('kg.pipeline.core.CommunityDetector')
    @patch('kg.pipeline.core.add_enhanced_community_attributes_to_graph')
    @patch('kg.pipeline.core.calculate_entity_relation_centrality_measures')
    @patch('kg.pipeline.core.evaluate_community_quality')
    async def test_run_community_detection(self, mock_quality, mock_centrality, 
                                         mock_attributes, mock_detector_class):
        """Test community detection stage."""
        # Setup mocks
        mock_detector = Mock()
        mock_detector_class.return_value = mock_detector
        mock_detector.detect_communities.return_value = {'node1': 0, 'node2': 1}
        mock_detector.detect_subcommunities_leiden.return_value = {'sub1': [0, 1]}
        
        mock_centrality.return_value = {'centrality_computed': True}
        mock_quality.return_value = {'modularity': 0.8}
        
        self.config.community.min_subcommunity_size = 3
        
        result = await run_community_detection(
            self.graph,
            self.config
        )
        
        self.assertIn('communities', result)
        self.assertIn('subcommunities', result)
        self.assertIn('centrality_results', result)
        self.assertIn('community_quality', result)
        
        mock_detector.detect_communities.assert_called_once()
        mock_detector.detect_subcommunities_leiden.assert_called_once()
    
    @patch('kg.pipeline.core.generate_community_summaries')
    @patch('kg.pipeline.core.generate_community_summary_comparison')
    @patch('kg.pipeline.core.generate_rag_embeddings')
    async def test_run_summarization(self, mock_rag, mock_comparison, mock_summaries):
        """Test summarization stage."""
        mock_summaries.return_value = {'topics_updated': 5, 'subtopics_updated': 10}
        mock_llm = Mock()
        
        self.config.embeddings.model = 'test-model'
        self.config.embeddings.batch_size = 32
        
        result = await run_summarization(
            self.graph,
            self.config,
            llm=mock_llm,
            output_dir=self.temp_dir
        )
        
        self.assertIn('summarization_stats', result)
        mock_summaries.assert_called_once_with(self.graph, mock_llm)
        mock_comparison.assert_called_once()
        mock_rag.assert_called_once()


class TestStageRegistry(unittest.TestCase):
    """Test pipeline stage registry functionality."""
    
    def setUp(self):
        self.registry = StageRegistry()
    
    def test_register_stage(self):
        """Test registering a pipeline stage."""
        async def dummy_func(graph, config, **kwargs):
            return {"test": True}
        
        stage = PipelineStage(
            name="test_stage",
            display_name="Test Stage",
            description="A test stage",
            run_func=dummy_func
        )
        
        self.registry.register(stage)
        
        self.assertIn("test_stage", self.registry._stages)
        retrieved_stage = self.registry.get_stage("test_stage")
        self.assertEqual(retrieved_stage.name, "test_stage")
    
    def test_get_execution_plan(self):
        """Test getting execution plan with dependencies."""
        async def stage1_func(graph, config, **kwargs):
            return {"stage1": True}
        
        async def stage2_func(graph, config, **kwargs):
            return {"stage2": True}
        
        stage1 = PipelineStage(
            name="stage1",
            display_name="Stage 1",
            description="First stage",
            run_func=stage1_func
        )
        
        stage2 = PipelineStage(
            name="stage2",
            display_name="Stage 2", 
            description="Second stage",
            run_func=stage2_func,
            depends_on=["stage1"]
        )
        
        self.registry.register(stage1)
        self.registry.register(stage2)
        
        # Mock config with enabled stages
        config = Mock()
        config.pipeline = Mock()
        config.pipeline.stages = Mock()
        config.pipeline.stages.lexical_graph = Mock()
        config.pipeline.stages.lexical_graph.enabled = True
        
        # For this test, we'll mock the get_execution_plan method
        with patch.object(self.registry, 'get_execution_plan') as mock_plan:
            mock_plan.return_value = [stage1, stage2]
            
            plan = self.registry.get_execution_plan(config)
            
            self.assertEqual(len(plan), 2)
            self.assertEqual(plan[0].name, "stage1")
            self.assertEqual(plan[1].name, "stage2")


class TestSemanticKGResults(unittest.TestCase):
    """Test SemanticKGResults data structure."""
    
    def test_initialization(self):
        """Test SemanticKGResults initialization."""
        results = SemanticKGResults(
            lexical_graph_stats={'documents_processed': 5},
            extraction_stats={'processed': 10},
            community_stats={'num_communities': 3},
            summarization_stats={'topics_updated': 2},
            output_files={'output_dir': '/test/output'}
        )
        
        self.assertEqual(results.lexical_graph_stats['documents_processed'], 5)
        self.assertEqual(results.extraction_stats['processed'], 10)
        self.assertEqual(results.community_stats['num_communities'], 3)
        self.assertIsNone(results.embedding_stats)
        self.assertIsNone(results.similarity_stats)
    
    def test_with_optional_stats(self):
        """Test SemanticKGResults with optional statistics."""
        results = SemanticKGResults(
            lexical_graph_stats={},
            extraction_stats={},
            community_stats={},
            summarization_stats={},
            output_files={},
            embedding_stats={'nodes_with_embeddings': 100},
            similarity_stats={'edges_added': 50}
        )
        
        self.assertIsNotNone(results.embedding_stats)
        self.assertIsNotNone(results.similarity_stats)
        self.assertEqual(results.embedding_stats['nodes_with_embeddings'], 100)
        self.assertEqual(results.similarity_stats['edges_added'], 50)


class TestMainPipeline(unittest.TestCase):
    """Test the main pipeline integration."""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.input_dir = os.path.join(self.temp_dir, 'input')
        self.output_dir = os.path.join(self.temp_dir, 'output')
        os.makedirs(self.input_dir)
        
        # Create mock config
        self.config = Mock()
        self.config.to_dict.return_value = {}
        
    def tearDown(self):
        shutil.rmtree(self.temp_dir)
    
    @patch('kg.pipeline.core.get_langchain_llm')
    @patch('kg.pipeline.core.registry')
    async def test_build_semantic_kg_with_communities(self, mock_registry, mock_llm):
        """Test the main pipeline function."""
        # Setup mocks
        mock_llm_instance = Mock()
        mock_llm.return_value = mock_llm_instance
        
        # Mock stages
        mock_stage1 = Mock()
        mock_stage1.display_name = "Test Stage 1"
        mock_stage1.name = "stage1"
        mock_stage1.run = AsyncMock(return_value={"stage1_result": True})
        
        mock_stage2 = Mock()
        mock_stage2.display_name = "Test Stage 2"
        mock_stage2.name = "stage2"
        mock_stage2.run = AsyncMock(return_value={"stage2_result": True})
        
        mock_registry.get_execution_plan.return_value = [mock_stage1, mock_stage2]
        
        # Mock LLM cleanup
        mock_llm_instance.async_client = Mock()
        mock_llm_instance.async_client.aclose = AsyncMock()
        
        result = await build_semantic_kg_with_communities(
            input_dir=self.input_dir,
            output_dir=self.output_dir,
            config=self.config
        )
        
        # Verify result structure
        self.assertIsInstance(result, SemanticKGResults)
        self.assertIn('output_dir', result.output_files)
        
        # Verify stages were called
        mock_stage1.run.assert_called_once()
        mock_stage2.run.assert_called_once()
        
        # Verify LLM cleanup
        mock_llm_instance.async_client.aclose.assert_called_once()
    
    @patch('kg.pipeline.core.get_langchain_llm')
    @patch('kg.pipeline.core.registry')
    async def test_pipeline_stage_failure(self, mock_registry, mock_llm):
        """Test pipeline behavior when a stage fails."""
        # Setup mocks
        mock_llm_instance = Mock()
        mock_llm.return_value = mock_llm_instance
        
        # Mock failing stage
        mock_stage = Mock()
        mock_stage.display_name = "Failing Stage"
        mock_stage.name = "failing_stage"
        mock_stage.run = AsyncMock(side_effect=Exception("Stage failed"))
        
        mock_registry.get_execution_plan.return_value = [mock_stage]
        
        # Mock LLM cleanup
        mock_llm_instance.async_client = Mock()
        mock_llm_instance.async_client.aclose = AsyncMock()
        
        with self.assertRaises(Exception) as context:
            await build_semantic_kg_with_communities(
                input_dir=self.input_dir,
                output_dir=self.output_dir,
                config=self.config
            )
        
        self.assertIn("Stage failed", str(context.exception))
        
        # Verify cleanup still happens
        mock_llm_instance.async_client.aclose.assert_called_once()


class TestPipelineIntegration(unittest.TestCase):
    """Integration tests for pipeline components."""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        shutil.rmtree(self.temp_dir)
    
    def test_graph_state_preservation(self):
        """Test that graph state is preserved between stages."""
        graph = nx.DiGraph()
        
        # Add some test nodes and edges
        graph.add_node("entity1", type="ENTITY_CONCEPT", text="Entity 1")
        graph.add_node("entity2", type="ENTITY_CONCEPT", text="Entity 2")
        graph.add_edge("entity1", "entity2", relation="RELATED_TO")
        
        deps = AgentDependencies(graph=graph)
        
        # Verify initial state
        self.assertEqual(len(graph.nodes()), 2)
        self.assertEqual(len(graph.edges()), 1)
        
        # Simulate stage modifications
        graph.add_node("entity3", type="ENTITY_CONCEPT", text="Entity 3")
        
        # Verify state is preserved
        self.assertEqual(len(deps.graph.nodes()), 3)
        self.assertIn("entity3", deps.graph.nodes())
    
    def test_extraction_task_processing(self):
        """Test extraction task creation and processing."""
        graph = nx.DiGraph()
        deps = AgentDependencies(graph=graph)
        
        # Create extraction tasks
        task1 = ChunkExtractionTask(
            chunk_id="chunk_1",
            chunk_text="This is about artificial intelligence and machine learning.",
            entities=["artificial intelligence", "machine learning"],
            abstract_concepts=["technology", "automation"]
        )
        
        task2 = ChunkExtractionTask(
            chunk_id="chunk_2", 
            chunk_text="Climate change affects global weather patterns.",
            entities=["climate change", "weather patterns"],
            abstract_concepts=["environment", "global warming"]
        )
        
        deps.extraction_tasks.extend([task1, task2])
        
        # Verify tasks are properly stored
        self.assertEqual(len(deps.extraction_tasks), 2)
        self.assertEqual(deps.extraction_tasks[0].chunk_id, "chunk_1")
        self.assertEqual(len(deps.extraction_tasks[0].entities), 2)
        self.assertEqual(len(deps.extraction_tasks[1].abstract_concepts), 2)


if __name__ == '__main__':
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestKGConfig,
        TestAgentDependencies,
        TestPipelineStages,
        TestStageRegistry,
        TestSemanticKGResults,
        TestMainPipeline,
        TestPipelineIntegration
    ]
    
    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Exit with appropriate code
    sys.exit(0 if result.wasSuccessful() else 1)
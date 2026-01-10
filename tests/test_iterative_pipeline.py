"""
Tests for Iterative Knowledge Graph Pipeline.
"""
import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock, MagicMock, mock_open
from pathlib import Path
import shutil
import networkx as nx

# Import the function to test
from src.kg.pipeline.iterative import run_iterative_pipeline

@pytest.fixture
def mock_config_loader(tmp_path):
    with patch('src.kg.config.loader.load_config') as mock_load:
        config_mock = Mock()
        
        # Setup config structure matching what run_iterative_pipeline expects
        config_mock.processing.input_dir = str(tmp_path / "input")
        config_mock.processing.output_dir = str(tmp_path / "output")
        config_mock.processing.file_pattern = "*.txt"
        config_mock.falkordb.upload_enabled = False
        config_mock.falkordb.clean_database = False
        config_mock.incremental.state_file = "state.json"
        
        # Mode settings
        config_mock.mode = 'incremental'
        config_mock.incremental.speech_limit = 10
        config_mock.incremental.max_documents = 2
        
        config_mock.to_dict.return_value = {
            "processing": {"input_dir": str(tmp_path / "input")},
            "llm": {"provider": "test"},
            "embeddings": {"model_name": "test-embed"}
        }
        
        mock_load.return_value = config_mock
        yield mock_load, config_mock

@pytest.fixture
def mock_components():
    """Mock all external dependencies of the pipeline."""
    with patch('src.kg.pipeline.iterative.IterativeGraphBuilder') as mock_builder_cls, \
         patch('src.kg.pipeline.iterative.build_lexical_graph') as mock_lexical, \
         patch('src.kg.pipeline.iterative.extract_all_entities_relations') as mock_extract, \
         patch('src.kg.pipeline.iterative.generate_rag_embeddings') as mock_embed, \
         patch('src.kg.pipeline.iterative.get_extractor') as mock_get_extractor, \
         patch('src.kg.pipeline.iterative.get_langchain_llm') as mock_get_llm, \
         patch('src.kg.pipeline.iterative.CommunityDetector') as mock_detector_cls, \
         patch('src.kg.pipeline.iterative.generate_community_summaries', new=AsyncMock(return_value={})) as mock_summarize, \
         patch('logging.FileHandler') as mock_file_handler_cls: # Mock FileHandler to prevent file creation
        
        # Setup FileHandler mock to have a valid level (int) to avoid TypeError in logging
        mock_file_handler = mock_file_handler_cls.return_value
        import logging
        mock_file_handler.level = logging.INFO
        
        # Extractor Mock
        mock_extractor = mock_get_extractor.return_value
        mock_extractor.close = AsyncMock()
        
        # LLM Mock
        mock_llm = mock_get_llm.return_value
        
        # Builder Mock
        mock_builder = mock_builder_cls.return_value
        mock_builder.state.processed_documents = []
        mock_builder.get_new_documents.return_value = ["doc1", "doc2"]
        mock_builder.merge_graph_incrementally.return_value = {
            "nodes_merged": 10, "relationships_merged": 5
        }
        mock_builder.calculate_and_get_metrics.return_value = {"total_nodes": 100}
        
        # Community Detector Mock
        mock_detector = mock_detector_cls.return_value
        mock_detector.detect_communities.return_value = {'assignments': {}, 'community_count': 1}
        mock_detector.detect_subcommunities_leiden.return_value = {}
        
        # Async mocks
        mock_lexical.return_value = {}
        mock_extract.return_value = {}
        
        yield {
            'builder': mock_builder,
            'lexical': mock_lexical,
            'extract': mock_extract,
            'embed': mock_embed
        }

@pytest.mark.asyncio
async def test_run_iterative_pipeline_success(mock_config_loader, mock_components, tmp_path):
    """Test successful execution of the iterative pipeline."""
    
    # We mock path ops but let the directories be created in tmp_path by mocks if needed, 
    # but run_iterative_pipeline uses Path() which are real objects.
    # We already injected tmp_path into config.
    # But run_iterative_pipeline calls mkdir on them.
    
    # If we use real Path objects, we don't need to mock mkdir unless we want to spy.
    # The glob needs mocking because we don't have real input files.
    
    with patch('pathlib.Path.glob') as mock_glob, \
         patch('builtins.open', mock_open()), \
         patch('json.dump'):
        
        # Setup files
        f1 = MagicMock()
        f1.stem = "doc1"
        f1.name = "doc1.txt"
        f1.__lt__.side_effect = lambda other: f1.stem < other.stem
        
        f2 = MagicMock()
        f2.stem = "doc2"
        f2.name = "doc2.txt"
        f2.__lt__.side_effect = lambda other: f2.stem < other.stem
        
        mock_glob.return_value = [f1, f2]
        
        # Run
        stats = await run_iterative_pipeline("config.yaml")
        
        assert stats['documents_processed'] == 2
        assert stats['nodes_merged'] == 20 # 10 * 2 documents
        
        # Verify stages called
        assert mock_components['lexical'].call_count == 2
        assert mock_components['extract'].call_count == 2
        assert mock_components['embed'].call_count == 2

@pytest.mark.asyncio
async def test_run_iterative_pipeline_no_new_docs(mock_config_loader, mock_components):
    """Test pipeline when no new documents are found."""
    mock_components['builder'].get_new_documents.return_value = []
    
    with patch('pathlib.Path.glob', return_value=[]):
        
        stats = await run_iterative_pipeline("config.yaml")
        
        assert stats['status'] == 'up_to_date'
        assert stats['documents_processed'] == 0

"""
Integration tests for KG pipeline configuration system.
"""

import pytest
import os
import yaml
import tempfile
from pathlib import Path
from src.kg.config import load_config, Config


class TestConfigLoading:
    """Test configuration loading from YAML."""
    
    def test_load_default_config(self):
        """Test loading default config.yaml."""
        config = load_config("config.yaml")
        
        assert isinstance(config, Config)
        assert config.processing.speech_limit == 1
        assert config.llm.model == "llama-3.3-70b-versatile"
        assert config.embeddings.model == "all-MiniLM-L6-v2"
    
    def test_load_custom_config(self, tmp_path):
        """Test loading custom config file."""
        config_data = {
            "processing": {
                "speech_limit": 10,
                "chunk_size": 5
            },
            "llm": {
                "model": "test-model",
                "temperature": 0.5
            }
        }
        
        config_file = tmp_path / "test_config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(config_data, f)
        
        config = load_config(str(config_file))
        
        assert config.processing.speech_limit == 10
        assert config.processing.chunk_size == 5
        assert config.llm.model == "test-model"
        assert config.llm.temperature == 0.5
    
    def test_environment_variable_secrets(self, monkeypatch, tmp_path):
        """Test that secrets are loaded from environment variables."""
        # Set env vars
        monkeypatch.setenv("GROQ_API_KEY", "test_api_key_123")
        monkeypatch.setenv("NEO4J_URI", "bolt://test:7687")
        
        config_file = tmp_path / "test_config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump({}, f)
        
        config = load_config(str(config_file))
        
        assert config.llm.api_key == "test_api_key_123"
        assert config.neo4j.uri == "bolt://test:7687"
    
    def test_output_dir_override(self, tmp_path):
        """Test output directory override."""
        config_file = tmp_path / "test_config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump({"processing": {"output_dir": "original"}}, f)
        
        config = load_config(str(config_file), output_dir_override="overridden")
        
        assert config.processing.output_dir == "overridden"


class TestPipelineStages:
    """Test pipeline stage configuration."""
    
    def test_all_stages_enabled(self, tmp_path):
        """Test config with all stages enabled."""
        config_data = {
            "pipeline": {
                "stages": {
                    "lexical_graph": True,
                    "extraction": True,
                    "embeddings": True,
                    "semantic_resolution": True,
                    "similarity_edges": True,
                    "community_detection": True,
                    "summarization": True,
                    "schema_export": True,
                    "neo4j_upload": True
                }
            }
        }
        
        config_file = tmp_path / "test_config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(config_data, f)
        
        config = load_config(str(config_file))
        
        assert config.pipeline.stages["lexical_graph"] == True
        assert config.pipeline.stages["embeddings"] == True
        assert config.pipeline.stages["neo4j_upload"] == True
    
    def test_selective_stages(self, tmp_path):
        """Test config with selective stages disabled."""
        config_data = {
            "pipeline": {
                "stages": {
                    "lexical_graph": True,
                    "extraction": True,
                    "embeddings": True,
                    "similarity_edges": False,
                    "neo4j_upload": False
                }
            }
        }
        
        config_file = tmp_path / "test_config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(config_data, f)
        
        config = load_config(str(config_file))
        
        assert config.pipeline.stages["similarity_edges"] == False
        assert config.pipeline.stages["neo4j_upload"] == False


class TestKGEToggle:
    """Test Knowledge Graph Embeddings toggle."""
    
    def test_kge_disabled_by_default(self):
        """Test that KGE is disabled by default."""
        config = load_config("config.yaml")
        assert config.graph.enable_kge == False
    
    def test_kge_can_be_enabled(self, tmp_path):
        """Test enabling KGE via config."""
        config_data = {
            "graph": {
                "enable_kge": True
            }
        }
        
        config_file = tmp_path / "test_config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(config_data, f)
        
        config = load_config(str(config_file))
        assert config.graph.enable_kge == True


class TestConfigDefaults:
    """Test configuration defaults."""
    
    def test_processing_defaults(self, tmp_path):
        """Test processing configuration defaults."""
        config_file = tmp_path / "empty_config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump({}, f)
        
        config = load_config(str(config_file))
        
        assert config.processing.speech_limit == 0
        assert config.processing.max_concurrent_extractions == 8
        assert config.processing.chunk_size == 10
        assert config.processing.use_overlapping_chunks == True
    
    def test_llm_defaults(self, tmp_path):
        """Test LLM configuration defaults."""
        config_file = tmp_path / "empty_config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump({}, f)
        
        config = load_config(str(config_file))
        
        assert config.llm.model == "llama-3.3-70b-versatile"
        assert config.llm.temperature == 0.0
    
    def test_graph_defaults(self, tmp_path):
        """Test graph configuration defaults."""
        config_file = tmp_path / "empty_config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump({}, f)
        
        config = load_config(str(config_file))
        
        assert config.graph.embedding_similarity_threshold == 0.85
        assert config.graph.semantic_resolution_threshold == 0.95
        assert config.graph.add_similarity_edges == True
        assert config.graph.enable_semantic_resolution == True
        assert config.graph.enable_kge == False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

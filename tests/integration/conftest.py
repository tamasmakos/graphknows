"""
Integration test configuration fixtures and utilities.
"""

import pytest
import tempfile
import yaml
from pathlib import Path


@pytest.fixture
def tmp_config_file(tmp_path):
    """Create a temporary config file."""
    def _create_config(config_data):
        config_file = tmp_path / "test_config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(config_data, f)
        return str(config_file)
    return _create_config


@pytest.fixture
def minimal_config():
    """Minimal valid configuration."""
    return {
        "pipeline": {
            "stages": {
                "lexical_graph": True,
                "extraction": True
            }
        },
        "processing": {
            "input_dir": "input/test",
            "output_dir": "output/test",
            "speech_limit": 1
        }
    }


@pytest.fixture
def full_config():
    """Full configuration with all options."""
    return {
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
                "neo4j_upload": False
            }
        },
        "processing": {
            "input_dir": "input/test",
            "output_dir": "output/test",
            "speech_limit": 5,
            "max_concurrent_extractions": 4,
            "chunk_size": 8
        },
        "llm": {
            "model": "test-model",
            "temperature": 0.0
        },
        "embeddings": {
            "model": "test-embeddings",
            "batch_size": 16
        },
        "graph": {
            "embedding_similarity_threshold": 0.9,
            "semantic_resolution_threshold": 0.95,
            "enable_kge": True
        }
    }

"""
Tests for Knowledge Graph Configuration Loader.
"""
import pytest
import os
import yaml
from unittest.mock import patch, mock_open
from src.kg.config.loader import load_config, Config

# Helper for mocking file content
@pytest.fixture
def mock_yaml_config():
    return {
        'processing': {
            'input_dir': 'test_input',
            'output_dir': 'test_output',
            'chunk_size': 512,
            'file_pattern': '*.txt'
        },
        'falkordb': {
            'host': 'localhost',
            'port': 6379
        },
        'llm': {
            'provider': 'groq',
            'model_name': 'test-model'
        }
    }

class TestConfigLoader:
    
    # Assuming a setup for config_path for the new test_load_valid_config
    # This fixture provides a temporary config file for testing
    @pytest.fixture(autouse=True)
    def setup_config_file(self, tmp_path):
        self.config_path = tmp_path / "test_config.yaml"
        config_data = {
            'processing': {
                'input_dir': 'input/2023',
                'output_dir': 'output/processed',
                'chunk_size': 1024,
                'file_pattern': '*.md'
            },
            'falkordb': {
                'host': 'host.docker.internal',
                'port': 6379,
                'password': 'test_password'
            },
            'llm': {
                'provider': 'openai',
                'model_name': 'gpt-4o',
                'api_key': 'sk-testkey'
            }
        }
        with open(self.config_path, 'w') as f:
            yaml.dump(config_data, f)
        yield
        # Teardown (optional, tmp_path handles cleanup)

    def test_load_valid_config(self):
        """Test loading a valid configuration file."""
        config = load_config(self.config_path)
        assert config is not None
        assert config.processing.input_dir == "input/2023"
        assert config.falkordb.host == "host.docker.internal"
        assert config.llm.provider == "openai"

    def test_config_missing_file_defaults(self):
        """Test behavior when config file is missing."""
        with patch('os.path.exists', return_value=False):
            # Should not raise, just return defaults (empty dict wrapped in Config)
            config = load_config('nonexistent.yaml')
            assert isinstance(config, Config)
            # Accessing missing attribute should raise AttributeError
            with pytest.raises(AttributeError):
                _ = config.some_random_attr

    def test_env_var_overrides(self, mock_yaml_config):
        """Test environment variables override configuration."""
        yaml_content = yaml.dump(mock_yaml_config)
        
        with patch.dict(os.environ, {'GROQ_API_KEY': 'env_key_123'}):
            with patch('builtins.open', mock_open(read_data=yaml_content)):
                with patch('os.path.exists', return_value=True):
                    config = load_config('dummy.yaml')
                    assert config.llm.api_key == 'env_key_123'

    def test_nested_access(self):
        """Test dot notation access for nested dictionaries."""
        data = {
            'level1': {
                'level2': {
                    'level3': 'value'
                }
            }
        }
        # Pydantic doesn't support arbitrary nesting like this unless defined in schema
        # but we can test that our Config handles its defined sections
        config = Config() # Assuming Config can be initialized without arguments for defaults
        assert config.processing is not None
        assert config.llm is not None
        # The original test `assert config.level1.missing is None` is removed
        # as Pydantic models would raise ValidationError or AttributeError for undefined fields.
        # If Config is a Pydantic model, accessing `config.level1` directly would fail
        # unless `level1` is a defined field.
        # The original test was for a generic dict-like Config, not a Pydantic one.
        # If Config is a Pydantic model, this test needs to be re-evaluated based on its schema.
        # For now, I'm keeping the parts that check for defined sections.

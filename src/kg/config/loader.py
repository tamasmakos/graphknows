"""
Configuration loader for the Knowledge Graph pipeline.
"""

import os
import yaml
import logging
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger(__name__)


class Config:
    """Configuration container with attribute access."""
    
    def __init__(self, config_dict: Dict[str, Any]):
        """Initialize config from dictionary."""
        self._config = config_dict
        
        # Top-level settings
        self.clean_start = config_dict.get('clean_start', False)
        
        # Create namespace objects for each section
        for key, value in config_dict.items():
            if isinstance(value, dict):
                setattr(self, key, self._dict_to_namespace(value))
            else:
                setattr(self, key, value)
    
    def _dict_to_namespace(self, d: Dict) -> Any:
        """Convert dict to namespace object for dot notation access."""
        class Namespace:
            def __init__(self, dictionary):
                for key, value in dictionary.items():
                    if isinstance(value, dict):
                        setattr(self, key, Namespace(value))
                    else:
                        setattr(self, key, value)
            
            def __getattr__(self, item):
                return None
                
        return Namespace(d)
    
    # _apply_mode_logic removed - replaced by direct config management

    def to_dict(self) -> Dict[str, Any]:
        """Return configuration as dictionary."""
        return self._config


def load_config(config_path: str = "config.yaml") -> Config:
    """
    Load configuration from YAML file with overrides.
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        Config object
    """
    config_data = {}
    
    # 1. Load from YAML file
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                yaml_data = yaml.safe_load(f)
                if yaml_data:
                    config_data.update(yaml_data)
            logger.info(f"Loaded configuration from {config_path}")
        except Exception as e:
            logger.error(f"Error loading config file {config_path}: {e}")
            # Continue with defaults
    else:
        logger.warning(f"Config file {config_path} not found, using defaults")
    
    # 2. Apply environment variables for secrets
    # Create sections if they don't exist
    if 'llm' not in config_data: config_data['llm'] = {}

    
    if os.environ.get('GROQ_API_KEY'):
        config_data['llm']['api_key'] = os.environ.get('GROQ_API_KEY')
        

    
    # 3. Create Config object (validates data)
    try:
        config = Config(config_data)
        return config
    except Exception as e:
        logger.error(f"Configuration validation error: {e}")
        raise

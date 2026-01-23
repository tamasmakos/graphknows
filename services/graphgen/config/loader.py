"""
Configuration loader for the Knowledge Graph pipeline.
"""

import os
import yaml
import logging
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger(__name__)


from .schema import Config
from pathlib import Path

def load_config(config_path: str = "config.yaml", output_dir_override: str = None) -> Config:
    """
    Load configuration from YAML file with overrides.
    
    Args:
        config_path: Path to YAML configuration file
        output_dir_override: Optional override for output directory
        
    Returns:
        Config object
    """
    config_data = {}
    
    # 1. Load from YAML file
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                yaml_data = yaml.safe_load(f)
                if yaml_data:
                    config_data.update(yaml_data)
            logger.info(f"Loaded configuration from {config_path}")
        except Exception as e:
            logger.error(f"Error loading config file {config_path}: {e}")
    else:
        logger.warning(f"Config file {config_path} not found, using defaults")
    
    # 2. Apply environment variables for secrets
    if 'llm' not in config_data: config_data['llm'] = {}
    if 'neo4j' not in config_data: config_data['neo4j'] = {}
    
    if os.environ.get('GROQ_API_KEY'):
        config_data['llm']['api_key'] = os.environ.get('GROQ_API_KEY')
    
    if os.environ.get('NEO4J_URI'):
        config_data['neo4j']['uri'] = os.environ.get('NEO4J_URI')

    if os.environ.get('FALKORDB_HOST'):
        if 'falkordb' not in config_data: config_data['falkordb'] = {}
        config_data['falkordb']['host'] = os.environ.get('FALKORDB_HOST')

    if os.environ.get('POSTGRES_HOST'):
        if 'postgres' not in config_data: config_data['postgres'] = {}
        config_data['postgres']['host'] = os.environ.get('POSTGRES_HOST')

    # 3. Apply overrides
    if output_dir_override:
        if 'processing' not in config_data: config_data['processing'] = {}
        config_data['processing']['output_dir'] = output_dir_override
    
    # 4. Create Config object (validates data)
    try:
        config = Config(**config_data)
        return config
    except Exception as e:
        logger.error(f"Configuration validation error: {e}")
        # In testing/dev, we might be lenient, but for now let's raise if it's fatal
        # or try to recover with what we have if we can't even load a model.
        # Let's try to return what we have if validation fails during tests?
        # Actually, let's stick to Pydantic validation and fix tests if they fail.
        raise

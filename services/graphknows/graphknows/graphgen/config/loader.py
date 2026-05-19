import yaml
import os
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

def load_yaml_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """
    Load configuration from a YAML file.
    Returns a dictionary suitable for passing to PipelineSettings and GraphSchema.
    """
    if not os.path.exists(config_path):
        logger.warning(f"Config file not found at {config_path}. Using Environment/Defaults.")
        return {}

    try:
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
            
        logger.info(f"Loaded configuration from {config_path}")
        return config_dict or {}
    except Exception as e:
        logger.error(f"Failed to load config file {config_path}: {e}")
        return {}

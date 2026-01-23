import os
import yaml
import logging
from .schema import Config
from .settings import PipelineSettings

logger = logging.getLogger(__name__)

def load_config(config_path: str = "config.yaml") -> Config:
    """
    Compatibility loader that reads YAML and applies environment overrides
    via the new Settings object where possible, or manual overrides.
    """
    config_data = {}
    
    # 1. Load YAML
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                yaml_data = yaml.safe_load(f)
                if yaml_data:
                    config_data.update(yaml_data)
        except Exception as e:
            logger.warning(f"Failed to load config.yaml: {e}")

    # 2. Apply Env Vars via Settings
    # We use PipelineSettings to get the authoritative env vars
    settings = PipelineSettings()
    
    if 'falkordb' not in config_data: config_data['falkordb'] = {}
    config_data['falkordb']['host'] = settings.falkordb_host
    config_data['falkordb']['port'] = settings.falkordb_port
    
    if settings.openai_api_key:
        if 'llm' not in config_data: config_data['llm'] = {}
        config_data['llm']['api_key'] = settings.openai_api_key
        
    # 3. Return validated Config
    return Config(**config_data)

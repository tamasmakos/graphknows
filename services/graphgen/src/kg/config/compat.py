import os
import yaml
import logging
from .schema import Config
from .settings import PipelineSettings

logger = logging.getLogger(__name__)

def load_config(config_path: str = None) -> Config:
    """
    Compatibility loader that reads YAML and applies environment overrides
    via the new Settings object where possible, or manual overrides.
    """
    config_data = {}
    
    # 1. Load YAML (Optional now)
    if config_path and os.path.exists(config_path):
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                yaml_data = yaml.safe_load(f)
                if yaml_data:
                    config_data.update(yaml_data)
        except Exception as e:
            logger.warning(f"Failed to load config.yaml: {e}")

    # 2. Apply Env Vars via Settings (Authoritative)
    settings = PipelineSettings()
    
    # Ensure nested dictionaries exist
    if 'processing' not in config_data: config_data['processing'] = {}
    if 'incremental' not in config_data: config_data['incremental'] = {}
    if 'falkordb' not in config_data: config_data['falkordb'] = {}
    if 'llm' not in config_data: config_data['llm'] = {}
    
    # Map settings to config structure
    config_data['processing']['input_dir'] = settings.input_dir
    config_data['processing']['output_dir'] = settings.output_dir
    config_data['processing']['file_pattern'] = settings.file_pattern
    config_data['processing']['chunk_size'] = settings.chunk_size
    config_data['processing']['chunk_overlap'] = settings.chunk_overlap
    
    config_data['incremental']['speech_limit'] = settings.speech_limit
    config_data['incremental']['max_documents'] = settings.max_documents
    
    config_data['falkordb']['host'] = settings.falkordb_host
    config_data['falkordb']['port'] = settings.falkordb_port
    
    if settings.openai_api_key:
        config_data['llm']['api_key'] = settings.openai_api_key
        
    # 3. Return validated Config
    return Config(**config_data)

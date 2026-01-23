"""
Configuration package for Knowledge Graph pipeline.
"""

from .schema import Config, PipelineConfig, ProcessingConfig, LLMConfig, EmbeddingConfig, CommunityConfig
from .compat import load_config

__all__ = [
    'Config',
    'PipelineConfig',
    'ProcessingConfig',
    'LLMConfig',
    'EmbeddingConfig',
    'CommunityConfig',
    'load_config'
]
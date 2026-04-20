"""
Centralized Embedding Model Management.

This module provides a singleton interface for the embedding model,
handling initialization, device selection, and unified configuration.
"""

import logging
from typing import List, Optional, Union
import numpy as np

try:
    from sentence_transformers import SentenceTransformer
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    SentenceTransformer = None

from kg.config.settings import PipelineSettings

logger = logging.getLogger(__name__)

class EmbeddingModel:
    _instance = None
    
    def __init__(self):
        if hasattr(self, '_initialized'):
            return
            
        self.settings = PipelineSettings().embedding
        self._model = None
        self._dimension = None
        self._initialized = True
        
        if TRANSFORMERS_AVAILABLE:
            self._load_model()
        else:
            logger.warning("sentence_transformers not installed. Embeddings are disabled.")

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _load_model(self):
        """Initialize the SentenceTransformer model."""
        model_name = self.settings.model_name
        device = self.settings.device
        
        if device == "auto" and torch:
             device = "cuda" if torch.cuda.is_available() else "cpu"

        logger.info(f"Loading embedding model '{model_name}' on {device}...")
        
        try:
            # Try loading from local cache first
            self._model = SentenceTransformer(
                model_name, 
                device=device, 
                local_files_only=True,
                cache_folder=self.settings.cache_folder,
            )
        except Exception:
            # Fallback to download
            logger.info(f"Model not found locally. Downloading '{model_name}'...")
            self._model = SentenceTransformer(
                model_name, 
                device=device,
                cache_folder=self.settings.cache_folder,
            )
            
        self._dimension = self._model.get_sentence_embedding_dimension()
        logger.info(f"Embedding model loaded. Dimension: {self._dimension}")

    @property
    def dimension(self) -> int:
        """Get the embedding dimension."""
        if self._dimension:
            return self._dimension
        return 384  # Default fallback

    @property
    def is_available(self) -> bool:
        return self._model is not None

    def encode(self, texts: Union[str, List[str]], batch_size: int = None) -> Union[List[float], np.ndarray]:
        """Generate embeddings for text(s)."""
        if not self._model:
            return np.array([]) if isinstance(texts, list) else np.array([])
            
        bs = batch_size or self.settings.batch_size
        return self._model.encode(texts, batch_size=bs, show_progress_bar=False)

def get_model() -> EmbeddingModel:
    """Get the singleton embedding model."""
    return EmbeddingModel.get_instance()

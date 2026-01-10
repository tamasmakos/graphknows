import networkx as nx
import os
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

def create_output_directory(path: str):
    """Create directory if it doesn't exist."""
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
        logger.info(f"Created directory: {path}")

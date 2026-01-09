"""
Pipeline stage registry and dependency management.
"""

import logging
import networkx as nx
from typing import List, Dict, Any, Callable, Optional, Set
from dataclasses import dataclass, field
from src.kg.config import Config

logger = logging.getLogger(__name__)

@dataclass
class PipelineStage:
    """Definition of a pipeline stage."""
    name: str
    display_name: str
    description: str
    run_func: Callable
    depends_on: List[str] = field(default_factory=list)

    
    async def run(self, graph: nx.DiGraph, config: Config, **kwargs) -> Dict[str, Any]:
        """Run the stage."""
        logger.info(f"Running stage: {self.display_name}")
        return await self.run_func(graph, config, **kwargs)

class StageRegistry:
    """Registry for pipeline stages."""
    
    def __init__(self):
        self.stages: Dict[str, PipelineStage] = {}
        
    def register(self, stage: PipelineStage):
        """Register a new stage."""
        self.stages[stage.name] = stage
        

        



# Global registry instance
registry = StageRegistry()



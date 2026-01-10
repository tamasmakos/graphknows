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
        
    def _ensure_registered(self):
        """Ensure stages are registered by importing the core pipeline module."""
        if not self.stages:
            try:
                import src.kg.pipeline.core # This triggers register_stages()
            except ImportError as e:
                logger.debug(f"Could not import core for registration: {e}")

    def register(self, stage: PipelineStage):
        """Register a new stage."""
        self.stages[stage.name] = stage
        
    def get_stage(self, name: str) -> Optional[PipelineStage]:
        """Get a stage by name."""
        self._ensure_registered()
        return self.stages.get(name)
        
    def get_execution_plan(self, config: Config) -> List[PipelineStage]:
        """
        Determine the execution plan based on enabled stages and their dependencies.
        
        Args:
            config: Pipeline configuration
            
        Returns:
            List of stages in topological order
        """
        self._ensure_registered()
        # Get stages explicitly enabled in config
        enabled_names = []
        if hasattr(config, 'pipeline') and hasattr(config.pipeline, 'stages'):
            for name, enabled in config.pipeline.stages.items():
                if enabled and name in self.stages:
                    enabled_names.append(name)
        
        # Build dependency graph for all registered stages
        dep_graph = nx.DiGraph()
        for name, stage in self.stages.items():
            dep_graph.add_node(name)
            for dep in stage.depends_on:
                dep_graph.add_edge(dep, name)
        
        # Find all required stages (enabled + their dependencies)
        required = set()
        stack = list(enabled_names)
        while stack:
            curr = stack.pop()
            if curr not in required:
                required.add(curr)
                if curr in self.stages:
                    stack.extend(self.stages[curr].depends_on)
        
        if not required:
            return []
            
        # Create subgraph and sort topologically
        sub_dep_graph = dep_graph.subgraph(required)
        try:
            sorted_names = list(nx.topological_sort(sub_dep_graph))
            return [self.stages[name] for name in sorted_names if name in self.stages]
        except nx.NetworkXUnfeasible:
            logger.error("Circular dependency detected in pipeline stages")
            raise ValueError("Circular dependency detected in pipeline stages")
        

        



# Global registry instance
registry = StageRegistry()



"""
Type definitions for Knowledge Graph pipeline.
"""

import os
import re
from dataclasses import dataclass, field
from typing import List, Dict, Any, Set, Optional
import networkx as nx
from pydantic import BaseModel


@dataclass
class ChunkExtractionTask:
    """Task for extracting entities and relations from a chunk."""
    chunk_id: str
    chunk_text: str
    entities: List[str]
    abstract_concepts: List[str]
    keywords: List[str] = field(default_factory=list)


@dataclass
class AgentDependencies:
    """Dependencies shared across pipeline agents."""
    graph: nx.DiGraph
    extraction_tasks: List[ChunkExtractionTask] = field(default_factory=list)
    total_segments: int = 0


class FalkorDBConfig(BaseModel):
    """Configuration for FalkorDB graph database."""
    upload_enabled: bool = True
    host: str = "localhost"
    port: int = 6379
    username: Optional[str] = None
    password: Optional[str] = None
    database: str = "kg"
    clean_database: bool = True

# IterativeConfig removed

class Config(BaseModel):
    """Main configuration object."""
    falkordb: FalkorDBConfig




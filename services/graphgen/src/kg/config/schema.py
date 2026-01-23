"""
Configuration schema using Pydantic models.
"""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
import os

class IncrementalConfig(BaseModel):
    """Configuration for incremental processing."""
    speech_limit: int = 1
    max_documents: int = 10
    state_file: str = "processing_state.json"
    enable_pruning: bool = True
    pruning_threshold: float = 0.01
    auto_recalculate_centrality: bool = True

class PipelineConfig(BaseModel):
    """Configuration for pipeline stages."""
    stages: Dict[str, bool] = Field(
        default_factory=lambda: {
            "lexical_graph": True,
            "extraction": True,
            "embeddings": True,
            "semantic_resolution": True,
            "similarity_edges": True,
            "community_detection": True,
            "summarization": True,
            "schema_export": True,
            "neo4j_upload": False
        }
    )

class ProcessingConfig(BaseModel):
    """Configuration for processing limits and file handling."""
    input_dir: str = "."  # Required - must be provided from config.yaml
    output_dir: str = "output"
    speech_limit: int = Field(0, description="Maximum number of speeches to process (0 = no limit)")
    max_concurrent_extractions: int = 8
    file_pattern: str = "*.txt"
    parser_type: str = Field("generic", description="Parser to use: 'generic', 'parlamint', 'cre', or 'auto'")
    parser_kwargs: Dict[str, Any] = Field(default_factory=dict, description="Additional kwargs for parser initialization")
    chunk_size: int = 10  # Match test defaults
    chunk_overlap: int = 2  # Match test defaults
    use_overlapping_chunks: bool = True
    enable_metadata_extractors: bool = Field(True, description="Enable metadata extractors (Title, Summary, Keywords, etc.)")
    max_documents: int = Field(0, description="Maximum number of documents to process in one run (0 = no limit)")
    state_file: str = "processing_state.json"

class LLMConfig(BaseModel):
    """Configuration for LLM."""
    model: str = "llama-3.3-70b-versatile"
    temperature: float = 0.0
    api_key: Optional[str] = Field(None, description="API key (usually from env var)")
    provider: str = "openai" # Groq is compatible with openai provider

class EmbeddingConfig(BaseModel):
    """Configuration for embeddings."""
    model_name: str = Field("all-MiniLM-L6-v2", alias="model") # Use alias for backward compat if needed, or just rename
    batch_size: int = 32

class GraphConfig(BaseModel):
    """Configuration for graph enrichment."""
    extractor_type: str = Field("langchain", description="Graph extractor to use: 'langchain'")
    embedding_similarity_threshold: float = 0.85
    semantic_resolution_threshold: float = 0.95
    add_similarity_edges: bool = True
    enable_semantic_resolution: bool = True
    enable_kge: bool = False  # Knowledge Graph Embeddings (structural)

class CommunityConfig(BaseModel):
    """Configuration for community detection."""
    resolution_parameter: float = 1.0
    random_seed: int = 42
    iterations: int = 10
    min_community_size: int = 2
    min_subcommunity_size: int = 2
    sub_max_depth: int = 1
    sub_resolution_min: float = 0.7
    sub_resolution_max: float = 1.3
    sub_resolution_steps: int = 7
    sub_consistency_threshold: float = 0.75

class Neo4jConfig(BaseModel):
    """Configuration for Neo4j (Legacy/Alternative)."""
    uri: str = "bolt://localhost:7687"
    username: str = "neo4j"
    password: str = "password"

class FalkorDBConfig(BaseModel):
    """Configuration for FalkorDB."""
    host: str = "localhost"
    port: int = 6379
    username: Optional[str] = Field(None, description="FalkorDB username")
    password: Optional[str] = Field(None, description="FalkorDB password")
    database: str = "kg"
    clean_database: bool = True
    upload_enabled: bool = False

class PostgresConfig(BaseModel):
    """Configuration for PostgreSQL."""
    enabled: bool = False
    host: str = "localhost"
    port: int = 5432
    database: str = "graphknows"
    user: str = "postgres"
    password: str = "password"
    table_name: str = "content_chunks"

class Config(BaseModel):
    """Main configuration class."""
    pipeline: PipelineConfig = Field(default_factory=PipelineConfig)
    processing: ProcessingConfig = Field(default_factory=ProcessingConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    embeddings: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    graph: GraphConfig = Field(default_factory=GraphConfig)
    community: CommunityConfig = Field(default_factory=CommunityConfig)
    falkordb: FalkorDBConfig = Field(default_factory=FalkorDBConfig)
    postgres: PostgresConfig = Field(default_factory=PostgresConfig)
    neo4j: Neo4jConfig = Field(default_factory=Neo4jConfig)
    incremental: IncrementalConfig = Field(default_factory=IncrementalConfig)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for backward compatibility."""
        # This mimics the structure expected by existing code
        return {
            'input_dir': self.processing.input_dir,
            'output_dir': self.processing.output_dir,
            'speech_limit': self.processing.speech_limit,
            'use_speech_limit': self.processing.speech_limit > 0,
            'max_concurrent_extractions': self.processing.max_concurrent_extractions,
            'file_pattern': self.processing.file_pattern,
            'parser_type': self.processing.parser_type,
            'parser_kwargs': self.processing.parser_kwargs,
            'chunk_size': self.processing.chunk_size,
            'chunk_overlap': self.processing.chunk_overlap,
            'use_overlapping_chunks': self.processing.use_overlapping_chunks,
            'enable_metadata_extractors': self.processing.enable_metadata_extractors,
            'llm_model': self.llm.model,
            'llm_temperature': self.llm.temperature,
            'extractor_type': self.graph.extractor_type,
            'embedding_model': self.embeddings.model_name,
            'embedding_batch_size': self.embeddings.batch_size,
            'embedding_similarity_threshold': self.graph.embedding_similarity_threshold,
            'add_similarity_edges': self.graph.add_similarity_edges,
            'enable_semantic_resolution': self.graph.enable_semantic_resolution,
            'semantic_resolution_threshold': self.graph.semantic_resolution_threshold,

            'falkordb': self.falkordb.dict(),
            'postgres': self.postgres.dict(),
            'falkordb_upload_enabled': self.falkordb.upload_enabled,
            'falkordb_clean_database': self.falkordb.clean_database,
            'community_detection': self.community.dict()
        }

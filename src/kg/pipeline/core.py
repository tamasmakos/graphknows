"""
Knowledge Graph Pipeline.

Main pipeline function to build the semantic knowledge graph using a stage-based approach
with strict dependency management and configuration.
"""

import asyncio
import logging
import os
import json
import time
import networkx as nx
from typing import Dict, Any, List, Optional

from src.kg.config import Config
from src.kg.types import AgentDependencies
from src.kg.llm import get_langchain_llm
from src.kg.pipeline.stages import registry, PipelineStage
from src.kg.graph.schema import save_graph_schema

# Import stage implementations
from src.kg.graph.extraction import (
    build_lexical_graph,
    extract_all_entities_relations
)
from src.kg.embeddings.rag import generate_rag_embeddings
# from src.kg.embeddings.kge import _train_and_cache_global_kge
from src.kg.graph.resolution import merge_similar_nodes
from src.kg.graph.similarity import compute_embedding_similarity_edges
from src.kg.community.detection import CommunityDetector
from src.kg.community.subcommunities import add_enhanced_community_attributes_to_graph
from src.kg.graph.centrality import calculate_entity_relation_centrality_measures
from src.kg.community.metrics import evaluate_community_quality
from src.kg.summarization.core import generate_community_summaries
from src.kg.summarization.reporting import (
    generate_community_summary_comparison
)




logger = logging.getLogger(__name__)

# ==============================================================================
# Stage Implementations
# ==============================================================================

async def run_lexical_graph(graph: nx.DiGraph, config: Config, **kwargs) -> Dict[str, Any]:
    """Stage 1: Build Lexical Graph (documents, segments, chunks)."""
    deps = kwargs.get('deps')
    input_dir = config.processing.input_dir
    
    logger.info("Building lexical graph structure...")
    lexical_result = await build_lexical_graph(deps, input_dir, config.to_dict())
    return {"lexical_result": lexical_result}

async def run_extraction(graph: nx.DiGraph, config: Config, **kwargs) -> Dict[str, Any]:
    """Stage 2: Entity/Relation Extraction."""
    deps = kwargs.get('deps')
    
    # Extract entities/relations
    logger.info(f"Processing {len(deps.extraction_tasks)} chunks for extraction...")
    extraction_result = await extract_all_entities_relations(deps, config.to_dict())
    return {"extraction_result": extraction_result}

async def run_embeddings(graph: nx.DiGraph, config: Config, **kwargs) -> Dict[str, Any]:
    """Stage 3: Generate Embeddings."""
    output_dir = kwargs.get('output_dir')
    
    # Train Global KGE (optional - structural graph embeddings)
    kge_trained = False
    if config.graph.enable_kge:
        # logger.info("Training Knowledge Graph Embeddings (KGE) - structural embeddings")
        # kge_output_dir = os.path.join(output_dir, "embeddings", "kge")
        # os.makedirs(kge_output_dir, exist_ok=True)
        # _train_and_cache_global_kge(graph, kge_output_dir)
        # kge_trained = True
        logger.warning("KGE training skipped - module missing")
        kge_trained = False
    else:
        logger.info("Skipping KGE training (disabled in config)")
    
    # RAG Embeddings (always needed for semantic similarity)
    logger.info("Generating RAG embeddings (text-based) for semantic similarity")
    embedding_model = config.embeddings.model
    batch_size = config.embeddings.batch_size
    
    rag_embeddings = generate_rag_embeddings(graph, embedding_model, batch_size)
    
    return {
        "rag_embeddings_count": len(rag_embeddings),
        "kge_trained": kge_trained
    }

async def run_semantic_resolution(graph: nx.DiGraph, config: Config, **kwargs) -> Dict[str, Any]:
    """Stage 4: Semantic Entity Resolution."""
    threshold = config.graph.semantic_resolution_threshold
    
    resolution_stats = merge_similar_nodes(
        graph,
        similarity_threshold=threshold,
        node_types=['ENTITY_CONCEPT']
    )
    return {"resolution_stats": resolution_stats}

async def run_similarity_edges(graph: nx.DiGraph, config: Config, **kwargs) -> Dict[str, Any]:
    """Stage 5: Compute Embedding Similarity Edges."""
    threshold = config.graph.embedding_similarity_threshold
    
    similarity_stats = compute_embedding_similarity_edges(
        graph,
        similarity_threshold=threshold,
        node_types=['ENTITY_CONCEPT'],
        add_new_edges=True,
        update_existing_weights=True
    )
    return {"similarity_stats": similarity_stats}

async def run_community_detection(graph: nx.DiGraph, config: Config, **kwargs) -> Dict[str, Any]:
    """Stage 6: Community Detection."""
    detector = CommunityDetector()
    
    # Detect main communities
    results = detector.detect_communities(graph)
    communities = results['assignments']
    
    # Detect subcommunities
    subcommunities = detector.detect_subcommunities_leiden(
        graph.to_undirected(),
        communities,
        min_sub_size=config.community.min_subcommunity_size
    )
    
    # Apply attributes
    nx.set_node_attributes(graph, communities, 'community')
    add_enhanced_community_attributes_to_graph(graph, communities, subcommunities)
    
    # Calculate centrality
    centrality_results = calculate_entity_relation_centrality_measures(graph)
    
    # Evaluate quality
    community_quality = evaluate_community_quality(graph, communities)
    
    return {
        "communities": communities,
        "subcommunities": subcommunities,
        "centrality_results": centrality_results,
        "community_quality": community_quality,
        "metrics": results
    }

async def run_summarization(graph: nx.DiGraph, config: Config, **kwargs) -> Dict[str, Any]:
    """Stage 7: Summarization."""
    llm = kwargs.get('llm')
    output_dir = kwargs.get('output_dir')
    
    # Community summaries
    summarization_stats = await generate_community_summaries(graph, llm)
    
    # Comparison
    analytics_dir = os.path.join(output_dir, "analytics")
    os.makedirs(analytics_dir, exist_ok=True)
    generate_community_summary_comparison(graph, analytics_dir)
    
    # Generate embeddings for new topic nodes
    generate_rag_embeddings(
        graph,
        embedding_model=config.embeddings.model,
        batch_size=config.embeddings.batch_size,
        node_types=['TOPIC', 'SUBTOPIC']
    )
    
    return {"summarization_stats": summarization_stats}

async def run_schema_export(graph: nx.DiGraph, config: Config, **kwargs) -> Dict[str, Any]:
    """Stage 9: Schema Export."""
    output_dir = kwargs.get('output_dir')
    save_graph_schema(graph, output_dir)
    return {"schema_exported": True}



async def run_falkordb_upload(graph: nx.DiGraph, config: Config, **kwargs) -> Dict[str, Any]:
    """Stage 11: FalkorDB Upload."""
    try:
        from src.kg.falkordb import KnowledgeGraphUploader
        
        uploader = KnowledgeGraphUploader(
            host=config.falkordb.host,
            port=config.falkordb.port,
            username=config.falkordb.username,
            password=config.falkordb.password,
            database=config.falkordb.database,
            postgres_config=config.to_dict().get('postgres')
        )
        stats = uploader.upload(
            graph,
            clean_database=config.falkordb.clean_database,
            create_indexes_flag=True # Always create indexes for now, or add config option if needed
        )
        return stats
    except Exception as e:
        logger.error(f"FalkorDB upload failed: {e}")
        return {"error": str(e)}

# ==============================================================================
# Register Stages
# ==============================================================================

def register_stages():
    """Register all pipeline stages."""
    registry.register(PipelineStage(
        name="lexical_graph",
        display_name="Lexical Graph Construction",
        description="Builds document structure (Segments, Chunks)",
        run_func=run_lexical_graph,

    ))
    
    registry.register(PipelineStage(
        name="extraction",
        display_name="Entity & Relation Extraction",
        description="Extracts entities and relations from text",
        run_func=run_extraction,
        depends_on=["lexical_graph"],

    ))
    
    registry.register(PipelineStage(
        name="embeddings",
        display_name="Embedding Generation",
        description="Generates vector embeddings for nodes",
        run_func=run_embeddings,
        depends_on=["extraction"],

    ))
    
    registry.register(PipelineStage(
        name="semantic_resolution",
        display_name="Semantic Entity Resolution",
        description="Merges duplicate entities based on semantic similarity",
        run_func=run_semantic_resolution,
        depends_on=["embeddings"],

    ))
    
    registry.register(PipelineStage(
        name="similarity_edges",
        display_name="Similarity Edge Computation",
        description="Adds SIMILAR_TO edges between related concepts",
        run_func=run_similarity_edges,
        depends_on=["embeddings"],

    ))
    
    registry.register(PipelineStage(
        name="community_detection",
        display_name="Community Detection",
        description="Detects communities and subcommunities",
        run_func=run_community_detection,
        depends_on=["similarity_edges"], # Enforce similarity edges for quality

    ))
    
    registry.register(PipelineStage(
        name="summarization",
        display_name="Summarization",
        description="Generates summaries for communities",
        run_func=run_summarization,
        depends_on=["community_detection"]
    ))
    
    registry.register(PipelineStage(
        name="schema_export",
        display_name="Schema Export",
        description="Exports graph schema to JSON",
        run_func=run_schema_export
    ))
    

    
    registry.register(PipelineStage(
        name="neo4j_upload",
        display_name="Neo4j/FalkorDB Upload",
        description="Uploads graph to Neo4j/FalkorDB database",
        run_func=run_falkordb_upload
    ))

# Register stages on module load
register_stages()

# ==============================================================================
# Main Pipeline
# ==============================================================================

# build_semantic_kg_with_communities (the old 'batch' pipeline) has been replaced by 
# the new iterative pipeline in iterative.py.



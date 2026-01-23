"""
Knowledge Graph Pipeline (Core).

Defines the KnowledgePipeline class which orchestrates the graph generation process.
Follows the Inversion of Control pattern where dependencies are injected.
"""

import os
import logging
import networkx as nx
from typing import Dict, Any, Optional

from ..falkordb.uploader import KnowledgeGraphUploader
from ..config.settings import PipelineSettings
from ..types import AgentDependencies
from ..graph.extraction import build_lexical_graph, extract_all_entities_relations
from ..graph.pruning import prune_graph
from ..graph.utils import create_output_directory
from ..graph.schema import save_graph_schema
from ..utils.health import check_falkordb

logger = logging.getLogger(__name__)

class KnowledgePipeline:
    """
    The main pipeline orchestrator.
    
    It accepts all necessary dependencies (infrastructure, configuration) via the constructor.
    It does NOT instantiate heavy objects itself.
    """

    def __init__(
        self, 
        settings: PipelineSettings,
        uploader: KnowledgeGraphUploader,
        extractor: Any = None
    ):
        self.settings = settings
        self.uploader = uploader
        self.extractor = extractor
        
    async def run(self):
        """
        Execute the full knowledge graph generation pipeline:
        1. Build Lexical Graph from Input Dir
        2. Extract Entities/Relations
        3. Semantic Enrichment (Embeddings, Similarity, Resolution)
        4. Community Detection & Summarization
        5. Pruning
        6. Upload to FalkorDB
        7. Save Artifacts to Disk
        """
        logger.info("Starting KnowledgePipeline run...")
        
        # Preflight Checks
        logger.info("Performing preflight health checks...")
        if not check_falkordb(self.settings.falkordb_host, self.settings.falkordb_port):
            error_msg = f"Preflight check failed: FalkorDB is not reachable at {self.settings.falkordb_host}:{self.settings.falkordb_port}."
            logger.critical(f"{error_msg} Aborting pipeline.")
            raise ConnectionError(error_msg)

        # 0. Setup Dependencies
        graph = nx.DiGraph()
        deps = AgentDependencies(graph=graph)
        config_dict = self.settings.model_dump() if hasattr(self.settings, 'model_dump') else self.settings.dict()
        
        # 1. Build Lexical Graph
        input_dir = self.settings.input_dir
        logger.info(f"Step 1: Reading from {input_dir}")
        
        results = await build_lexical_graph(deps, input_dir, config_dict)
        logger.info(f"Lexical Graph Built: {results.get('documents_processed')} docs, {results.get('total_segments')} segments")

        # 2. Extraction
        if self.extractor:
             logger.info("Step 2: Starting Extraction...")
             extract_results = await extract_all_entities_relations(deps, config_dict, extractor=self.extractor)
             logger.info(f"Extraction Complete: {extract_results.get('successful')} successful chunks")
        else:
             logger.warning("No extractor provided, skipping extraction.")

        # 3. Semantic Enrichment
        try:
            from ..embeddings.rag import generate_rag_embeddings
            logger.info("Step 3: Generating RAG Embeddings...")
            generate_rag_embeddings(graph)
            
            from ..graph.similarity import compute_embedding_similarity_edges
            logger.info("Step 4: Computing Similarity Edges...")
            compute_embedding_similarity_edges(graph)
            
            from ..graph.resolution import merge_similar_nodes
            logger.info("Step 5: Semantic Resolution...")
            merge_similar_nodes(graph)
        except Exception as e:
            logger.error(f"Semantic enrichment failed: {e}")

        # 4. Community Detection & Summarization
        try:
            from ..community.detection import CommunityDetector
            from ..community.subcommunities import add_enhanced_community_attributes_to_graph
            from ..summarization.core import generate_community_summaries
            from ..llm import get_langchain_llm
            
            logger.info("Step 6: Community Detection...")
            detector = CommunityDetector()
            comm_results = detector.detect_communities(graph)
            communities = comm_results['assignments']
            
            subcommunities = detector.detect_subcommunities_leiden(graph, communities)
            add_enhanced_community_attributes_to_graph(graph, communities, subcommunities)
            
            logger.info("Step 7: Summarization...")
            llm = get_langchain_llm(config_dict, purpose='summarization')
            await generate_community_summaries(graph, llm)
        except Exception as e:
            logger.error(f"Community detection or summarization failed: {e}")

        # 5. Pruning
        logger.info("Step 8: Pruning graph...")
        prune_stats = prune_graph(graph, {'pruning_threshold': 0.01})
        logger.info(f"Pruning Stats: {prune_stats}")
        
        # 6. Upload
        if self.uploader:
            logger.info("Step 9: Uploading to FalkorDB...")
            try:
                if self.uploader.connect():
                    stats = self.uploader.upload(graph, clean_database=True)
                    logger.info(f"Upload Stats: {stats}")
                    self.uploader.close()
                else:
                    logger.warning("Uploader could not connect.")
            except Exception as e:
                 logger.error(f"Upload failed: {e}")

        # 7. Save Artifacts
        output_dir = self.settings.output_dir
        logger.info(f"Step 10: Saving artifacts to {output_dir}")
        create_output_directory(output_dir)
        
        try:
            save_graph_schema(graph, output_dir)
            
            # Save GraphML for visualization
            graph_path = os.path.join(output_dir, "knowledge_graph.graphml")
            clean_graph = graph.copy()
            # GraphML doesn't support list/dict properties, so we stringify them
            import json
            for _, d in clean_graph.nodes(data=True):
                for k, v in list(d.items()):
                    if isinstance(v, (dict, list)):
                        d[k] = json.dumps(v, ensure_ascii=False)
            for _, _, d in clean_graph.edges(data=True):
                for k, v in list(d.items()):
                    if isinstance(v, (dict, list)):
                        d[k] = json.dumps(v, ensure_ascii=False)
            
            nx.write_graphml(clean_graph, graph_path)
            logger.info(f"GraphML saved to {graph_path}")
        except Exception as e:
            logger.error(f"Failed to save artifacts: {e}")
        
        logger.info("Pipeline Run Finished.")
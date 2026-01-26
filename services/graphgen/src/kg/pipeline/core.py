"""
Knowledge Graph Pipeline (Core).

Defines the KnowledgePipeline class which orchestrates the graph generation process.
Follows the Inversion of Control pattern where dependencies are injected.
"""

import os
import asyncio
import uuid
import logging
import networkx as nx
from typing import Dict, Any, List

from kg.types import PipelineContext
from kg.config.settings import PipelineSettings
from kg.falkordb.uploader import KnowledgeGraphUploader
from kg.graph.extraction import build_lexical_graph, extract_all_entities_relations
from kg.graph.extractors import BaseExtractor
from kg.graph.pruning import prune_graph


from kg.graph.utils import create_output_directory
from kg.graph.schema import save_graph_schema
from kg.utils.health import check_falkordb

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
        self.run_id = str(uuid.uuid4())[:8]
        
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
        logger.info(f"🚀 Starting KnowledgePipeline run [{self.run_id}]...")
        
        # Preflight Checks
        self._run_preflight_checks()

        # 0. Initialize Pipeline Context (The "Bus")
        graph = nx.DiGraph()
        # PipelineContext (aliased as AgentDependencies) holds the state
        ctx = PipelineContext(graph=graph)
        
        # Convert settings to dict for legacy functions
        # TODO: Refactor downstream functions to accept PipelineSettings object directly
        config_dict = self.settings.model_dump() if hasattr(self.settings, 'model_dump') else self.settings.dict()
        
        try:
            # 1. Build Lexical Graph
            await self._step_lexical_graph(ctx, config_dict)

            # 2. Extraction
            await self._step_extraction(ctx, config_dict)

            # 3. Semantic Enrichment
            await self._step_enrichment(ctx)

            # 4. Community Detection & Summarization
            await self._step_communities(ctx, config_dict)

            # 5. Pruning
            await self._step_pruning(ctx)
            
            # 6. Upload
            await self._step_upload(ctx)

            # 7. Save Artifacts
            self._step_save_artifacts(ctx)
        
        except Exception as e:
            logger.critical(f"🔥 Pipeline [{self.run_id}] failed: {e}", exc_info=True)
            raise
        
        logger.info(f"✅ Pipeline Run [{self.run_id}] Finished Successfully.")

    def _run_preflight_checks(self):
        """Check external dependencies."""
        logger.info("Performing preflight health checks...")
        if not check_falkordb(self.settings.infra.falkordb_host, self.settings.infra.falkordb_port):
            error_msg = f"Preflight check failed: FalkorDB is not reachable at {self.settings.infra.falkordb_host}:{self.settings.infra.falkordb_port}."
            logger.critical(f"{error_msg} Aborting pipeline.")
            raise ConnectionError(error_msg)

    async def _step_lexical_graph(self, ctx: PipelineContext, config: Dict[str, Any]):
        input_dir = self.settings.infra.input_dir
        logger.info(f"Step 1: Building Lexical Graph from {input_dir}")
        
        results = await build_lexical_graph(ctx, input_dir, config)
        
        ctx.stats['lexical'] = results
        logger.info(f"Lexical Graph Built: {results.get('documents_processed')} docs, {results.get('total_segments')} segments")

    async def _step_extraction(self, ctx: PipelineContext, config: Dict[str, Any]):
        if not self.extractor:
             logger.warning("Step 2: Skipped (No extractor provided).")
             return

        logger.info("Step 2: Extracting Entities & Relations...")
        extract_results = await extract_all_entities_relations(ctx, config, extractor=self.extractor)
        
        ctx.stats['extraction'] = extract_results
        logger.info(f"Extraction Complete: {extract_results.get('successful')} successful chunks")

    async def _step_enrichment(self, ctx: PipelineContext):
        try:
            from ..embeddings.rag import generate_rag_embeddings
            from ..graph.similarity import compute_embedding_similarity_edges
            from ..graph.resolution import merge_similar_nodes
            
            logger.info("Step 3: Semantic Enrichment")
            
            logger.info("  3.1: Generating RAG Embeddings...")
            generate_rag_embeddings(ctx.graph)
            
            logger.info("  3.2: Computing Similarity Edges...")
            compute_embedding_similarity_edges(ctx.graph)
            
            logger.info("  3.3: Semantic Resolution...")
            merge_similar_nodes(ctx.graph)
            
        except Exception as e:
            logger.error(f"Semantic enrichment failed: {e}")
            ctx.add_error("enrichment", str(e))

    async def _step_communities(self, ctx: PipelineContext, config: Dict[str, Any]):
        try:
            from ..community.detection import CommunityDetector
            from ..community.subcommunities import add_enhanced_community_attributes_to_graph
            from ..summarization.core import generate_community_summaries
            from ..llm import get_langchain_llm
            
            logger.info("Step 4: Community Detection & Summarization")
            
            logger.info("  4.1: Detecting Communities...")
            detector = CommunityDetector()
            comm_results = detector.detect_communities(ctx.graph)
            communities = comm_results['assignments']
            
            subcommunities = detector.detect_subcommunities_leiden(ctx.graph, communities)
            add_enhanced_community_attributes_to_graph(ctx.graph, communities, subcommunities)
            
            logger.info("  4.2: Generating Summaries...")
            llm = get_langchain_llm(config, purpose='summarization')
            summary_stats = await generate_community_summaries(ctx.graph, llm)
            ctx.stats['summarization'] = summary_stats
            
        except Exception as e:
            logger.error(f"Community detection or summarization failed: {e}")
            ctx.add_error("communities", str(e))

    async def _step_pruning(self, ctx: PipelineContext):
        logger.info("Step 5: Pruning Graph...")
        prune_stats = prune_graph(ctx.graph, {'pruning_threshold': 0.01})
        ctx.stats['pruning'] = prune_stats
        logger.info(f"Pruning Stats: {prune_stats}")

    async def _step_upload(self, ctx: PipelineContext):
        if not self.uploader:
            return
            
        logger.info("Step 6: Uploading to FalkorDB...")
        try:
            if self.uploader.connect():
                stats = self.uploader.upload(ctx.graph, clean_database=True)
                ctx.stats['upload'] = stats
                logger.info(f"Upload Stats: {stats}")
                self.uploader.close()
            else:
                logger.warning("Uploader could not connect.")
                ctx.add_error("upload", "Could not connect")
        except Exception as e:
             logger.error(f"Upload failed: {e}")
             ctx.add_error("upload", str(e))

    def _step_save_artifacts(self, ctx: PipelineContext):
        output_dir = self.settings.infra.output_dir
        logger.info(f"Step 7: Saving artifacts to {output_dir}")
        create_output_directory(output_dir)
        
        try:
            save_graph_schema(ctx.graph, output_dir)
            
            # Save GraphML
            graph_path = os.path.join(output_dir, "knowledge_graph.graphml")
            clean_graph = ctx.graph.copy()
            
            # Serialize complex types for GraphML
            import json
            from datetime import date, datetime
            
            for _, d in clean_graph.nodes(data=True):
                for k, v in list(d.items()):
                    if v is None:
                        del d[k]
                        continue
                    if isinstance(v, (dict, list)):
                        d[k] = json.dumps(v, ensure_ascii=False)
                    elif isinstance(v, (date, datetime)):
                        d[k] = v.isoformat()
                        
            for _, _, d in clean_graph.edges(data=True):
                for k, v in list(d.items()):
                    if v is None:
                        del d[k]
                        continue
                    if isinstance(v, (dict, list)):
                        d[k] = json.dumps(v, ensure_ascii=False)
                    elif isinstance(v, (date, datetime)):
                        d[k] = v.isoformat()
            
            nx.write_graphml(clean_graph, graph_path)
            logger.info(f"GraphML saved to {graph_path}")
        except Exception as e:
            logger.error(f"Failed to save artifacts: {e}")
            ctx.add_error("artifacts", str(e))
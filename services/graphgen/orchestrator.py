"""
Knowledge Graph Pipeline (Core).

Defines the KnowledgePipeline class which orchestrates the graph generation process.
Follows the Inversion of Control pattern where dependencies are injected.
"""

import os
import asyncio
import uuid
import logging
from datetime import date, datetime
import json
import networkx as nx
from copy import deepcopy
from typing import Dict, Any, List

from graphgen.config.settings import PipelineSettings
from graphgen.data_types import PipelineContext
from graphgen.utils.graphdb.neo4j_adapter import Neo4jGraphUploader
from graphgen.pipeline.lexical_graph_building.builder import build_lexical_graph
from graphgen.pipeline.entity_relation.extraction import extract_all_entities_relations
from graphgen.pipeline.entity_relation.extractors import BaseExtractor
from graphgen.pipeline.graph_cleaning.pruning import prune_graph
from graphgen.utils.utils import create_output_directory
from graphgen.utils.schema_utils import save_graph_schema
from graphgen.utils.provenance import (
    write_analysis_run_manifest,
    write_pipeline_config_snapshot,
)

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
        uploader: Neo4jGraphUploader,
        extractor: Any = None
    ):
        self.settings = settings
        self.uploader = uploader
        self.extractor = extractor
        self.run_id = str(uuid.uuid4())[:8]
        
    async def run(self) -> None:
        """
        Execute the full knowledge graph generation pipeline:
        1. Build Lexical Graph from Input Dir
        2. Extract Entities/Relations
        3. Semantic Enrichment (Embeddings, Similarity, Resolution)
        4. Community Detection & Summarization
        5. Topic Analysis (Statistical tests)
        6. Pruning
        7. Upload to Graph Database
        8. Save Artifacts to Disk
        """
        logger.info(f"Starting KnowledgePipeline run [{self.run_id}]...")
        self.run_started_at = datetime.now()
        output_dir = self.settings.infra.output_dir
        create_output_directory(output_dir)

        thesis_output_dir = os.path.join(output_dir, self.settings.analytics.outputs_subdir)
        create_output_directory(thesis_output_dir)
        if self.settings.analytics.save_provenance:
            write_pipeline_config_snapshot(thesis_output_dir, self.settings)
            write_analysis_run_manifest(
                thesis_output_dir,
                self.settings,
                stage="started",
                run_id=self.run_id,
                started_at=self.run_started_at.isoformat(),
            )
        
        # Preflight Checks
        self._run_preflight_checks()

        # 0. Initialize Pipeline Context (The "Bus")
        graph = nx.DiGraph()
        # PipelineContext (aliased as AgentDependencies) holds the state
        ctx = PipelineContext(graph=graph)
        
        # Convert settings to dict for legacy functions
        # TODO: Refactor downstream functions to accept PipelineSettings object directly
        config_dict = self.settings.model_dump() if hasattr(self.settings, 'model_dump') else self.settings.dict()
        ctx.stats['pipeline_config'] = config_dict
        
        try:
            # 1. Build Lexical Graph
            from graphgen.config.schema import GraphSchema
            # Parse schema from settings
            schema = None
            if self.settings.schema_config:
                try:
                    schema = GraphSchema(**self.settings.schema_config)
                except Exception as ex:
                    logger.warning(f"Failed to parse schema: {ex}")

            await self._step_lexical_graph(ctx, config_dict, schema=schema)

            # 2. Extraction
            await self._step_extraction(ctx, config_dict)

            # 3. Semantic Enrichment
            await self._step_enrichment(ctx)

            # 4. Community Detection & Summarization
            await self._step_communities(ctx, config_dict)

            # 4.5. Topic Separation Analysis
            await self._step_topic_analysis(ctx)

            # 5. Pruning
            await self._step_pruning(ctx)
            
            # 6. Upload
            await self._step_upload(ctx)

            # 7. Save Artifacts
            self._step_save_artifacts(ctx)
        
        except Exception as e:
            logger.critical(f"Pipeline [{self.run_id}] failed: {e}", exc_info=True)
            if self.settings.analytics.save_provenance:
                write_analysis_run_manifest(
                    thesis_output_dir,
                    self.settings,
                    stage="failed",
                    run_id=self.run_id,
                    started_at=self.run_started_at.isoformat(),
                    completed_at=datetime.now().isoformat(),
                    extra={"error": str(e)},
                )
            raise
        
        if self.settings.analytics.save_provenance:
            write_analysis_run_manifest(
                thesis_output_dir,
                self.settings,
                stage="completed",
                run_id=self.run_id,
                started_at=self.run_started_at.isoformat(),
                completed_at=datetime.now().isoformat(),
                extra={
                    "stats": ctx.stats,
                    "graph": {
                        "nodes": ctx.graph.number_of_nodes(),
                        "edges": ctx.graph.number_of_edges(),
                    },
                },
            )

        logger.info(f"Pipeline run [{self.run_id}] finished successfully.")

    def _run_preflight_checks(self) -> None:
        """Check external dependencies."""
        logger.debug("Performing preflight health checks...")
        
        # Basic check via uploader connectivity
        if self.uploader and not self.uploader.connect():
            logger.warning("Preflight check failed: Neo4j is not reachable. Continuing without uploader.")
            self.uploader = None
            return
        if self.uploader:
            self.uploader.close() # Close after check

    async def _step_lexical_graph(
        self,
        ctx: PipelineContext,
        config: Dict[str, Any],
        schema: Any = None,
    ) -> None:
        input_dir = self.settings.infra.input_dir
        logger.info(f"Step 1: Building Lexical Graph from {input_dir}")
        
        results = await build_lexical_graph(ctx, input_dir, config, schema=schema)
        
        ctx.stats['lexical'] = results
        logger.info(f"Lexical Graph Built: {results.get('documents_processed')} docs, {results.get('total_segments')} segments")

    async def _step_extraction(self, ctx: PipelineContext, config: Dict[str, Any]) -> None:
        if not self.extractor:
            logger.warning("Step 2: Skipped (No extractor provided).")
            return

        logger.debug("Step 2: Extracting Entities & Relations...")
        extract_results = await extract_all_entities_relations(ctx, config, extractor=self.extractor)
        
        ctx.stats['extraction'] = extract_results
        logger.info(f"Extraction Complete: {extract_results.get('successful')} successful chunks")

    async def _step_enrichment(self, ctx: PipelineContext) -> None:
        try:
            from graphgen.utils.vector_embedder.rag import generate_rag_embeddings
            from graphgen.pipeline.graph_cleaning.resolution import resolve_entities_semantically
            from graphgen.evaluation import summarize_entity_resolution_effects
            
            logger.debug("Step 3: Semantic Enrichment")
            
            logger.info("  3.1: Generating entity embeddings for semantic resolution...")
            generate_rag_embeddings(ctx.graph, node_types=['ENTITY_CONCEPT'])
            
            logger.info("  3.2: Semantic Resolution...")
            graph_before_resolution = deepcopy(ctx.graph)
            resolution_stats = resolve_entities_semantically(ctx.graph)
            resolution_eval = summarize_entity_resolution_effects(graph_before_resolution, ctx.graph)
            resolution_stats['evaluation'] = resolution_eval
            ctx.stats['entity_resolution'] = resolution_stats
            logger.info(f"Resolution Stats: {resolution_stats}")

            extraction_cfg = ctx.stats.get('pipeline_config', {}).get('extraction', {})
            if getattr(extraction_cfg, 'get', None) and extraction_cfg.get('diagnostic_mode'):
                diagnostics_dir = os.path.join(
                    self.settings.infra.output_dir,
                    extraction_cfg.get('diagnostic_output_subdir', 'diagnostics')
                )
                create_output_directory(diagnostics_dir)
                diagnostics_path = os.path.join(diagnostics_dir, 'entity_resolution_diagnostics.json')
                payload = {
                    'entity_resolution': resolution_stats,
                    'evaluation': resolution_eval,
                }
                with open(diagnostics_path, 'w', encoding='utf-8') as f:
                    json.dump(payload, f, indent=2, ensure_ascii=False)
                ctx.diagnostics['entity_resolution_diagnostics'] = diagnostics_path
            
        except Exception as e:
            logger.error(f"Semantic enrichment failed: {e}")
            ctx.add_error("enrichment", str(e))

    async def _step_communities(self, ctx: PipelineContext, config: Dict[str, Any]) -> None:
        try:
            from graphgen.pipeline.community.detection import CommunityDetector
            from graphgen.pipeline.community.subcommunities import add_enhanced_community_attributes_to_graph
            from graphgen.pipeline.summarization.core import generate_community_summaries
            from graphgen.config.llm import get_langchain_llm
            from graphgen.utils.vector_embedder.rag import generate_rag_embeddings

            logger.debug("Step 4: Community Detection & Summarization")

            logger.info("  4.1: Detecting Communities...")
            detector = CommunityDetector(self.settings.community)

            baseline_modularity = None
            node2vec_enabled = bool(getattr(self.settings.community, 'node2vec_enabled', False))
            if node2vec_enabled:
                logger.info("  4.1a: Computing unweighted baseline modularity...")
                baseline_graph = ctx.graph.copy()
                for u, v, d in baseline_graph.edges(data=True):
                    if d.get('graph_type') == 'entity_relation':
                        d['weight'] = 1.0
                baseline_results = detector.detect_communities(baseline_graph)
                baseline_modularity = baseline_results.get('modularity', 0.0)
                logger.info("  4.1b: Applying Node2Vec edge weights...")
                from graphgen.pipeline.embeddings.node2vec_wrapper import compute_node2vec_weights
                weights = compute_node2vec_weights(
                    ctx.graph,
                    dimensions=self.settings.community.node2vec_dimensions,
                    walk_length=self.settings.community.node2vec_walk_length,
                    num_walks=self.settings.community.node2vec_num_walks,
                    workers=1,
                    seed=self.settings.community.seed or 42,
                )

                weighted_count = 0
                for (u, v), weight in weights.items():
                    if ctx.graph.has_edge(u, v) and ctx.graph[u][v].get('graph_type') == 'entity_relation':
                        ctx.graph[u][v]['weight'] = weight
                        weighted_count += 1
                if weighted_count == 0:
                    logger.warning(
                        "Node2Vec weighting enabled but no entity-relation edge weights were produced; continuing with unweighted detection."
                    )
                else:
                    logger.info("Applied Node2Vec weights to %d entity-relation edges.", weighted_count)


            comm_results = detector.detect_communities(ctx.graph)
            communities = comm_results['assignments']

            if baseline_modularity is not None:
                comm_results['modularity_baseline'] = baseline_modularity
                comm_results['node2vec_enabled'] = True
                comm_results['modularity_delta'] = comm_results.get('modularity', 0.0) - baseline_modularity
            else:
                comm_results['node2vec_enabled'] = False

            # Save stats
            ctx.stats['communities'] = comm_results

            subcommunities = detector.detect_subcommunities_leiden(ctx.graph, communities)
            add_enhanced_community_attributes_to_graph(ctx.graph, communities, subcommunities)

            logger.info("  4.2: Generating Summaries...")
            llm = get_langchain_llm(config, purpose='summarization')
            summary_stats = await generate_community_summaries(ctx.graph, llm)
            ctx.stats['summarization'] = summary_stats

            logger.info("  4.3: Generating topic/subtopic embeddings for analytics...")
            generate_rag_embeddings(ctx.graph, node_types=['TOPIC', 'SUBTOPIC'])

        except Exception as e:
            logger.error(f"Community detection or summarization failed: {e}")
            ctx.add_error("communities", str(e))

    async def _step_topic_analysis(self, ctx: PipelineContext) -> None:
        """Run statistical tests on topic embeddings."""
        if not self.settings.analysis.topic_separation_test:
            logger.info("Step 4.5: Topic Analysis (Skipped - disabled in config)")
            return
        
        try:
            from graphgen.analytics.reporting import generate_topic_separation_report
            
            logger.info("Step 4.5: Topic Separation Analysis")
            
            # Generate statistical report
            output_path = os.path.join(
                self.settings.infra.output_dir, 
                self.settings.analysis.output_file
            )
            
            report = generate_topic_separation_report(
                ctx.graph, 
                output_path, 
                self.settings.analysis
            )
            
            ctx.stats['topic_analysis'] = {
                'output_file': output_path,
                'community_silhouette': report.get('community_level', {}).get('silhouette_score') if report.get('community_level') else None,
                'subcommunity_silhouette': report.get('subcommunity_level', {}).get('silhouette_score') if report.get('subcommunity_level') else None,
                'overall_interpretation': report.get('overall_interpretation', 'N/A')
            }
            
            logger.info(f"Topic Analysis Complete: {report.get('overall_interpretation', 'N/A')}")
            
        except Exception as e:
            logger.error(f"Topic analysis failed: {e}")
            ctx.add_error("topic_analysis", str(e))

    async def _step_pruning(self, ctx: PipelineContext) -> None:
        logger.info("Step 5: Pruning Graph...")
        # Use config if available
        config = self.settings.model_dump() if hasattr(self.settings, 'model_dump') else self.settings.dict()
        prune_stats = prune_graph(ctx.graph, config)
        ctx.stats['pruning'] = prune_stats
        logger.info(f"Pruning Stats: {prune_stats}")

    async def _step_upload(self, ctx: PipelineContext) -> None:
        if not self.uploader:
            return
            
        settings = self.settings
        db_type = settings.infra.graph_db_type if hasattr(settings.infra, 'graph_db_type') else "falkordb"
        logger.info(f"Step 6: Uploading to {db_type}...")
        try:
            if self.uploader.connect():
                stats = self.uploader.upload(ctx.graph, clean_database=settings.infra.clean_start)
                ctx.stats['upload'] = stats
                logger.info(f"Upload Stats: {stats}")
                self.uploader.close()
            else:
                logger.warning("Uploader could not connect.")
                ctx.add_error("upload", "Could not connect")
        except Exception as e:
            logger.error(f"Upload failed: {e}")
            ctx.add_error("upload", str(e))

    def _step_save_artifacts(self, ctx: PipelineContext) -> None:
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
            nx.write_graphml(clean_graph, graph_path)
            logger.info(f"GraphML saved to {graph_path}")
            
            # Save Entity Resolution Report
            er_report_path = os.path.join(output_dir, "entity_resolution_report.json")
            er_stats = {
                "extraction": ctx.stats.get('extraction', {}),
                "entity_resolution": ctx.stats.get('entity_resolution', {}),
                "pruning": ctx.stats.get('pruning', {})
            }
            with open(er_report_path, 'w') as f:
                json.dump(er_stats, f, indent=2)
            logger.info(f"Entity Resolution Report saved to {er_report_path}")

            if ctx.diagnostics:
                diagnostics_dir = os.path.join(output_dir, "diagnostics")
                create_output_directory(diagnostics_dir)
                diagnostics_path = os.path.join(diagnostics_dir, "diagnostic_index.json")
                with open(diagnostics_path, 'w') as f:
                    json.dump(ctx.diagnostics, f, indent=2)
                logger.info(f"Diagnostic index saved to {diagnostics_path}")
        except Exception as e:
            logger.error(f"Failed to save artifacts: {e}")
            ctx.add_error("artifacts", str(e))

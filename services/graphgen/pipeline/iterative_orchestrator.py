"""Iterative pipeline runner for batch experiments."""

import logging
import json
import os
import hashlib
from datetime import datetime
from collections import defaultdict
from typing import Any, Dict, List

import networkx as nx
import pandas as pd

from graphgen.config.settings import PipelineSettings
from graphgen.data_types import PipelineContext
from graphgen.pipeline.iterative_loader import IterativeLoader
from graphgen.pipeline.lexical_graph_building.builder import add_segments_to_graph
from graphgen.pipeline.entity_relation.extraction import extract_all_entities_relations
from graphgen.pipeline.community.detection import CommunityDetector
from graphgen.pipeline.community.subcommunities import add_enhanced_community_attributes_to_graph
from graphgen.pipeline.graph_cleaning.resolution import resolve_entities_semantically
from graphgen.pipeline.summarization.core import generate_community_summaries
from graphgen.analytics.reporting import generate_topic_separation_report
from graphgen.analytics.metrics import calculate_node2vec_significance
from graphgen.analytics.visualizer import plot_node2vec_impact, plot_community_centrality, plot_global_centrality
from graphgen.analytics.centrality import (
    calculate_centrality_measures, 
    get_top_entities_global, 
    get_top_entities_per_community
)
from graphgen.utils.vector_embedder.rag import generate_rag_embeddings
from graphgen.config.llm import get_langchain_llm
from graphgen.config.schema import GraphSchema
from graphgen.utils.utils import create_output_directory
from graphgen.utils.provenance import (
    write_analysis_run_manifest,
    write_pipeline_config_snapshot,
)

logger = logging.getLogger(__name__)

class IterativeOrchestrator:
    """Run the pipeline in iterative batches for experiments."""

    def __init__(self, settings: PipelineSettings, uploader=None, extractor=None) -> None:
        self.settings = settings
        self.uploader = uploader
        self.extractor = extractor
        # Use file_pattern from extraction settings
        file_pattern = settings.extraction.file_pattern if settings.extraction else "*.txt"
        self.loader = IterativeLoader(settings.infra.input_dir, settings.iterative, file_pattern)
        self.results = []

    async def run(self) -> None:
        if not self.settings.iterative.enabled:
            logger.info("Iterative mode disabled.")
            return

        logger.info(f"Starting Iterative Experiment: {self.settings.iterative.iterations} iterations, batch size {self.settings.iterative.batch_size}")
        self.run_started_at = datetime.now()

        # Ensure output directory exists before any file writes
        create_output_directory(self.settings.infra.output_dir)
        thesis_output_dir = f"{self.settings.infra.output_dir}/{self.settings.analytics.outputs_subdir}"
        create_output_directory(thesis_output_dir)
        if self.settings.analytics.save_provenance:
            write_pipeline_config_snapshot(thesis_output_dir, self.settings)
            write_analysis_run_manifest(
                thesis_output_dir,
                self.settings,
                stage="started",
                run_id="iterative",
                started_at=self.run_started_at.isoformat(),
            )

        # Initialize cumulative graph
        graph = nx.DiGraph() 
        ctx = PipelineContext(graph=graph)
        
        # Parse schema
        schema = None
        if self.settings.schema_config:
            try:
                schema = GraphSchema(**self.settings.schema_config)
            except Exception:
                pass
        
        config_dict = self.settings.model_dump()

        for i in range(self.settings.iterative.iterations):
            logger.info(f"=== Iteration {i+1}/{self.settings.iterative.iterations} ===")
            
            # 1. Load Batch
            segments = self.loader.get_batch(i)
            if not segments:
                logger.warning("No segments loaded. Stopping.")
                break
                
            logger.info(f"Loaded batch of {len(segments)} segments.")

            if self.settings.analytics.save_sampling_manifest:
                samples_dir = os.path.join(thesis_output_dir, "samples")
                create_output_directory(samples_dir)
                sample_manifest = {
                    "iteration": i + 1,
                    "batch_size": len(segments),
                    "random_seed": self.settings.iterative.random_seed + i,
                    "samples": [
                        {
                            "segment_id": seg.segment_id,
                            "filename": seg.metadata.get("filename"),
                            "line_number": seg.line_number,
                            "content_sha256": hashlib.sha256(seg.content.encode("utf-8")).hexdigest(),
                        }
                        for seg in segments
                    ],
                }
                manifest_path = os.path.join(samples_dir, f"iteration_{i+1}_sample_manifest.json")
                with open(manifest_path, "w", encoding="utf-8") as f:
                    json.dump(sample_manifest, f, indent=2, ensure_ascii=False)

            # 2. Add to Graph (Lexical)
            by_file = defaultdict(list)
            for seg in segments:
                filename = seg.metadata.get('filename', f'batch_{i}')
                by_file[filename].append(seg)
            
            for filename, file_segments in by_file.items():
                doc_id = f"DOC_{filename}"
                # Ensure Doc node exists (idempotent)
                if not ctx.graph.has_node(doc_id):
                    ctx.graph.add_node(doc_id, node_type="Document", name=filename)
                
                await add_segments_to_graph(ctx, file_segments, doc_id, config_dict, schema)

            # 3. Extraction (for new chunks only? extract_all checks tasks)
            # extraction_tasks are appended by add_segments_to_graph
            await extract_all_entities_relations(ctx, config_dict, self.extractor)
            
            # Clear tasks after extraction to avoid re-extracting old chunks next iteration
            # But wait, ctx is cumulative. 
            # `extract_all_entities_relations` dedups tasks. 
            # But we should probably clear `ctx.extraction_tasks` to keep memory low and logic clean
            ctx.extraction_tasks = []

            # --- WEIGHTED (Node2Vec) ---
            if self.settings.community.node2vec_enabled:
                from graphgen.pipeline.embeddings.node2vec_wrapper import compute_node2vec_weights, compute_node_embeddings
                
                # 4a. Compute Structural Embeddings for Resolution
                # We do this BEFORE resolution to help identify structurally similar duplicates
                logger.debug("Computing Node2Vec embeddings for resolution...")
                structural_embeddings = compute_node_embeddings(
                    ctx.graph,
                    dimensions=self.settings.community.node2vec_dimensions,
                    workers=4 # Use defaults or settings
                )
            else:
                structural_embeddings = None

            # 4. Enrichment
            # Generate RAG embeddings for Entities (needed for resolution)
            generate_rag_embeddings(ctx.graph, node_types=['ENTITY_CONCEPT'])
            resolution_stats = resolve_entities_semantically(
                ctx.graph, 
                similarity_threshold=self.settings.processing.similarity_threshold,
                structural_embeddings=structural_embeddings
            )
            
            # Save resolution report for this iteration
            er_report_path = os.path.join(thesis_output_dir, f"iteration_{i+1}_entity_resolution.json")
            with open(er_report_path, 'w', encoding='utf-8') as f:
                json.dump(resolution_stats, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved entity resolution report to {er_report_path}")
            
            # 6. Community Detection
            # --- BASELINE (Unweighted) ---
            # Create a temporary unweighted view/copy for baseline detection
            # Just stripping weights from entity edges
            baseline_graph = ctx.graph.copy()
            for u, v, d in baseline_graph.edges(data=True):
                if d.get('graph_type') == 'entity_relation':
                    d['weight'] = 1.0
            
            detector = CommunityDetector(self.settings.community)
            baseline_res = detector.detect_communities(baseline_graph)
            modularity_baseline = baseline_res.get('modularity', 0.0)
            logger.info(f"Baseline Modularity (Unweighted): {modularity_baseline:.4f}")
            del baseline_graph # Cleanup

            # --- WEIGHTED (Node2Vec) ---
            if self.settings.community.node2vec_enabled:
                from graphgen.pipeline.embeddings.node2vec_wrapper import compute_node2vec_weights
                
                logger.debug("Computing Node2Vec weights...")
                # Compute weights on the entity-relation graph only so structural
                # weighting aligns with the graph actually used for community detection.
                entity_relation_graph = detector._get_entity_graph(ctx.graph).copy()
                weights = compute_node2vec_weights(
                    entity_relation_graph, 
                    dimensions=self.settings.community.node2vec_dimensions,
                    walk_length=self.settings.community.node2vec_walk_length,
                    num_walks=self.settings.community.node2vec_num_walks,
                    seed=self.settings.iterative.random_seed + i
                )
                
                # Apply weights
                weighted_count = 0
                for (u, v), w in weights.items():
                    if ctx.graph.has_edge(u, v):
                        ctx.graph[u][v]['weight'] = w
                        weighted_count += 1
                logger.debug(f"Applied Node2Vec weights to {weighted_count} edges.")

            # Run Detection (Weighted if enabled, else defaults to 1.0)
            comm_results = detector.detect_communities(ctx.graph)
            modularity = comm_results.get('modularity', 0.0)
            communities = comm_results['assignments']
            
            # Report Delta
            if self.settings.community.node2vec_enabled:
                delta = modularity - modularity_baseline
                logger.debug(f"Node2Vec Impact: Delta={delta:+.4f} (Baseline={modularity_baseline:.4f} -> Weighted={modularity:.4f})")

            
            # 7. Summarization
            # Detect subcommunities
            subcommunities = detector.detect_subcommunities_leiden(ctx.graph, communities)
            add_enhanced_community_attributes_to_graph(ctx.graph, communities, subcommunities)
            
            # Generate Summaries (and Topic Embeddings)
            # Ideally we only summarize NEW or CHANGED communities?
            # But Leiden can change assignments globally.
            # For robustness, we re-summarize (or at least update embeddings).
            llm = get_langchain_llm(config_dict, purpose='summarization')
            await generate_community_summaries(ctx.graph, llm)
            
            # Ensure TOPIC nodes have embeddings (generate_community_summaries might generate text, but not embeddings explicitly if not in its logic? 
            # `update_community_node_with_summary_async` adds title/summary.
            # We need `generate_rag_embeddings` for TOPIC nodes after summary generation.
            generate_rag_embeddings(ctx.graph, node_types=['TOPIC', 'SUBTOPIC'])

            # 8. Graph Analytics
            # We explicitly run the topic separation statistical report for every iteration
            # as this is critical for the hypothesis testing.
            stats_report = generate_topic_separation_report(
                ctx.graph, 
                f"{self.settings.infra.output_dir}/iteration_{i+1}_report.json",
                self.settings.analytics
            )
            
            # Extract metrics for CSV
            global_separation = stats_report.get('global_separation', 0.0) or 0.0
            global_overlap = stats_report.get('global_overlap', 0.0) or 0.0
            
            # Extract silhouette scores if available
            community_sil = 0.0
            if stats_report.get('community_level'):
                community_sil = stats_report['community_level'].get('silhouette_score', 0.0) or 0.0
                 
            subcommunity_sil = 0.0
            if stats_report.get('subcommunity_level'):
                subcommunity_sil = stats_report['subcommunity_level'].get('silhouette_score', 0.0) or 0.0

            if self.settings.analytics.enabled:
                from graphgen.analytics.analyzer import GraphAnalyzer
                analyzer = GraphAnalyzer(
                    config=self.settings.analytics.model_dump(), 
                    output_base_dir=self.settings.infra.output_dir
                )
                # Reconstruct communities mapping
                communities_map = comm_results['assignments']
                
                # This runs other analytics (heatmap, KGE comparison if enabled)
                # We don't overwrite our stats with this
                _ = analyzer.run_full_analysis(ctx.graph, communities_map)

                # --- CENTRALITY ANALYTICS ---
                try:
                    logger.info("Running advanced centrality analysis...")
                    
                    # 1. Calculate Centrality
                    centrality_scores = calculate_centrality_measures(ctx.graph)
                    
                    if centrality_scores:
                        # 2. Global Top Entities
                        global_top = get_top_entities_global(centrality_scores, ctx.graph, top_k=20)
                        
                        # Save Global JSON (per iteration)
                        with open(f"{self.settings.infra.output_dir}/iteration_{i+1}_global_centrality.json", 'w') as f:
                            json.dump(global_top, f, indent=2)

                        # 3. Community Top Entities
                        comm_top = get_top_entities_per_community(centrality_scores, communities_map, ctx.graph, top_k=10)
                        
                        # Save Community JSON (per iteration)
                        with open(f"{self.settings.infra.output_dir}/iteration_{i+1}_community_centrality.json", 'w') as f:
                            json.dump(comm_top, f, indent=2)

                        # Generate plots only on the FINAL iteration
                        is_final = (i == self.settings.iterative.iterations - 1)
                        if is_final:
                            plots_dir = f"{self.settings.infra.output_dir}/centrality_plots"
                            plot_global_centrality(global_top, plots_dir)
                            plot_community_centrality(comm_top, plots_dir)
                            logger.info("Final centrality plots saved to %s", plots_dir)
                        
                        logger.info("Centrality analysis completed and saved.")
                        
                except Exception as e:
                    logger.error(f"Centrality analysis failed: {e}", exc_info=True)
                # --------------------------------

            if self.settings.analytics.save_checkpoints:
                checkpoints_dir = os.path.join(thesis_output_dir, "checkpoints")
                create_output_directory(checkpoints_dir)

                # Save GraphML checkpoint
                graph_path = os.path.join(checkpoints_dir, f"iteration_{i+1}_graph.graphml")
                clean_graph = ctx.graph.copy()
                from datetime import date, datetime as dt
                for _, d in clean_graph.nodes(data=True):
                    for k, v in list(d.items()):
                        if v is None:
                            del d[k]
                            continue
                        if isinstance(v, (dict, list)):
                            d[k] = json.dumps(v, ensure_ascii=False)
                        elif isinstance(v, (date, dt)):
                            d[k] = v.isoformat()
                for _, _, d in clean_graph.edges(data=True):
                    for k, v in list(d.items()):
                        if v is None:
                            del d[k]
                            continue
                        if isinstance(v, (dict, list)):
                            d[k] = json.dumps(v, ensure_ascii=False)
                        elif isinstance(v, (date, dt)):
                            d[k] = v.isoformat()
                nx.write_graphml(clean_graph, graph_path)

                # Save compact topic snapshot
                topics_snapshot = []
                for node_id, data in ctx.graph.nodes(data=True):
                    node_type = str(data.get("node_type", "")).upper()
                    if node_type in {"COMMUNITY", "SUBCOMMUNITY", "TOPIC", "SUBTOPIC"}:
                        embedding = data.get("embedding")
                        if hasattr(embedding, "tolist"):
                            embedding = embedding.tolist()
                        topics_snapshot.append({
                            "node_id": node_id,
                            "node_type": node_type,
                            "title": data.get("title") or data.get("name"),
                            "summary": data.get("summary"),
                            "community_id": data.get("community_id"),
                            "subcommunity_id": data.get("subcommunity_id"),
                            "embedding": embedding,
                        })
                snapshot_path = os.path.join(checkpoints_dir, f"iteration_{i+1}_topics.json")
                with open(snapshot_path, "w", encoding="utf-8") as f:
                    json.dump({"iteration": i + 1, "topics": topics_snapshot}, f, indent=2, ensure_ascii=False)
            
            result = {
                'iteration': i + 1,
                'cumulative_speeches': (i + 1) * self.settings.iterative.batch_size,
                'modularity': modularity,
                'modularity_baseline': modularity_baseline,
                'topic_separation': global_separation,
                'topic_overlap': global_overlap,
                'community_silhouette': community_sil,
                'subcommunity_silhouette': subcommunity_sil,
                'nodes': ctx.graph.number_of_nodes(),
                'edges': ctx.graph.number_of_edges(),
                'communities': comm_results.get('community_count', 0)
            }
            self.results.append(result)
            logger.info(f"Iteration {i+1} Stats: Modularity={modularity:.4f}, Separation={global_separation:.4f}, Silhouette(Com/Sub)={community_sil:.3f}/{subcommunity_sil:.3f}")
            
            # Optional: Upload intermediate?
            if self.uploader:
                # Maybe not upload every time for speed, unless requested.
                pass

        # Final Summary
        try:
            df = pd.DataFrame(self.results)
            output_csv = f"{self.settings.infra.output_dir}/iterative_experiment_results.csv"
            df.to_csv(output_csv, index=False)
            logger.info(f"Experiment complete. Results saved to {output_csv}")
            
            # Node2Vec Statistical Analysis & Plotting
            if self.settings.community.node2vec_enabled and len(self.results) >= 2:
                try:
                    # Significance Test
                    n2v_stats = calculate_node2vec_significance(self.results)
                    if n2v_stats:
                        sig_msg = "SIGNIFICANT" if n2v_stats.get('significant') else "NOT SIGNIFICANT"
                        logger.info(f"Node2Vec Improvement is {sig_msg} (p={n2v_stats.get('p_value'):.4f})")
                        
                        # Save stats to file
                        stats_path = f"{self.settings.infra.output_dir}/node2vec_stats.json"
                        with open(stats_path, 'w') as f:
                            json.dump(n2v_stats, f, indent=2)

                    # Plotting
                    plot_node2vec_impact(
                        self.results, 
                        f"{self.settings.infra.output_dir}/node2vec_impact.png"
                    )
                except Exception as e:
                    logger.error(f"Node2Vec analysis failed: {e}")
            
            # Node2Vec Statistical Analysis & Plotting
            if self.settings.community.node2vec_enabled and len(self.results) >= 2:
                try:
                    # Significance Test
                    n2v_stats = calculate_node2vec_significance(self.results)
                    if n2v_stats:
                        sig_msg = "SIGNIFICANT" if n2v_stats.get('significant') else "NOT SIGNIFICANT"
                        logger.info(f"Node2Vec Improvement is {sig_msg} (p={n2v_stats.get('p_value'):.4f})")
                        
                        # Save stats to file
                        stats_path = f"{self.settings.infra.output_dir}/node2vec_stats.json"
                        with open(stats_path, 'w') as f:
                            json.dump(n2v_stats, f, indent=2)

                    # Plotting
                    plot_node2vec_impact(
                        self.results, 
                        f"{self.settings.infra.output_dir}/node2vec_impact.png"
                    )
                except Exception as e:
                    logger.error(f"Node2Vec analysis failed: {e}")

        except Exception as e:
            logger.error(f"Failed to save CSV: {e}")

        # Generate thesis-quality plots from accumulated experiment data
        try:
            from graphgen.pipeline.visualization.thesis_plots import generate_all_thesis_plots
            thesis_results = generate_all_thesis_plots(self.settings.infra.output_dir)
            generated = len([v for v in thesis_results.values() if v])
            logger.info(f"Generated {generated} thesis plots.")
        except Exception as e:
            logger.warning(f"Thesis plot generation failed (non-critical): {e}")

        if self.settings.analytics.save_provenance:
            write_analysis_run_manifest(
                thesis_output_dir,
                self.settings,
                stage="completed",
                run_id="iterative",
                started_at=self.run_started_at.isoformat(),
                completed_at=datetime.now().isoformat(),
                extra={
                    "results": self.results,
                    "graph": {
                        "nodes": ctx.graph.number_of_nodes(),
                        "edges": ctx.graph.number_of_edges(),
                    },
                },
            )

        # Final Upload
        if self.uploader:
            db_type = self.settings.infra.graph_db_type
            logger.info(f"Uploading final iterative graph to {db_type}...")
            try:
                if self.uploader.connect():
                    # We use clean_start from settings. 
                    # Since ctx.graph is cumulative, we can wipe and replace or just merge.
                    # Wiping ensures consistency with the in-memory graph.
                    self.uploader.upload(ctx.graph, clean_database=self.settings.infra.clean_start)
                    self.uploader.close()
                    logger.info("Upload complete.")
                else:
                    logger.warning("Failed to connect to graph database for upload. Continuing without DB upload (output graphs saved locally).")
            except Exception as e:
                logger.warning(f"Upload failed: {e}. Continuing without DB upload (output graphs saved locally).")

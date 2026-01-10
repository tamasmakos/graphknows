"""
Iterative Knowledge Graph Pipeline.

Processes documents one by one and merges them into the graph database.
"""

import logging
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional
import networkx as nx
import shutil
import json
from datetime import datetime

from ..config.loader import Config
from ..service import IterativeGraphBuilder
from ..types import AgentDependencies
from ..graph.extraction import build_lexical_graph, extract_all_entities_relations
from ..embeddings.rag import generate_rag_embeddings
from ..graph.extractors import get_extractor
from ..community.detection import CommunityDetector
from ..summarization.reporting import generate_community_summary_comparison
from ..summarization.core import generate_community_summaries
from ..community.subcommunities import add_enhanced_community_attributes_to_graph
from ..llm import get_model_name, get_langchain_llm

logger = logging.getLogger(__name__)

async def run_iterative_pipeline(config_path: str, reset: bool = False) -> Dict[str, Any]:
    """
    Run the iterative graph update pipeline.
    
    Args:
        config_path: Path to configuration file
        reset: If True, reset processing state and start fresh
        
    Returns:
        Statistics about the run
    """
    from ..config.loader import load_config
    
    # Load configuration
    config = load_config(config_path)
    logger.info(f"Loaded config from: {config_path}")
    
    # Setup Run Directory
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_base = Path(config.processing.output_dir)
    run_dir = output_base / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories for graphs, analytics, and visualizations
    (run_dir / "graphs").mkdir(exist_ok=True)
    (run_dir / "analytics").mkdir(exist_ok=True)
    (run_dir / "visualizations").mkdir(exist_ok=True)
    
    # Setup File Logging
    file_handler = logging.FileHandler(run_dir / "pipeline.log")
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logging.getLogger().addHandler(file_handler)
    
    logger.info(f"📂 Created Run Directory: {run_dir}")
    
    # Save Config Snapshot (Redacted)
    try:
        import copy
        config_dict = copy.deepcopy(config.to_dict())
        
        def redact_sensitive_data(data):
            if isinstance(data, dict):
                for k, v in data.items():
                    if any(sensitive in k.lower() for sensitive in ['key', 'password', 'secret', 'token']):
                        data[k] = "***REDACTED***"
                    else:
                        redact_sensitive_data(v)
            elif isinstance(data, list):
                for item in data:
                    redact_sensitive_data(item)
            return data

        redacted_config = redact_sensitive_data(config_dict)
        
        # Add resolved LLM model name explicitly
        try:
            model_name = get_model_name(config.to_dict())
            if 'llm' not in redacted_config:
                redacted_config['llm'] = {}
            if isinstance(redacted_config['llm'], dict):
                redacted_config['llm']['resolved_model'] = model_name
        except Exception:
            pass
        
        with open(run_dir / "run_config.json", "w") as f:
            json.dump(redacted_config, f, indent=2, default=str)
    except Exception as e:
        logger.warning(f"Failed to save config snapshot: {e}")
    
    # Initialize iterative builder
    builder = IterativeGraphBuilder(config)
    
    
    # Reset if requested
    if reset:
        logger.warning("Resetting processing state and clearing database...")
        # Clear database if needed
        if config.falkordb.upload_enabled:
            if builder.uploader.connect():
                builder.uploader.clear_database()
                builder.uploader.close()
        builder.reset_state()
    
    # Ensure database schema (indexes) is created/valid
    builder.ensure_schema()
    
    # Get all available documents
    input_dir = Path(config.processing.input_dir)
    pattern = config.processing.file_pattern
    all_files = sorted(input_dir.glob(pattern))
    all_document_ids = [f.stem for f in all_files]
    
    logger.info(f"Found {len(all_document_ids)} total documents in {input_dir}")
    logger.info(f"Already processed: {len(builder.state.processed_documents)} documents")
    
    # Resolve limits based on mode
    mode = getattr(config, 'mode', 'incremental')
    mode_config = getattr(config, mode, None)
    
    # helper to get safe int value
    def get_limit(name, default=0):
        val = 0
        if mode_config:
            val = getattr(mode_config, name, 0)
        if not val and hasattr(config, 'processing'):
            val = getattr(config.processing, name, 0)
        return int(val) if val is not None else default

    speech_limit = get_limit('speech_limit')
    max_docs = get_limit('max_documents')
    
    logger.info(f"Pipeline Mode: {mode}, Speech Limit: {speech_limit}, Max Docs: {max_docs}")
    
    # Check for new documents
    new_documents = builder.get_new_documents(all_document_ids, speech_limit)
    
    if not new_documents:
        logger.info("✅ No new documents to process. Graph is up to date!")
        return {
            'status': 'up_to_date',
            'documents_processed': 0,
            'total_in_state': len(builder.state.processed_documents)
        }
    
    logger.info(f"📥 Processing {len(new_documents)} new documents incrementally...")
    
    total_stats = {
        'documents_processed': 0,
        'nodes_merged': 0,
        'relationships_merged': 0,
        'errors': []
    }
    
    # Initialize Extractor once for all documents
    extractor = get_extractor(config.to_dict())
    
    # max_docs already resolved above
    docs_processed_count = 0
    
    try:
        for doc_id in new_documents:
            # Check global limit
            if max_docs > 0 and docs_processed_count >= max_docs:
                logger.info(f"Reached max_documents limit ({max_docs}). Stopping.")
                break

            try:
                logger.info(f"\n{'='*60}")
                logger.info(f"Processing: {doc_id}")
                logger.info(f"{'='*60}")
                
                # Build graph for this single document
                graph = nx.DiGraph()
                deps = AgentDependencies(graph=graph)
                
                # 1. Lexical Graph Construction
                logger.info("Step 1: Building lexical graph...")
                await build_lexical_graph(
                    deps=deps,
                    input_dir=str(input_dir),
                    config={
                        'segment_limit': speech_limit,
                        'file_pattern': f"{doc_id}.txt"
                    }
                )
                
                # 2. Entity Extraction
                logger.info("Step 2: Extracting entities and relations...")
                extract_result = await extract_all_entities_relations(deps, config.to_dict(), extractor=extractor)
                
                # 3. Embeddings
                logger.info("Step 3: Generating embeddings...")
                graph = deps.graph
                generate_rag_embeddings(
                    graph, 
                    embedding_model=config.embeddings.model_name,
                    batch_size=32
                )
                
                # 4. Upload & Merge into Database
                logger.info("Step 4: Uploading and merging into database...")
                segments = [n for n, d in graph.nodes(data=True) if d.get('node_type') == 'SEGMENT']
                merge_stats = builder.merge_graph_incrementally(
                    graph, 
                    doc_id,
                    segment_ids=segments,
                    speech_limit=speech_limit
                )
                
                total_stats['nodes_merged'] += merge_stats['nodes_merged']
                total_stats['relationships_merged'] += merge_stats['relationships_merged']
                
                # 5. Update State
                builder.state.mark_document_processed(
                    doc_id,
                    segment_ids=segments,
                    speech_limit=speech_limit
                )
                for seg in segments:
                    builder.state.mark_segment_processed(seg)
                
                total_stats['documents_processed'] += 1
                docs_processed_count += 1
                
                logger.info(f"✅ Successfully processed and merged {doc_id}")

                # 6. Run Community Detection & Centrality (Incremental)
                comm_metrics = {}
                comm_graph_size = {'nodes': 0, 'edges': 0}
                try:
                    logger.info("Running Incremental Community Detection...")
                    full_graph = builder.fetch_entity_graph()
                    comm_graph_size['nodes'] = full_graph.number_of_nodes()
                    comm_graph_size['edges'] = full_graph.number_of_edges()
                    
                    detector = CommunityDetector()
                    comm_result = detector.detect_communities(full_graph)
                    builder.update_communities(comm_result['assignments'])
                    comm_metrics = comm_result
                    logger.info("✅ Community detection complete")
                    
                    # 6a. Subcommunity Detection
                    logger.info("Running subcommunity detection...")
                    subcommunities = detector.detect_subcommunities_leiden(
                        full_graph,
                        comm_result['assignments'],
                        min_sub_size=2,
                        max_depth=1
                    )
                    logger.info(f"✅ Detected {len(subcommunities)} subcommunities")
                    
                    # 6b. Create Hierarchy (TOPIC and SUBTOPIC nodes)
                    logger.info("Creating topic hierarchy...")
                    
                    # Create a temporary NetworkX graph with entities and communities
                    hierarchy_graph = nx.DiGraph()
                    
                    # Add entity nodes
                    for node_id in full_graph.nodes():
                        hierarchy_graph.add_node(node_id, node_type='ENTITY_CONCEPT')
                    
                    # Add edges from full_graph
                    for u, v, data in full_graph.edges(data=True):
                        hierarchy_graph.add_edge(u, v, **data)
                    
                    # Add hierarchy (TOPIC and SUBTOPIC nodes)
                    hierarchy_graph = add_enhanced_community_attributes_to_graph(
                        hierarchy_graph,
                        comm_result['assignments'],
                        subcommunities
                    )
                    
                    # 6c. Add chunks for summarization context
                    logger.info("Fetching chunks for summarization context...")
                    chunks_graph = builder.fetch_chunks_for_summarization()
                    
                    # Merge chunks_graph into hierarchy_graph
                    for node_id, node_data in chunks_graph.nodes(data=True):
                        if node_id not in hierarchy_graph:
                            hierarchy_graph.add_node(node_id, **node_data)
                        else:
                            hierarchy_graph.nodes[node_id].update(node_data)
                            
                    for u, v, edge_data in chunks_graph.edges(data=True):
                        if not hierarchy_graph.has_edge(u, v):
                            hierarchy_graph.add_edge(u, v, **edge_data)
                    
                    logger.info(f"✅ Context enrichment complete: {hierarchy_graph.number_of_nodes()} total nodes in hierarchy graph")
                    
                    llm = get_langchain_llm(config.to_dict())
                    summary_stats = await generate_community_summaries(hierarchy_graph, llm)
                    logger.info(f"✅ Summarization complete: {summary_stats.get('topics_updated', 0)} topics, {summary_stats.get('subtopics_updated', 0)} subtopics")
                    
                    # 6e. Sync hierarchy to FalkorDB
                    logger.info("Step 6e: Uploading and syncing topic hierarchy to FalkorDB...")
                    # Extract only TOPIC and SUBTOPIC nodes and their edges
                    hierarchy_subgraph = nx.DiGraph()
                    for node_id, node_data in hierarchy_graph.nodes(data=True):
                        if node_data.get('node_type') in ['TOPIC', 'SUBTOPIC']:
                            hierarchy_subgraph.add_node(node_id, **node_data)
                    
                    # Add edges connecting to/from topics
                    for u, v, edge_data in hierarchy_graph.edges(data=True):
                        u_type = hierarchy_graph.nodes[u].get('node_type')
                        v_type = hierarchy_graph.nodes[v].get('node_type')
                        if u_type in ['TOPIC', 'SUBTOPIC', 'ENTITY_CONCEPT'] and v_type in ['TOPIC', 'SUBTOPIC', 'ENTITY_CONCEPT']:
                            # Add source and target nodes if not already added
                            if u not in hierarchy_subgraph.nodes():
                                hierarchy_subgraph.add_node(u, **hierarchy_graph.nodes[u])
                            if v not in hierarchy_subgraph.nodes():
                                hierarchy_subgraph.add_node(v, **hierarchy_graph.nodes[v])
                            hierarchy_subgraph.add_edge(u, v, **edge_data)
                    
                    # Merge hierarchy into FalkorDB
                    hierarchy_stats = builder.merge_graph_incrementally(
                        hierarchy_subgraph,
                        f"{doc_id}_hierarchy",
                        segment_ids=[],
                        speech_limit=speech_limit
                    )
                    logger.info(f"✅ Synced {hierarchy_stats['nodes_merged']} hierarchy nodes, {hierarchy_stats['relationships_merged']} relationships")
                    
                    # 6f. Save Graph Artifacts
                    try:
                        # Save local graph snapshot (NetworkX JSON)
                        graph_json_path = run_dir / "graphs" / "hierarchy.json"
                        with open(graph_json_path, 'w') as f:
                            json.dump(nx.node_link_data(hierarchy_graph), f, indent=2)
                        logger.info(f"📊 Graph snapshot saved to {graph_json_path}")
                        
                        # Save Schema (into analytics/metadata/schema.json)
                        save_graph_schema(hierarchy_graph, str(run_dir / "analytics"))
                        
                        # Save Topic Comparison (into analytics/topic_summary_comparison.json)
                        generate_community_summary_comparison(hierarchy_graph, str(run_dir / "analytics"))
                    except Exception as e:
                        logger.warning(f"Failed to save graph artifacts: {e}")
                    
                except Exception as e:
                    logger.warning(f"Community detection/hierarchy/summarization failed: {e}", exc_info=True)
                
                 # Track Graph Growth Metrics
                try:
                    import csv
                    
                    metrics_file = run_dir / "analytics" / "graph_metrics.csv"
                    
                    # Check for schema migration
                    file_exists = metrics_file.exists()
                    if file_exists:
                        try:
                            with open(metrics_file, 'r') as f:
                                header = f.readline().strip()
                            if 'min_community_size' not in header:
                                logger.warning("⚠️  Old metrics CSV format detected. Archiving and creating new file.")
                                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                                backup_path = metrics_file.parent / f"graph_metrics.old.{timestamp}.csv"
                                metrics_file.rename(backup_path)
                                file_exists = False
                                logger.info(f"Archived old metrics to {backup_path}")
                        except Exception as e:
                            logger.warning(f"Failed to check metrics CSV header: {e}")

                    # Calculate metrics
                    metrics = builder.calculate_and_get_metrics()
                    
                    with open(metrics_file, 'a', newline='') as f:
                        writer = csv.writer(f)
                        if not file_exists:
                            writer.writerow([
                                'timestamp', 'document_id', 'nodes_merged', 
                                'total_nodes', 'total_edges', 
                                'avg_pagerank', 'avg_betweenness',
                                'pagerank_time_ms', 'betweenness_time_ms',
                                'modularity', 'community_count', 'community_detection_time_ms',
                                'comm_graph_nodes', 'comm_graph_edges',
                                'min_community_size', 'max_community_size', 'avg_community_size'
                            ])
                        
                        writer.writerow([
                            datetime.now().isoformat(),
                            doc_id,
                            merge_stats['nodes_merged'],
                            metrics.get('total_nodes', 0),
                            metrics.get('total_edges', 0),
                            metrics.get('avg_pagerank', 0),
                            metrics.get('avg_betweenness', 0),
                            metrics.get('pagerank_time_ms', 0),
                            metrics.get('betweenness_time_ms', 0),
                            comm_metrics.get('modularity', 0.0),
                            comm_metrics.get('community_count', 0),
                            comm_metrics.get('execution_time_ms', 0.0),
                            comm_graph_size.get('nodes', 0),
                            comm_graph_size.get('edges', 0),
                            comm_metrics.get('min_community_size', 0),
                            comm_metrics.get('max_community_size', 0),
                            comm_metrics.get('avg_community_size', 0.0)
                        ])
                    logger.info(f"📈 Graph metrics saved to {metrics_file}")
                    
                    # Explicitly log important calculated outputs
                    logger.info(f"""
    📊 Run Metrics for {doc_id}:
    ----------------------------------------
    • Nodes Merged:      {merge_stats['nodes_merged']}
    • Edges Merged:      {merge_stats['relationships_merged']}
    • Total Nodes:       {metrics.get('total_nodes', 0)}
    • Total Edges:       {metrics.get('total_edges', 0)}
    • Communities:       {comm_metrics.get('community_count', 0)}
    • Modularity:        {comm_metrics.get('modularity', 0.0):.4f}
    • Avg PageRank:      {metrics.get('avg_pagerank', 0):.6f}
    ----------------------------------------
                    """)
                except Exception as e:
                    logger.warning(f"Failed to track graph metrics: {e}")
                
            except Exception as e:
                logger.error(f"Failed to process {doc_id}: {e}", exc_info=True)
                total_stats['errors'].append({'document': doc_id, 'error': str(e)})

    finally:
        if hasattr(extractor, 'close'):
            await extractor.close()
    
    # Save state
    builder.state.save()
    
    # Backup processing state
    try:
        state_file = Path(config.processing.input_dir).parent / config.incremental.state_file # assuming relative or fix path issue
        #Actually builder.state.file_path should track this if exposed, but let's just use the config path logic or rely on where it is saved.
        # It is usually input_dir/../processing_state.json or defined in config.
        # Let's rely on builder.state which might not expose path easily without looking at code.
        # Re-reading config: state_file: "processing_state.json".
        # Loader logic determines where it is.
        # Safest is if builder.state.save() just saved it, we can copy it if we know the path.
        # Let's try to locate it.
        possible_state_file = Path(config.processing.output_dir) / config.incremental.state_file
        if not possible_state_file.exists():
             possible_state_file = Path(config.processing.input_dir) / config.incremental.state_file
             
        if possible_state_file.exists():
             shutil.copy2(possible_state_file, run_dir / "processing_state_backup.json")
             logger.info(f"💾 Backed up processing state to {run_dir / 'processing_state_backup.json'}")
    except Exception as e:
        logger.warning(f"Could not backup state file: {e}")
    
    # Save Post-Run Summary
    run_summary = {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(),
        "total_stats": total_stats,
        "config_path": str(config_path),
        "documents_processed": total_stats['documents_processed']
    }
    
    with open(run_dir / "run_summary.json", "w") as f:
        json.dump(run_summary, f, indent=2)
        
    logger.info(f"💾 Run summary saved to {run_dir / 'run_summary.json'}")

    # Post-Processing
    if total_stats['nodes_merged'] > 0:
        logger.info("\nPost-Processing Complete.")
        # Centrality & Communities already handled in loop.


    return total_stats

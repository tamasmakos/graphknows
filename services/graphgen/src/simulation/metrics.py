
import logging
import json
import networkx as nx
import numpy as np
from typing import Dict, Any, List, Optional
from datetime import datetime

from ..kg.community.detection import CommunityDetector
from ..kg.community.subcommunities import add_enhanced_community_attributes_to_graph
from ..kg.summarization.core import generate_community_summaries
from ..kg.falkordb.uploader import KnowledgeGraphUploader
from ..kg.llm import get_langchain_llm
from ..kg.config.settings import PipelineSettings
from ..kg.embeddings.model import get_model
from ..kg.embeddings.rag import generate_rag_embeddings

logger = logging.getLogger(__name__)

class GraphMetricTracker:
    """
    Tracks graph evolution by performing global analysis on the FalkorDB graph.
    """
    
    def __init__(self, settings: Optional[PipelineSettings] = None):
        self.settings = settings or PipelineSettings()
        self.uploader = KnowledgeGraphUploader(
            host=self.settings.infra.falkordb_host,
            port=self.settings.infra.falkordb_port,
            database="kg",
            postgres_config={
                "enabled": self.settings.infra.postgres_enabled,
                "host": self.settings.infra.postgres_host,
                "port": self.settings.infra.postgres_port,
                "database": self.settings.infra.postgres_db,
                "user": self.settings.infra.postgres_user,
                "password": self.settings.infra.postgres_password,
                "table_name": self.settings.infra.postgres_table
            }
        )
        
    async def analyze_evolution(self, day_index: int) -> Dict[str, Any]:
        """
        Perform a health check on the graph.
        """
        logger.info(f"🧠 [Metrics] Day {day_index} Analysis Started.")
        
        # 1. Load Graph
        try:
            graph = self.load_graph_from_falkordb()
        except Exception as e:
            logger.error(f"❌ [Metrics] Failed to load graph: {e}", exc_info=True)
            raise

        node_count = graph.number_of_nodes()
        edge_count = graph.number_of_edges()
        logger.info(f"   📥 [Metrics] Loaded Graph: {node_count} nodes, {edge_count} edges")
        
        if node_count == 0:
            logger.warning("   ⚠️ [Metrics] Graph is empty. Skipping analysis.")
            return {"modularity": 0, "node_count": 0, "top_entity": "N/A"}

        # 2. Global Community Detection
        logger.info("   🕵️ [Metrics] Running Global Community Detection (Leiden)...")
        detector = CommunityDetector()
        
        try:
            comm_results = detector.detect_communities(graph)
            communities = comm_results['assignments']
            modularity = comm_results.get('modularity', 0.0)
            logger.info(f"   ✅ [Metrics] Community Detection Complete. Modularity: {modularity:.4f}, Communities: {len(set(communities.values()))}")
            
            # Run Sub-community detection
            logger.info("   🕵️ [Metrics] Running Sub-community detection...")
            subcommunities = detector.detect_subcommunities_leiden(graph, communities)
            logger.info(f"   ✅ [Metrics] Sub-communities detected: {len(set(subcommunities.values()))}")
            
            # Apply attributes
            add_enhanced_community_attributes_to_graph(graph, communities, subcommunities)
            
        except Exception as e:
            logger.error(f"❌ [Metrics] Community Detection Failed: {e}", exc_info=True)
            raise
        
        # 3. Global Summarization
        logger.info("   📝 [Metrics] Generating Community Summaries (LLM)...")
        try:
            config_dict = self.settings.model_dump() if hasattr(self.settings, 'model_dump') else self.settings.dict()
            llm = get_langchain_llm(config_dict, purpose='summarization')
            
            summary_stats = await generate_community_summaries(graph, llm)
            logger.info(f"   ✅ [Metrics] Summarization Complete. Stats: {summary_stats}")
            
            # 3.5 Generate Embeddings for Topics
            logger.info("   🧠 [Metrics] Generating Topic Embeddings...")
            generate_rag_embeddings(graph, node_types=['TOPIC', 'SUBTOPIC'])

        except Exception as e:
             logger.error(f"❌ [Metrics] Summarization Failed: {e}", exc_info=True)
             raise
        
        # 4. Sync Updates to FalkorDB
        logger.info("   💾 [Metrics] Syncing Updates to FalkorDB...")
        try:
            await self._sync_updates(graph)
            logger.info("   ✅ [Metrics] Sync Complete.")
        except Exception as e:
            logger.error(f"❌ [Metrics] Sync Failed: {e}", exc_info=True)
            raise
        
        # 5. Compute Centrality Metrics
        logger.info("   📊 [Metrics] Computing Centrality...")
        centrality = nx.degree_centrality(graph)
        entity_centrality = {
            n: c for n, c in centrality.items() 
            if graph.nodes[n].get('node_type') == 'ENTITY_CONCEPT' or graph.nodes[n].get('label') == 'ENTITY_CONCEPT' 
        }
        
        top_entities = sorted(entity_centrality.items(), key=lambda x: x[1], reverse=True)[:5]
        top_entity_name = top_entities[0][0] if top_entities else "None"
        
        stats = {
            "day": day_index,
            "modularity": modularity,
            "node_count": node_count,
            "edge_count": edge_count,
            "community_count": len(set(communities.values())),
            "top_entity": top_entity_name,
            "top_entity": top_entity_name,
            "top_5_entities": [k for k, v in top_entities],
            "subcommunity_count": len(set(subcommunities.values())),
        }
        
        # 6. Compute Semantic Distances (Topic Average Distance)
        try:
             avg_dist = await self._compute_community_semantic_distance(graph, communities)
             stats["avg_community_distance"] = avg_dist
             logger.info(f"   📊 [Metrics] Avg Community Topic Distance: {avg_dist:.4f}")
        except Exception as e:
             logger.warning(f"Failed to compute community distances: {e}")
             stats["avg_community_distance"] = 0.0
             
             stats["avg_community_distance"] = 0.0
             
        # Log all stats nicely
        logger.info("="*30)
        logger.info(f"📊 DAY {day_index} EVOLUTION METRICS")
        for k, v in stats.items():
            if isinstance(v, float):
                logger.info(f"  • {k}: {v:.4f}")
            elif isinstance(v, list):
                logger.info(f"  • {k}: {v}")
            else:
                logger.info(f"  • {k}: {v}")
        logger.info("="*30)

        return stats

    async def _compute_community_semantic_distance(self, graph: nx.DiGraph, communities: Dict[str, int]) -> float:
        """
        Calculate average distance between community centroids based on topic embeddings.
        """
        model = get_model()
        if not model.is_available:
            return 0.0
            
        # 1. Collect Topic Summaries per Community
        community_topics = {} # comm_id -> [text]
        topics = []
        topic_node_ids = []
        
        for n, data in graph.nodes(data=True):
            if data.get('label') == 'TOPIC' or data.get('node_type') == 'TOPIC':
                # Find which community this topic belongs to
                # Typically topics represent a community.
                # Assuming TOPIC_id format implies community_id, OR we check the community assignment of elements?
                # Actually, `metrics` detects communities on ENTITIES.
                # Topics are created *based* on communities in the summarization step.
                # "TOPIC_X" where X is community ID.
                # So we can just map TOPIC_X to Community X.
                
                node_id = str(n)
                if node_id.startswith('TOPIC_'):
                    try:
                        comm_id = int(node_id.split('_')[1])
                        summary = data.get('summary', '') or data.get('title', '')
                        if summary:
                            if comm_id not in community_topics:
                                community_topics[comm_id] = []
                            community_topics[comm_id].append(summary)
                    except:
                        pass
                        
        if len(community_topics) < 2:
            return 0.0
            
        # 2. Embed and Compute Centroids
        centroids = []
        for comm_id, texts in community_topics.items():
            if not texts:
                continue
            embeddings = model.encode(texts)
            if len(embeddings) == 0:
                continue
            # Centroid = mean of embeddings
            centroid = np.mean(embeddings, axis=0)
            # Normalize for cosine distance stability (optional but good practice)
            norm = np.linalg.norm(centroid)
            if norm > 0:
                centroid = centroid / norm
            centroids.append(centroid)
            
        if len(centroids) < 2:
            return 0.0
            
        # 3. Compute Pairwise Distances (Cosine Distance = 1 - Cosine Similarity)
        # Since we normalized, Cosine Sim = dot product.
        # Dist = 1 - dot product.
        
        distances = []
        for i in range(len(centroids)):
            for j in range(i + 1, len(centroids)):
                sim = np.dot(centroids[i], centroids[j])
                dist = 1.0 - sim
                distances.append(dist)
                
        if not distances:
            return 0.0
            
        return float(np.mean(distances))

    def load_graph_from_falkordb(self) -> nx.DiGraph:
        """Fetch full graph from FalkorDB into NetworkX."""
        logger.info("   🔌 [Metrics] Connecting to FalkorDB for load...")
        if not self.uploader.connect():
             raise ConnectionError("Could not connect to FalkorDB")
             
        client = self.uploader.graph_client
        g = nx.DiGraph()
        
        try:
            # 1. Fetch Nodes
            logger.info("   ⬇️ [Metrics] Fetching Nodes...")
            res = client.query("MATCH (n) RETURN n")
            logger.info(f"   ⬇️ [Metrics] Fetched {len(res.result_set)} nodes.")
            
            for record in res.result_set:
                node = record[0] 
                props = node.properties.copy()
                node_id = props.get('id', str(node.id))
                labels = node.labels
                primary_label = labels[0] if labels else "Entity"
                if 'node_type' in props:
                    del props['node_type']
                g.add_node(node_id, **props, label=primary_label, node_type=primary_label)
                
            # 2. Fetch Relationships
            logger.info("   ⬇️ [Metrics] Fetching Relationships...")
            res_edges = client.query("MATCH (s)-[r]->(t) RETURN s.id, type(r), r, t.id")
            logger.info(f"   ⬇️ [Metrics] Fetched {len(res_edges.result_set)} relationships.")
            
            for record in res_edges.result_set:
                src_id = record[0]
                rel_type = record[1]
                rel_obj = record[2]
                tgt_id = record[3]
                
                if src_id is None or tgt_id is None:
                    continue
                    
                props = rel_obj.properties.copy()
                # Remove 'label' from props to avoid conflict with keyword argument
                props.pop('label', None)
                g.add_edge(src_id, tgt_id, label=rel_type, **props)
                
            return g
            
        except Exception as e:
            logger.error(f"Failed to load graph from FalkorDB: {e}")
            raise
        finally:
            pass

    async def _sync_updates(self, graph: nx.DiGraph):
        nodes_to_merge = []
        for n, data in graph.nodes(data=True):
            node_type = data.get('node_type') or data.get('label')
            
            if node_type in ['TOPIC', 'SUBTOPIC']:
                 nodes_to_merge.append({
                     'id': n,
                     'label': node_type,
                     'properties': data
                 })
            elif node_type == 'ENTITY_CONCEPT':
                if 'community_id' in data:
                    nodes_to_merge.append({
                        'id': n,
                        'label': node_type,
                        'properties': {
                            'id': n,
                            'community_id': data.get('community_id'),
                            'sub_community_id': data.get('sub_community_id'),
                        }
                    })
        
        if nodes_to_merge:
            logger.info(f"   💾 [Metrics] Merging {len(nodes_to_merge)} updated nodes...")
            self.uploader.merge_nodes(nodes_to_merge)
            
        edges_to_merge = []
        for u, v, data in graph.edges(data=True):
            rel_type = data.get('label') or data.get('type')
            if rel_type in ['HAS_TOPIC', 'HAS_SUBTOPIC', 'NEXT_TOPIC', 'IN_TOPIC', 'PARENT_TOPIC']:
                 edges_to_merge.append({
                     'source_id': u,
                     'target_id': v,
                     'type': rel_type,
                     'properties': data
                 })
        
        if edges_to_merge:
            logger.info(f"   💾 [Metrics] Merging {len(edges_to_merge)} structural edges...")
            self.uploader.merge_relationships(edges_to_merge)

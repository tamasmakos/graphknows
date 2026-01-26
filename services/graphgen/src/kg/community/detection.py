import logging
import networkx as nx
import numpy as np
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Optional, Set, Any

# Configure logging
logger = logging.getLogger(__name__)

# igraph and leidenalg for community detection
try:
    import igraph as ig
    import leidenalg as la
    IGRAPH_AVAILABLE = True
except ImportError:
    IGRAPH_AVAILABLE = False

class CommunityDetector:
    """Community detection using Leiden algorithm."""
    
    def __init__(self):
        """Initialize the community detector."""
        pass
    
    def _get_entity_graph(self, graph: nx.Graph) -> nx.Graph:
        """Create a view of the graph containing only entity-relation edges."""
        def filter_edge(u, v, k=None): # Handle MultiGraph if needed, though likely DiGraph/Graph
             # k is for MultiGraph keys, ignore if present
             if graph.is_multigraph():
                 edge_data = graph[u][v][k]
             else:
                 edge_data = graph[u][v]
             return edge_data.get('graph_type') == 'entity_relation'
        
        def filter_node(n):
            """Filter for entity nodes only."""
            node_type = graph.nodes[n].get('node_type', '')
            return node_type in {'ENTITY_CONCEPT'}
        
        # Create a subgraph view that only includes the specific edges AND entity nodes
        # We need to ensure we don't just get the induced subgraph of nodes, but the specific edges.
        # nx.subgraph_view with filter_edge does exactly that.
        return nx.subgraph_view(graph, filter_node=filter_node, filter_edge=filter_edge)
    
    def run_leiden(self, graph, resolution=1.0):
        """Run Leiden algorithm once."""
        if graph.number_of_nodes() < 3 or graph.number_of_edges() == 0:
            return {node: 0 for node in graph.nodes()}
        
        if not IGRAPH_AVAILABLE:
            logger.warning("igraph and leidenalg not available, using simple community detection fallback")
            # Simple fallback: group nodes by degree
            degrees = dict(graph.degree())
            max_degree = max(degrees.values()) if degrees else 1
            communities = {}
            for node, degree in degrees.items():
                # Create communities based on degree bins
                community_id = min(int(degree / (max_degree / 4)), 3)  # 4 communities max
                communities[node] = community_id
            return communities
        
        # Convert to igraph
        # Note: subgraph_view returns a view, but for igraph conversion we usually iterate edges
        node_list = list(graph.nodes())
        node_to_idx = {node: i for i, node in enumerate(node_list)}
        
        g_ig = ig.Graph()
        g_ig.add_vertices(len(node_list))
        
        # Extract edges from the view
        edge_list = [(node_to_idx[source], node_to_idx[target]) for source, target in graph.edges()]
        g_ig.add_edges(edge_list)
        
        # Extract edge weights if they exist (though usually 1.0 for entity_relation unless enriched)
        # We trust the view to give us only the correct edges
        edge_weights = [graph[u][v].get('weight', 1.0) for u, v in graph.edges()]
        
        try:
            partition_obj = la.find_partition(
                g_ig, la.RBConfigurationVertexPartition,
                resolution_parameter=resolution, 
                weights=edge_weights
            )
            
            partition = {}
            for idx, community in enumerate(partition_obj):
                for node_idx in community:
                    node_id = node_list[node_idx]
                    partition[node_id] = idx
            
            return partition
            
        except Exception as e:
            logger.error(f"Leiden algorithm failed: {e}")
            return {node: 0 for node in graph.nodes()}

    def detect_communities(self, graph) -> Dict[str, Any]:
        """Main community detection method.
        
        Subselects only the entity relation graph type and runs community detection ONLY on that subgraph.
        """
        import time
        start_time = time.time()
        
        logger.info("=== LEIDEN COMMUNITY DETECTION (OPTIMIZED) ===")
        logger.info("Subselecting 'entity_relation' graph type...")
        
        # Use the entity relation subgraph
        entity_graph = self._get_entity_graph(graph)
        
        if entity_graph.number_of_nodes() < 3 or entity_graph.number_of_edges() == 0:
            logger.info("Entity graph too small for community detection")
            return {
                "assignments": {node: 0 for node in graph.nodes()},
                "modularity": 0.0,
                "community_count": 1,
                "resolution": 1.0,
                "execution_time_ms": 0.0
            }
            
        logger.info(f"Entity Graph: {entity_graph.number_of_nodes()} nodes, {entity_graph.number_of_edges()} edges")

        # Run Leiden Once
        resolution = 1.0
        communities = self.run_leiden(entity_graph, resolution)
        
        # Recalculate modularity
        try:
            community_sets = defaultdict(set)
            for node, comm in communities.items():
                if node in entity_graph:
                    community_sets[comm].add(node)
            
            # Use entity_graph for modularity calculation
            modularity = nx.algorithms.community.modularity(entity_graph, list(community_sets.values()))
        except Exception:
            modularity = 0.0

        # Log stats
        community_counts = Counter(communities.values())
        community_sizes = list(community_counts.values())
        min_size = min(community_sizes) if community_sizes else 0
        max_size = max(community_sizes) if community_sizes else 0
        avg_size = float(np.mean(community_sizes)) if community_sizes else 0.0
        
        logger.info(f"Leiden algorithm found {len(set(communities.values()))} communities")
        logger.info(f"Stats: Min={min_size}, Max={max_size}, Avg={avg_size:.2f}")
        logger.info(f"Modularity: {modularity:.4f}")
        
        duration_ms = (time.time() - start_time) * 1000
        logger.info(f"Community detection finished in {duration_ms:.2f}ms")
        
        return {
            "assignments": communities,
            "modularity": modularity,
            "community_count": len(set(communities.values())),
            "resolution": resolution,
            "execution_time_ms": duration_ms,
            "min_community_size": min_size,
            "max_community_size": max_size,
            "avg_community_size": avg_size
        }


    def _merge_small_communities(self, g: nx.Graph, partition: Dict[str, int], min_size: int) -> Dict[str, int]:
        """Merge communities smaller than min_size into the best neighboring community."""
        if min_size <= 1:
            return partition
        from collections import defaultdict as _dd
        comm_to_nodes: Dict[int, List[str]] = _dd(list)
        for n, cid in partition.items():
            # Only include nodes that are in the partition (duh)
            comm_to_nodes[cid].append(n)
        assign = dict(partition)
        for cid, members in list(comm_to_nodes.items()):
            if len(members) >= min_size:
                continue
            # Count boundary edges to other communities
            neighbor_counts: Dict[int, int] = {}
            for n in members:
                # Use the provided graph (subgraph) neighbors
                if n in g:
                    for nbr in g.neighbors(n):
                        nid = partition.get(nbr)
                        if nid is None or nid == cid:
                            continue
                        neighbor_counts[nid] = neighbor_counts.get(nid, 0) + 1
            if not neighbor_counts:
                # fallback to largest existing community (other than cid)
                largest = None
                largest_size = -1
                for ocid, onodes in comm_to_nodes.items():
                    if ocid == cid:
                        continue
                    if len(onodes) > largest_size:
                        largest = ocid
                        largest_size = len(onodes)
                target = largest if largest is not None else cid
            else:
                target = max(neighbor_counts.items(), key=lambda x: x[1])[0]
            for n in members:
                assign[n] = target
        return assign

    def detect_subcommunities_leiden(
        self,
        entity_graph: nx.Graph, # This might be the full graph passed from pipeline
        communities: Dict[str, int],
        min_sub_size: int = 2,
        sub_resolution_min: float = 0.7, # Ignored now
        sub_resolution_max: float = 1.3, # Ignored now
        sub_resolution_steps: int = 7,  # Ignored now
        max_depth: int = 1
    ) -> Dict[str, Tuple[int, int]]:
        """Run Leiden inside each parent community to find meaningful subcommunities.

        Returns mapping node_id -> (parent_community_id, local_sub_id).
        """
        if max_depth <= 0:
            return {}
        
        # Ensure we are using the entity subgraph
        filtered_graph = self._get_entity_graph(entity_graph)

        node_to_sub: Dict[str, Tuple[int, int]] = {}
        # Group nodes by parent community
        by_comm: Dict[int, List[str]] = defaultdict(list)
        for n, cid in communities.items():
            by_comm[cid].append(n)

        # Fixed resolution for sub-communities
        gamma = 1.0

        for comm_id, nodes in by_comm.items():
            if len(nodes) < max(2 * min_sub_size, 4):
                continue
            
            # Create subgraph of the community using the filtered graph
            subg = filtered_graph.subgraph(nodes).copy()
            
            if subg.number_of_edges() == 0:
                 continue

            # Run Leiden once
            part = self.run_leiden(subg, gamma)
            
            if len(set(part.values())) < 2:
                continue
                
            fixed = self._merge_small_communities(subg, part, min_sub_size)
            if len(set(fixed.values())) < 2:
                continue
            
            # Store results
            old_to_local: Dict[int, int] = {}
            next_local = 0
            for n in nodes:
                if n in fixed: # Should be all nodes in subg
                    sid = fixed[n]
                    if sid not in old_to_local:
                        old_to_local[sid] = next_local
                        next_local += 1
                    node_to_sub[n] = (comm_id, old_to_local[sid])

        return node_to_sub

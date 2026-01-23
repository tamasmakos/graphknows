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
    
    def run_leiden_with_consistency(self, graph, resolution, n_runs=5):
        """Run Leiden algorithm multiple times and select most consistent result."""
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
        node_list = list(graph.nodes())
        node_to_idx = {node: i for i, node in enumerate(node_list)}
        
        g_ig = ig.Graph()
        g_ig.add_vertices(len(node_list))
        edge_list = [(node_to_idx[source], node_to_idx[target]) for source, target in graph.edges()]
        g_ig.add_edges(edge_list)
        
        # Extract edge weights from the NetworkX graph in the same order as the edges
        # Default to 1.0 if a weight is missing for any reason.
        edge_weights = [graph[u][v].get('weight', 1.0) for u, v in graph.edges()]
        
        # Run multiple times
        all_partitions = []
        community_counts = []
        
        for i in range(n_runs):
            try:
                partition_obj = la.find_partition(
                    g_ig, la.RBConfigurationVertexPartition,
                    resolution_parameter=resolution, 
                    seed=i,
                    weights=edge_weights
                )
                
                partition = {}
                for idx, community in enumerate(partition_obj):
                    for node_idx in community:
                        node_id = node_list[node_idx]
                        partition[node_id] = idx
                
                all_partitions.append(partition)
                community_counts.append(len(set(partition.values())))
                
            except Exception:
                continue
        
        if not all_partitions:
            return {node: 0 for node in graph.nodes()}
        
        # Select most common number of communities
        most_common_count = Counter(community_counts).most_common(1)[0][0]
        for i, partition in enumerate(all_partitions):
            if community_counts[i] == most_common_count:
                return partition
        
        return all_partitions[0]
    
    def optimize_resolution(self, graph):
        """Find optimal resolution parameter with comprehensive evaluation."""
        logger.info("Optimizing resolution parameter in range (0.3, 2.0) with 15 steps...")
        
        resolution_values = np.linspace(0.1, 3.0, 30)
        results = []
        
        for resolution in resolution_values:
            try:
                # Run multiple times for consistency evaluation
                logger.info("Evaluating community consistency across 3 runs...")
                consistency_results = []
                community_counts = []
                
                for run in range(7):
                    partition = self.run_leiden_with_consistency(graph, resolution, n_runs=1)
                    consistency_results.append(partition)
                    community_counts.append(len(set(partition.values())))
                
                # Calculate consistency score (prefer NMI, fallback to pairwise agreement)
                if len(consistency_results) >= 2:
                    consistency_score = self.calculate_partition_consistency_nmi(consistency_results)
                else:
                    consistency_score = 1.0
                
                logger.info(f"Community consistency score: {consistency_score:.3f}")
                logger.info(f"Community counts across runs: {community_counts}")
                
                # Use the first partition for modularity calculation
                partition = consistency_results[0]
                
                # Calculate modularity
                try:
                    community_sets = []
                    for comm_id in set(partition.values()):
                        community_nodes = set([node for node, comm in partition.items() if comm == comm_id])
                        community_nodes = community_nodes.intersection(set(graph.nodes()))
                        if community_nodes:
                            community_sets.append(community_nodes)
                    
                    modularity = nx.algorithms.community.modularity(graph, community_sets)
                except:
                    modularity = 0.0
                
                logger.info(f"Resolution {resolution:.2f}: {len(set(partition.values()))} communities, "
                           f"consistency={consistency_score:.3f}, modularity={modularity:.3f}")
                
                results.append({
                    'resolution': resolution,
                    'modularity': modularity,
                    'consistency': consistency_score,
                    'n_communities': len(set(partition.values())),
                    'partition': partition,
                    'score': consistency_score * 0.6 + modularity * 0.4  # Combined score
                })
                
            except Exception as e:
                logger.warning(f"Failed to evaluate resolution {resolution}: {e}")
                continue
        
        if not results:
            logger.warning("No valid results, using default resolution 1.0")
            return 1.0, self.run_leiden_with_consistency(graph, 1.0)
        
        # Select best based on combined score (consistency + modularity)
        best_result = max(results, key=lambda x: x['score'])
        
        logger.info(f"Best resolution: {best_result['resolution']:.2f} "
                   f"(consistency={best_result['consistency']:.3f}, "
                   f"modularity={best_result['modularity']:.3f})")
        
        return best_result['resolution'], best_result['partition']
    
    def calculate_partition_consistency(self, partitions):
        """Calculate consistency score between multiple partitions."""
        if len(partitions) < 2:
            return 1.0
        
        agreement_scores = []
        
        for i in range(len(partitions)):
            part_i = partitions[i]
            for j in range(i + 1, len(partitions)):
                part_j = partitions[j]
                
                # Get common nodes
                common_nodes = list(set(part_i.keys()) & set(part_j.keys()))
                
                if not common_nodes:
                    continue
                
                # Count node pairs that have the same community assignment
                same_assignment = 0
                total_pairs = 0
                
                for n1_idx in range(len(common_nodes)):
                    n1 = common_nodes[n1_idx]
                    for n2_idx in range(n1_idx + 1, len(common_nodes)):
                        n2 = common_nodes[n2_idx]
                        
                        # Check if nodes are in same community in both partitions
                        same_in_i = part_i[n1] == part_i[n2]
                        same_in_j = part_j[n1] == part_j[n2]
                        
                        if (same_in_i and same_in_j) or (not same_in_i and not same_in_j):
                            same_assignment += 1
                        
                        total_pairs += 1
                
                if total_pairs > 0:
                    agreement = same_assignment / total_pairs
                    agreement_scores.append(agreement)
        
        # Return average agreement
        return sum(agreement_scores) / len(agreement_scores) if agreement_scores else 0.0

    def calculate_partition_consistency_nmi(self, partitions):
        """Calculate consistency using Normalized Mutual Information (NMI) when available.

        Falls back to calculate_partition_consistency if sklearn is not installed or an error occurs.
        """
        if len(partitions) < 2:
            return 1.0
        try:
            from sklearn.metrics import normalized_mutual_info_score
        except Exception:
            return self.calculate_partition_consistency(partitions)

        try:
            scores = []
            for i in range(len(partitions)):
                part_i = partitions[i]
                for j in range(i + 1, len(partitions)):
                    part_j = partitions[j]
                    # Intersect node sets
                    common_nodes = list(set(part_i.keys()) & set(part_j.keys()))
                    if not common_nodes:
                        continue
                    labels_i = [part_i[n] for n in common_nodes]
                    labels_j = [part_j[n] for n in common_nodes]
                    nmi = normalized_mutual_info_score(labels_i, labels_j)
                    scores.append(float(nmi))
            return float(sum(scores) / len(scores)) if scores else 0.0
        except Exception:
            return self.calculate_partition_consistency(partitions)
    
    def detect_communities(self, graph) -> Dict[str, Any]:
        """Main community detection method with comprehensive optimization.
        
        Uses edge weights for community detection. If similarity edges have been
        added to the graph (via compute_embedding_similarity_edges), they will
        be used to improve community quality by grouping semantically similar
        entities together.
        
        Args:
            graph: NetworkX graph with optional 'weight' attribute on edges
            
        Returns:
            Dictionary containing:
            - assignments: Dict mapping node_id to community_id
            - modularity: Modularity score of the partition
            - community_count: Number of communities found
            - resolution: Resolution parameter used
            - execution_time_ms: Execution time in milliseconds
        """
        import time
        start_time = time.time()
        
        logger.info("=== LEIDEN COMMUNITY DETECTION ===")
        logger.info("Running Leiden algorithm with edge weights (includes similarity edges if present)")
        
        if graph.number_of_nodes() < 3 or graph.number_of_edges() == 0:
            logger.info("Graph too small for community detection")
            return {
                "assignments": {node: 0 for node in graph.nodes()},
                "modularity": 0.0,
                "community_count": 1,
                "resolution": 1.0,
                "execution_time_ms": 0.0
            }
        
        # Count edges with custom weights (similarity edges)
        weighted_edges = sum(1 for _, _, d in graph.edges(data=True) if d.get('weight', 1.0) != 1.0)
        similarity_edges = sum(1 for _, _, d in graph.edges(data=True) if d.get('graph_type') == 'similarity')
        logger.info(f"Graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges "
                   f"({weighted_edges} weighted, {similarity_edges} similarity)")
        
        # Step 1: Comprehensive resolution optimization for entire graph
        # optimize_resolution returns (resolution, partition) - ideally it should return metrics too
        # But we can recalculate modularity easily or refactor optimize_resolution.
        # Let's inspect optimize_resolution again... 
        # It calculates best_result (dict) but returns only res, partition.
        # Let's just refactor optimize_resolution usage or trust it picked the best.
        # Actually, let's just make detect_communities calculate modularity again or grab it if we refactor optimize_resolution.
        # Since I'm only editing this block, I can't easily change optimize_resolution's signature without editing lines 91-163. 
        # But wait, lines 91-163 are earlier in the file. I CAN edit multiple chunks if I use multi_replace, but replace_file_content is single block.
        # I'll stick to calculating modularity here for simplicty, or just use what I have.
        
        best_resolution, communities = self.optimize_resolution(graph)
        
        logger.info(f"Using optimized resolution {best_resolution:.2f}")
        
        # Recalculate modularity for the final result (it's cheap)
        try:
            community_sets = defaultdict(set)
            for node, comm in communities.items():
                if node in graph:
                    community_sets[comm].add(node)
            
            modularity = nx.algorithms.community.modularity(graph, list(community_sets.values()))
        except Exception:
            modularity = 0.0

        # Log final community statistics
        community_counts = Counter(communities.values())
        community_sizes = list(community_counts.values())
        
        min_size = min(community_sizes) if community_sizes else 0
        max_size = max(community_sizes) if community_sizes else 0
        avg_size = float(np.mean(community_sizes)) if community_sizes else 0.0
        
        logger.info(f"Leiden algorithm found {len(set(communities.values()))} communities")
        logger.info(f"Community sizes: {dict(community_counts)}")
        logger.info(f"Stats: Min={min_size}, Max={max_size}, Avg={avg_size:.2f}")
        
        duration_ms = (time.time() - start_time) * 1000
        logger.info(f"Community detection finished in {duration_ms:.2f}ms")
        
        return {
            "assignments": communities,
            "modularity": modularity,
            "community_count": len(set(communities.values())),
            "resolution": best_resolution,
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
            comm_to_nodes[cid].append(n)
        assign = dict(partition)
        for cid, members in list(comm_to_nodes.items()):
            if len(members) >= min_size:
                continue
            # Count boundary edges to other communities
            neighbor_counts: Dict[int, int] = {}
            for n in members:
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
        entity_graph: nx.Graph,
        communities: Dict[str, int],
        min_sub_size: int = 2,
        sub_resolution_min: float = 0.7,
        sub_resolution_max: float = 1.3,
        sub_resolution_steps: int = 7,
        max_depth: int = 1
    ) -> Dict[str, Tuple[int, int]]:
        """Run Leiden inside each parent community to find meaningful subcommunities.

        Returns mapping node_id -> (parent_community_id, local_sub_id).
        """
        if max_depth <= 0:
            return {}
        node_to_sub: Dict[str, Tuple[int, int]] = {}
        # Group nodes by parent community
        by_comm: Dict[int, List[str]] = defaultdict(list)
        for n, cid in communities.items():
            by_comm[cid].append(n)

        gammas = list(np.linspace(sub_resolution_min, sub_resolution_max, max(2, sub_resolution_steps)))

        for comm_id, nodes in by_comm.items():
            if len(nodes) < max(2 * min_sub_size, 4):
                continue
            subg = entity_graph.subgraph(nodes).copy()
            # Baseline modularity: one cluster
            try:
                baseline_mod = nx.algorithms.community.modularity(subg, [set(nodes)])
            except Exception:
                baseline_mod = 0.0

            best_fixed: Optional[Dict[str, int]] = None
            best_mod = -1e9

            for gamma in gammas:
                part = self.run_leiden_with_consistency(subg, gamma, n_runs=3)
                if len(set(part.values())) < 2:
                    continue
                fixed = self._merge_small_communities(subg, part, min_sub_size)
                if len(set(fixed.values())) < 2:
                    continue
                comm_sets = []
                for sid in set(fixed.values()):
                    comm_sets.append({n for n, c in fixed.items() if c == sid})
                try:
                    mod = nx.algorithms.community.modularity(subg, comm_sets)
                except Exception:
                    mod = 0.0
                if mod > best_mod:
                    best_mod = mod
                    best_fixed = fixed

            if best_fixed is None:
                continue
            if best_mod <= baseline_mod:
                continue
            # Relabel local ids densely
            old_to_local: Dict[int, int] = {}
            next_local = 0
            for n in nodes:
                sid = best_fixed[n]
                if sid not in old_to_local:
                    old_to_local[sid] = next_local
                    next_local += 1
                node_to_sub[n] = (comm_id, old_to_local[sid])

        return node_to_sub

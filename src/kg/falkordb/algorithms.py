"""
FalkorDB Native Algorithms Wrapper.

This module provides a wrapper around FalkorDB's Graph Data Science (GDS) procedures
to execute algorithms like PageRank, Betweenness Centrality, and others directly
on the database.
"""

import logging
from typing import Dict, Any, List, Optional
from falkordb import FalkorDB, Graph

logger = logging.getLogger(__name__)


class FalkorDBAlgorithms:
    """Wrapper for FalkorDB native graph algorithms."""

    def __init__(self, graph: Graph):
        """
        Initialize with a connected FalkorDB graph client.
        
        Args:
            graph: Simple Client for FalkorDB graph
        """
        self.graph = graph

    def run_pagerank(
        self, 
        iterations: int = 20, 
        damping_factor: float = 0.85, 
        write_property: str = 'pagerank',
        label: Optional[str] = None,
        relationship_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Run PageRank algorithm.
        
        Args:
            iterations: Number of iterations
            damping_factor: Damping factor
            write_property: Property name to write results to
            label: Node label to restrict prediction to (optional)
            relationship_type: Relationship type to traverse (optional)
            
        Returns:
            Statistics about the execution including timing
        """
        import time
        start_time = time.time()
        logger.info(f"Running PageRank (iter={iterations}, damping={damping_factor}, label={label}, rel={relationship_type})...")
        
        try:
            # FalkorDB syntax: gds.pageRank(graph_name, {param: value, ...})
            # Note: This version of FalkorDB/RedisGraph algo.pageRank requires exactly 2 arguments (Label, Type).
            # Passing 'null' checks all.
            
            label_arg = f"'{label}'" if label else "null"
            rel_arg = f"'{relationship_type}'" if relationship_type else "null"
            
            query = f"""
            CALL algo.pageRank({label_arg}, {rel_arg})
            YIELD node, score
            SET node.{write_property} = score
            """
            
            # Execute
            self.graph.query(query)
            
            duration_ms = (time.time() - start_time) * 1000
            logger.info(f"PageRank finished in {duration_ms:.2f}ms")
            
            return {
                "status": "success", 
                "algorithm": "PageRank",
                "execution_time_ms": duration_ms
            }
            
        except Exception as e:
            logger.error(f"PageRank failed: {e}")
            raise

    def run_betweenness_centrality(
        self, 
        write_property: str = 'betweenness',
        label: Optional[str] = None,
        relationship_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Run Betweenness Centrality algorithm.
        
        Args:
            write_property: Property name to write results to
            label: Node label to restrict prediction to (optional)
            relationship_type: Relationship type to traverse (optional)
            
        Returns:
            Statistics about the execution
        """
        import time
        start_time = time.time()
        logger.info(f"Running Betweenness Centrality (label={label}, rel={relationship_type})...")
        
        try:
            label_arg = f"'{label}'" if label else "null"
            rel_arg = f"'{relationship_type}'" if relationship_type else "null"

            query = f"""
            CALL algo.betweenness({label_arg}, {rel_arg})
            YIELD node, score
            SET node.{write_property} = score
            """
            
            self.graph.query(query)
            
            duration_ms = (time.time() - start_time) * 1000
            logger.info(f"Betweenness Centrality finished in {duration_ms:.2f}ms")
            
            return {
                "status": "success", 
                "algorithm": "Betweenness Centrality",
                "execution_time_ms": duration_ms
            }
            
        except Exception as e:
            logger.error(f"Betweenness Centrality failed: {e}")
            raise



    def get_graph_metrics(
        self,
        label: Optional[str] = None,
        relationship_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Calculate and retrieve graph metrics.
        
        This first runs centrality algorithms to ensure the graph properties 
        are up to date, then aggregates the results.
        
        Args:
            label: Node label to restrict metrics/centrality to
            relationship_type: Relationship type to traverse
            
        Returns:
            Dictionary containing:
            - total_nodes
            - total_edges
            - avg_pagerank
            - avg_betweenness
            - avg_closeness (optional, if enabled)
            - pagerank_time_ms
            - betweenness_time_ms
        """
        logger.info(f"Calculating graph metrics (counts + centrality averages) for label={label}...")
        
        metrics = {}
        
        # 1. Run centrality algorithms first to update properties
        # We catch errors so we can still return count metrics even if algorithms fail
        try:
            pr_stats = self.run_pagerank(label=label, relationship_type=relationship_type)
            metrics['pagerank_time_ms'] = pr_stats.get('execution_time_ms', 0)
        except Exception as e:
            logger.warning(f"PageRank calc failed during metrics collection: {e}")
            metrics['pagerank_time_ms'] = 0
            
        try:
            bc_stats = self.run_betweenness_centrality(label=label, relationship_type=relationship_type)
            metrics['betweenness_time_ms'] = bc_stats.get('execution_time_ms', 0)
        except Exception as e:
            logger.warning(f"Betweenness calc failed during metrics collection: {e}")
            metrics['betweenness_time_ms'] = 0
            
        # 2. Get Counts and Averages
        try:
            # We can do this in one query or a few simple ones.
            # Cypher aggregation is efficient.
            
            query = """
            MATCH (n)
            OPTIONAL MATCH (n)-[r]->()
            RETURN 
                count(DISTINCT n) as total_nodes, 
                count(r) as total_edges,
                avg(n.pagerank) as avg_pagerank,
                avg(n.betweenness) as avg_betweenness,
                avg(n.closeness) as avg_closeness
            """
            
            res = self.graph.query(query)
            
            if res.result_set:
                row = res.result_set[0]
                metrics['total_nodes'] = row[0]
                metrics['total_edges'] = row[1]
                
                # Handle None values if properties don't exist
                metrics['avg_pagerank'] = row[2] if row[2] is not None else 0.0
                metrics['avg_betweenness'] = row[3] if row[3] is not None else 0.0
                metrics['avg_closeness'] = row[4] if row[4] is not None else 0.0
                
            logger.info(f"Graph Metrics: Nodes={metrics.get('total_nodes')}, Edges={metrics.get('total_edges')}, "
                       f"AvgPR={metrics.get('avg_pagerank', 0):.4f}, AvgBC={metrics.get('avg_betweenness', 0):.2f}, "
                       f"Time(PR)={metrics.get('pagerank_time_ms', 0):.2f}ms, Time(BC)={metrics.get('betweenness_time_ms', 0):.2f}ms")
            
        except Exception as e:
            logger.error(f"Failed to query graph metrics: {e}")
            # process failed, return zeroes or partial
            metrics.update({
                'total_nodes': 0, 
                'total_edges': 0, 
                'avg_pagerank': 0.0, 
                'avg_betweenness': 0.0
            })
            
        return metrics

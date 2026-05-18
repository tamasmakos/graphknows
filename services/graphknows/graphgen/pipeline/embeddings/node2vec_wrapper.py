import logging
import networkx as nx
import numpy as np
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

try:
    from node2vec import Node2Vec
    NODE2VEC_AVAILABLE = True
except ImportError:
    NODE2VEC_AVAILABLE = False

def train_node2vec_model(
    graph: nx.Graph,
    dimensions: int = 64,
    walk_length: int = 16,
    num_walks: int = 20,
    window: int = 10,
    min_count: int = 1,
    batch_words: int = 4,
    workers: int = 1,
    seed: int = 42
):
    """
    Train and return Node2Vec model.
    """
    if not NODE2VEC_AVAILABLE:
        logger.warning("node2vec not installed. Skipping.")
        return None

    # 1. Prepare Graph (Undirected, Unweighted for training)
    g_train = nx.Graph()
    for u, v in graph.edges():
        g_train.add_edge(u, v, weight=1.0)
        
    if g_train.number_of_nodes() < 5:
        logger.warning("Graph too small for Node2Vec.")
        return None

    logger.info(f"Training Node2Vec on {g_train.number_of_nodes()} nodes...")
    
    try:
        n2v = Node2Vec(
            g_train, 
            dimensions=dimensions, 
            walk_length=walk_length, 
            num_walks=num_walks, 
            workers=workers, 
            quiet=True,
            seed=seed
        )
        model = n2v.fit(
            window=window, 
            min_count=min_count, 
            batch_words=batch_words,
            epochs=1
        )
        return model
    except Exception as e:
        logger.error(f"Node2Vec training failed: {e}")
        return None

def compute_node_embeddings(
    graph: nx.Graph,
    dimensions: int = 64,
    walk_length: int = 16,
    num_walks: int = 20,
    window: int = 10,
    workers: int = 1,
    seed: int = 42
) -> Dict[str, np.ndarray]:
    """
    Compute Node2Vec embeddings for all nodes in the graph.
    Returns Dict[node_id, embedding_array]
    """
    model = train_node2vec_model(
        graph, dimensions, walk_length, num_walks, window, 
        workers=workers, seed=seed
    )
    
    if not model:
        return {}
        
    embeddings = {}
    for node in graph.nodes():
        if str(node) in model.wv:
            embeddings[node] = model.wv[str(node)]
            
    return embeddings

def compute_node2vec_weights(
    graph: nx.Graph,
    dimensions: int = 64,
    walk_length: int = 16,
    num_walks: int = 20,
    window: int = 10,
    min_count: int = 1,
    batch_words: int = 4,
    workers: int = 1,
    seed: int = 42
) -> Dict[str, float]:
    """
    Train Node2Vec and compute edge weights based on cosine similarity.
    
    Returns a dictionary of (u, v) -> weight.
    Only computes weights for edges that already exist in the graph.
    """
    model = train_node2vec_model(
        graph, dimensions, walk_length, num_walks, window, 
        min_count, batch_words, workers, seed
    )
    
    if not model:
        return {}

    # 3. Compute Weights for Existing Edges
    edge_weights = {}
    weighted_count = 0
    
    for u, v in graph.edges():
        if str(u) in model.wv and str(v) in model.wv:
            vec_u = model.wv[str(u)]
            vec_v = model.wv[str(v)]
            
            # Cosine Similarity
            norm_u = np.linalg.norm(vec_u)
            norm_v = np.linalg.norm(vec_v)
            
            if norm_u == 0 or norm_v == 0:
                sim = 0.0
            else:
                sim = float(np.dot(vec_u, vec_v) / (norm_u * norm_v))
            
            # Weight must be positive.
            # Use raw similarity clamped to small positive, or (1+sim)/2
            # Sanity check used max(0.001, sim)
            weight = max(0.001, sim)
            
            # Tuple key for edge
            edge_weights[(u, v)] = weight
            weighted_count += 1
            
    logger.info(f"Computed Node2Vec weights for {weighted_count} edges.")
    return edge_weights

def apply_node2vec_weights(graph: nx.Graph, weights: Dict[str, float]) -> None:
    """Apply computed weights to the graph in-place."""
    count = 0
    for (u, v), w in weights.items():
        if graph.has_edge(u, v):
            graph[u][v]['weight'] = w
            count += 1
    logger.info(f"Applied weights to {count} edges in graph.")

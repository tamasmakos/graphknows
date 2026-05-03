
import networkx as nx
import logging
import random

logger = logging.getLogger(__name__)

def generate_rag_embeddings(graph: nx.DiGraph):
    """Mock RAG embeddings (random vectors) for testing."""
    logger.info("Generating mock embeddings (dim=384)...")
    
    count = 0
    for node_id, data in graph.nodes(data=True):
        if data.get('node_type') in ['CHUNK', 'TextChunk', 'Document', 'TOPIC']:
            # Generate random 384-d vector
            data['embedding'] = [random.random() for _ in range(384)]
            count += 1
            
    logger.info(f"Added mock embeddings to {count} nodes.")

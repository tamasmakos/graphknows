"""
RAG Embeddings Generation for Knowledge Graph.

Generates embeddings for ALL node types in the knowledge graph using
SentenceTransformers for RAG (Retrieval-Augmented Generation) applications.

Supported node types:
- ENTITY_CONCEPT: Embedded using name + entity_type context
- TOPIC: Embedded using title + summary + contained entities
- SUBTOPIC: Embedded using title + summary + contained entities
- CHUNK: Embedded using text
- EPISODE: Embedded using content (includes speaker metadata as properties)
"""

import networkx as nx
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

try:
    from sentence_transformers import SentenceTransformer
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False


def get_embedding_dimension(model_name: str = "all-MiniLM-L6-v2") -> int:
    """Get the dimension of the embedding model."""
    if not EMBEDDINGS_AVAILABLE:
        return 384  # Default fallback
    
    try:
        # Try loading from local cache first to avoid network timeouts
        model = SentenceTransformer(model_name, local_files_only=True)
        return int(model.get_sentence_embedding_dimension())
    except Exception:
        # Fallback to default loading (might download)
        try:
            model = SentenceTransformer(model_name)
            return int(model.get_sentence_embedding_dimension())
        except Exception:
            return 384


def _get_embedding_text_for_node(
    node_id: str,
    node_data: Dict,
    graph: nx.DiGraph
) -> Optional[str]:
    """
    Extract text for embedding based on node type.
    
    Returns None if no suitable text is found.
    """
    node_type = node_data.get('node_type', '')
    
    if node_type == 'ENTITY_CONCEPT':
        # Entity: name + entity_type context
        text_parts = []
        if 'name' in node_data:
            text_parts.append(node_data['name'])
        if 'entity_type' in node_data:
            text_parts.append(f"This is a {node_data['entity_type']}")
        return ' '.join(text_parts) if text_parts else None
    
    elif node_type in ['TOPIC', 'SUBTOPIC']:
        # Topic: title + summary + contained entities
        text_parts = []
        if 'title' in node_data:
            text_parts.append(node_data['title'])
        if 'summary' in node_data:
            text_parts.append(node_data['summary'])
        
        # Find related entities
        related_entities = []
        for pred in graph.predecessors(node_id):
            pred_data = graph.nodes.get(pred, {})
            if pred_data.get('node_type') == 'ENTITY_CONCEPT':
                entity_name = pred_data.get('name', pred)
                related_entities.append(entity_name)
        
        if related_entities:
            text_parts.append(f"Contains entities: {', '.join(related_entities[:10])}")
        
        return ' '.join(text_parts) if text_parts else None
    
    elif node_type == 'CHUNK':
        # Chunk: text only
        chunk_text = node_data.get('text', '')
        if isinstance(chunk_text, str) and chunk_text.strip():
            return chunk_text
        elif isinstance(chunk_text, list):
            return ' '.join(chunk_text)
        
        return None
    
    elif node_type == 'EPISODE':
        # Episode: content
        content = node_data.get('content', '')
        if isinstance(content, str) and len(content) > 10:
            # Limit episode content to first 2000 chars to keep embedding focused
            return content[:2000] if len(content) > 2000 else content
        return None
    
    # DAY type - typically not embedded (date-based container)
    elif node_type == 'DAY':
        # Could embed date metadata if needed
        return None
    
    return None


def generate_rag_embeddings(
    graph: nx.DiGraph,
    embedding_model: str = "all-MiniLM-L6-v2",
    batch_size: int = 32,
    node_types: Optional[List[str]] = None
) -> Dict[str, np.ndarray]:
    """
    Generate embeddings for nodes in the knowledge graph.
    
    Generates embeddings for all supported node types and stores them
    directly on the graph nodes as 'embedding' property.
    
    Args:
        graph: NetworkX DiGraph containing the knowledge graph
        embedding_model: SentenceTransformer model name
            - 'all-MiniLM-L6-v2': 384 dims, fast, good quality (default)
            - 'all-mpnet-base-v2': 768 dims, best quality, slower
        batch_size: Batch size for embedding generation
        node_types: Optional list of node types to embed (default: all supported types)
        
    Returns:
        Dictionary mapping node_id to embedding numpy array
    """
    if not EMBEDDINGS_AVAILABLE:
        logger.warning("SentenceTransformers not available. Skipping embedding generation.")
        return {}
    
    # Default to all supported node types
    if node_types is None:
        node_types = ['ENTITY_CONCEPT', 'TOPIC', 'SUBTOPIC', 'CHUNK', 'EPISODE']
    
    logger.info(f"Generating RAG embeddings for node types: {node_types}")
    logger.info(f"Using embedding model: {embedding_model}")
    
    # Load embedding model
    try:
        logger.info(f"Attempting to load {embedding_model} from local cache...")
        model = SentenceTransformer(embedding_model, local_files_only=True)
    except Exception as e:
        logger.warning(f"Could not load from local cache ({e}). Attempting download...")
        model = SentenceTransformer(embedding_model)
    embedding_dim = model.get_sentence_embedding_dimension()
    logger.info(f"Embedding dimension: {embedding_dim}")
    
    # Collect texts to embed per node type
    texts_to_embed: List[str] = []
    node_ids: List[str] = []
    node_type_counts: Dict[str, int] = {}
    
    for node_id, node_data in graph.nodes(data=True):
        node_type = node_data.get('node_type', '')
        
        if node_type not in node_types:
            continue
        
        # Get text for embedding
        text = _get_embedding_text_for_node(node_id, node_data, graph)
        
        if text:
            texts_to_embed.append(text)
            node_ids.append(node_id)
            node_type_counts[node_type] = node_type_counts.get(node_type, 0) + 1
    
    if not texts_to_embed:
        logger.warning("No texts found for embedding generation")
        return {}
    
    logger.info(f"Found {len(texts_to_embed)} nodes to embed:")
    for node_type, count in node_type_counts.items():
        logger.info(f"  - {node_type}: {count}")
    
    # Generate embeddings in batches
    logger.info(f"Generating embeddings in batches of {batch_size}...")
    all_embeddings: List[np.ndarray] = []
    
    for i in range(0, len(texts_to_embed), batch_size):
        batch_texts = texts_to_embed[i:i + batch_size]
        batch_embeddings = model.encode(batch_texts, show_progress_bar=False)
        all_embeddings.extend(batch_embeddings)
        
        if (i + batch_size) % (batch_size * 10) == 0:
            logger.info(f"  Processed {min(i + batch_size, len(texts_to_embed))}/{len(texts_to_embed)} nodes")
    
    # Store embeddings on graph and build return dictionary
    embeddings: Dict[str, np.ndarray] = {}
    
    for node_id, embedding in zip(node_ids, all_embeddings):
        embeddings[node_id] = embedding
        # Store embedding on graph node (convert to list for JSON serialization)
        graph.nodes[node_id]['embedding'] = embedding.tolist()
    
    logger.info(f"Generated {len(embeddings)} embeddings ({embedding_dim} dimensions)")
    
    return embeddings


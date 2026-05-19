
import logging

import networkx as nx

logger = logging.getLogger(__name__)


def generate_rag_embeddings(graph: nx.DiGraph) -> None:
    """Generate real L2-normalised BGE-small-en-v1.5 embeddings for graph nodes.

    Replaces the previous mock random-vector implementation.  Nodes whose
    ``node_type`` is CHUNK, TextChunk, Document, or TOPIC receive a 384-dim
    embedding computed from their ``text`` attribute (falling back to the node
    label / ID when text is absent).
    """
    try:
        from graphknows.retrieval.stores.embedder import embed
    except ImportError:
        # Fallback: use sentence-transformers directly when graphknows isn't installed
        from sentence_transformers import SentenceTransformer
        import numpy as np

        _model = SentenceTransformer("BAAI/bge-small-en-v1.5")

        def embed(texts: list[str]) -> "np.ndarray":  # type: ignore[misc]
            return _model.encode(texts, normalize_embeddings=True, show_progress_bar=False)

    logger.info("Generating BGE-small-en-v1.5 embeddings for graph nodes …")

    target_types = {"CHUNK", "TextChunk", "Document", "TOPIC"}
    nodes_to_embed: list[tuple[str, dict]] = [
        (nid, data)
        for nid, data in graph.nodes(data=True)
        if data.get("node_type") in target_types
    ]

    if not nodes_to_embed:
        logger.info("No embeddable nodes found.")
        return

    texts = [
        data.get("text") or data.get("label") or str(nid)
        for nid, data in nodes_to_embed
    ]

    embeddings = embed(texts)

    for (nid, data), emb in zip(nodes_to_embed, embeddings):
        data["embedding"] = emb.tolist()

    logger.info("Embedded %d nodes with BGE-small-en-v1.5.", len(nodes_to_embed))

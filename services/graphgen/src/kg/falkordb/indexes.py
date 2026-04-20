"""
FalkorDB index creation for Knowledge Graph.
"""

import logging
from typing import List
from falkordb import FalkorDB, Graph

logger = logging.getLogger(__name__)

def create_indexes(graph: Graph) -> List[str]:
    """
    Create standard indexes for efficient node lookups.
    
    Args:
        graph: FalkorDB Graph instance
        
    Returns:
        List of created index names
    """
    indexes = [
        "CREATE INDEX FOR (d:DAY) ON (d.id)",
        "CREATE INDEX FOR (s:SEGMENT) ON (s.id)",
        "CREATE INDEX FOR (e:EPISODE) ON (e.id)",
        "CREATE INDEX FOR (c:CHUNK) ON (c.id)",
        "CREATE INDEX FOR (e:ENTITY_CONCEPT) ON (e.id)",
        "CREATE INDEX FOR (st:SUBTOPIC) ON (st.id)",
        "CREATE INDEX FOR (t:TOPIC) ON (t.id)",
    ]
    
    created = []
    for index_query in indexes:
        try:
            graph.query(index_query)
            # Rough parsing to get index name/description
            created.append(index_query)
            logger.info(f"Created index: {index_query}")
        except Exception as e:
            error_msg = str(e).lower()
            if "already exists" in error_msg:
                logger.debug(f"Index already exists: {index_query}")
            else:
                logger.warning(f"Failed to create index: {e}")
    
    return created

def create_vector_indexes(graph: Graph, embedding_dim: int = 384) -> List[str]:
    """
    Create vector indexes for semantic search.
    Note: FalkorDB vector index syntax might vary.
    Using 'CALL db.idx.vector.createNodeIndex' style if supported, 
    or standard Cypher if FalkorDB supports it.
    
    For now, this is a placeholder or uses a common syntax.
    """
    # TODO: Verify FalkorDB vector index syntax. 
    # Current FalkorDB versions often support:
    # CALL db.idx.vector.createNodeIndex('Label', 'property', dimensions, metric)
    
    vector_indexes = [
        ('ENTITY_CONCEPT', 'embedding'),
        ('EPISODE', 'embedding'),
        # ('CHUNK', 'embedding'),     # Offloaded to Postgres
        # ('SUBTOPIC', 'embedding'),  # Offloaded to Postgres
        # ('TOPIC', 'embedding'),     # Offloaded to Postgres
    ]
    
    created = []
    for label, property_name in vector_indexes:
        try:
            # Create vector index using correct syntax for FalkorDB
            query = (
                f"CREATE VECTOR INDEX FOR (n:{label}) ON (n.{property_name}) "
                f"OPTIONS {{dimension:{embedding_dim}, similarityFunction:'cosine'}}"
            )
            graph.query(query)
            created.append(f"{label}.{property_name}")
            logger.info(f"Created vector index for {label}.{property_name}")
        except Exception as e:
            error_msg = str(e).lower()
            if "already exists" in error_msg:
                logger.debug(f"Vector index already exists: {label}.{property_name}")
            else:
                logger.warning(f"Failed to create vector index for {label}.{property_name}: {e}")
            
    return created


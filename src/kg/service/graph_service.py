"""
Iterative Graph Building Service.

Manages incremental document processing and graph updates.
"""

import logging
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Set, Tuple
from datetime import datetime
import networkx as nx

from ..config.loader import Config
from ..falkordb import KnowledgeGraphUploader

logger = logging.getLogger(__name__)


class ProcessingState:
    """Tracks what has been processed to enable incremental updates with segment-level granularity."""
    
    def __init__(self, state_file: Path):
        """
        Initialize processing state.
        
        Args:
            state_file: Path to JSON file storing processing state
        """
        self.state_file = state_file
        self.processed_documents: Set[str] = set()
        self.processed_segments: Set[str] = set()
        self.document_metadata: Dict[str, Dict[str, Any]] = {}  # Track segments per document
        self.last_update: Optional[str] = None
        self.document_count: int = 0
        self.segment_count: int = 0
        self.last_speech_limit: Optional[int] = None
        
        self.load()
    
    def load(self):
        """Load state from file if it exists."""
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r') as f:
                    data = json.load(f)
                    self.processed_documents = set(data.get('processed_documents', []))
                    self.processed_segments = set(data.get('processed_segments', []))
                    self.document_metadata = data.get('document_metadata', {})
                    self.last_update = data.get('last_update')
                    self.document_count = data.get('document_count', 0)
                    self.segment_count = data.get('segment_count', 0)
                    self.last_speech_limit = data.get('last_speech_limit')
                logger.info(f"Loaded processing state: {self.document_count} documents, {self.segment_count} segments (speech_limit: {self.last_speech_limit})")
            except Exception as e:
                logger.warning(f"Failed to load processing state: {e}, starting fresh")
    
    def save(self):
        """Save current state to file."""
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        data = {
            'processed_documents': list(self.processed_documents),
            'processed_segments': list(self.processed_segments),
            'document_metadata': self.document_metadata,
            'last_update': datetime.now().isoformat(),
            'document_count': self.document_count,
            'segment_count': self.segment_count,
            'last_speech_limit': self.last_speech_limit
        }
        with open(self.state_file, 'w') as f:
            json.dump(data, f, indent=2)
        logger.info(f"Saved processing state: {self.document_count} documents, {self.segment_count} segments")
    
    def mark_document_processed(self, document_id: str, segment_ids: List[str], speech_limit: int):
        """
        Mark a document as processed with its segments.
        
        Args:
            document_id: Document identifier
            segment_ids: List of segment IDs extracted from this document
            speech_limit: Speech limit used for this processing
        """
        if document_id not in self.processed_documents:
            self.processed_documents.add(document_id)
            self.document_count += 1
        
        # Store metadata about this document
        self.document_metadata[document_id] = {
            'segments': segment_ids,
            'segment_count': len(segment_ids),
            'speech_limit': speech_limit,
            'last_processed': datetime.now().isoformat()
        }
        
        # Update global speech limit
        self.last_speech_limit = speech_limit
    
    def mark_segment_processed(self, segment_id: str):
        """Mark a segment as processed."""
        if segment_id not in self.processed_segments:
            self.processed_segments.add(segment_id)
            self.segment_count += 1
    
    def needs_reprocessing(self, document_id: str, current_speech_limit: int) -> bool:
        """
        Check if a document needs reprocessing.
        
        Args:
            document_id: Document to check
            current_speech_limit: Current speech limit setting
            
        Returns:
            True if document should be reprocessed
        """
        # New document - needs processing
        if document_id not in self.processed_documents:
            return True
        
        # Check if speech limit has increased
        metadata = self.document_metadata.get(document_id, {})
        old_limit = metadata.get('speech_limit', 0)
        
        # If limit increased, we might have more segments to extract
        if current_speech_limit > old_limit:
            logger.info(f"Document {document_id} needs reprocessing: limit increased from {old_limit} to {current_speech_limit}")
            return True
        
        return False
    



class IterativeGraphBuilder:
    """
    Service for iterative graph building with incremental updates.
    
    Supports:
    - Processing new documents without full rebuild
    - State tracking to avoid reprocessing
    - Incremental graph updates using MERGE operations
    - Centrality recalculation on updated graph
    """
    
    def __init__(
        self,
        config: Config,
        state_file: Optional[Path] = None
    ):
        """
        Initialize the iterative graph builder.
        
        Args:
            config: Configuration object
            state_file: Optional path to state file (defaults to output dir)
        """
        self.config = config
        
        # Set up state tracking
        if state_file is None:
            state_file = Path(config.processing.output_dir) / "processing_state.json"
        self.state = ProcessingState(state_file)
        
        # Initialize FalkorDB uploader
        self.uploader = KnowledgeGraphUploader(
            host=config.falkordb.host,
            port=config.falkordb.port,
            username=getattr(config.falkordb, 'username', None),
            password=getattr(config.falkordb, 'password', None),
            database=config.falkordb.database,
            postgres_config=config.to_dict().get('postgres')
        )
        
        logger.info("Initialized IterativeGraphBuilder")
        logger.info(f"State: {len(self.state.processed_documents)} documents, {len(self.state.processed_segments)} segments processed")
    
    def get_new_documents(self, all_documents: List[str], current_speech_limit: int) -> List[str]:
        """
        Filter documents that need processing.
        
        Includes:
        - New documents never processed before
        - Documents that need reprocessing (e.g., speech limit increased)
        
        Args:
            all_documents: List of all available document IDs
            current_speech_limit: Current speech_limit configuration
            
        Returns:
            List of document IDs that need processing
        """
        docs_to_process = []
        
        for doc in all_documents:
            if self.state.needs_reprocessing(doc, current_speech_limit):
                docs_to_process.append(doc)
        
        logger.info(f"Found {len(docs_to_process)} document(s) to process out of {len(all_documents)} total")
        if len(docs_to_process) < len(all_documents):
            logger.info(f"Skipping {len(all_documents) - len(docs_to_process)} already-processed documents")
        
        return docs_to_process
    
    def merge_graph_incrementally(
        self,
        graph: nx.DiGraph,
        document_id: str,
        segment_ids: List[str],
        speech_limit: int
    ) -> Dict[str, Any]:
        """
        Merge a new document's graph into FalkorDB incrementally.
        
        Args:
            graph: NetworkX graph containing new nodes and relationships
            document_id: ID of the document being processed
            segment_ids: List of segment IDs from this document
            speech_limit: Speech limit used for processing
            
        Returns:
            Statistics about the merge operation
        """
        logger.info(f"🚀 Uploading graph for document {document_id} ({len(segment_ids)} segments) to FalkorDB...")
        
        # Connect to FalkorDB
        if not self.uploader.connect():
            raise RuntimeError("Failed to connect to FalkorDB")
        
        stats = {
            'document_id': document_id,
            'segment_count': len(segment_ids),
            'nodes_merged': 0,
            'relationships_merged': 0
        }
        
        try:
            # Prepare nodes for merging
            nodes = self.uploader._prepare_nodes(graph)
            
            # Merge nodes (MERGE instead of CREATE)
            merge_result = self.uploader.merge_nodes(nodes)
            stats['nodes_merged'] = merge_result.get('nodes_merged', 0)
            
            # Prepare and merge relationships
            edges = []
            for source, target, edge_data in graph.edges(data=True):
                # Determine relationship type (same logic as KnowledgeGraphUploader)
                rel_type = (
                    edge_data.get('edge_type') or
                    edge_data.get('label') or
                    edge_data.get('relation_type') or
                    edge_data.get('relationship') or
                    edge_data.get('type') or
                    'RELATED_TO'
                )
                
                # Filter out type keys from properties
                props = {k: v for k, v in edge_data.items() if k not in [
                    'edge_type', 'label', 'relation_type', 'relationship', 'type'
                ]}
                
                edge_dict = {
                    'source_id': str(source),
                    'target_id': str(target),
                    'type': rel_type,
                    'properties': props
                }
                edges.append(edge_dict)
            
            if edges:
                merge_result = self.uploader.merge_relationships(edges)
                stats['relationships_merged'] = merge_result.get('relationships_merged', 0)
            
            logger.info(f"✅ Merged {stats['nodes_merged']} nodes, {stats['relationships_merged']} relationships")
            
        finally:
            self.uploader.close()
        
        return stats
    
    def calculate_and_get_metrics(self) -> Dict[str, Any]:
        """
        Recalculate centrality and retrieve graph metrics.
        
        Returns:
            Dictionary containing node counts and average centrality measures.
        """
        logger.info("Recalculating centrality and gathering metrics...")
        
        if not self.uploader.connect():
            raise RuntimeError("Failed to connect to FalkorDB")
        
        try:
            from src.kg.falkordb.algorithms import FalkorDBAlgorithms
            algorithms = FalkorDBAlgorithms(self.uploader.graph_client)
            # Restrict metrics to ENTITY_CONCEPT nodes to avoid skewing stats with chunks/segments
            results = algorithms.get_graph_metrics(label='ENTITY_CONCEPT')
            logger.info("✅ Metrics calculation complete")
            return results
        finally:
            self.uploader.close()
    

    
    def ensure_schema(self, embedding_dim: int = 384) -> None:
        """
        Ensure database schema (indexes) is initialized.
        """
        logger.info("Verifying database schema and indexes...")
        
        if not self.uploader.connect():
            raise RuntimeError("Failed to connect to FalkorDB")
            
        try:
            from ..falkordb.indexes import create_indexes, create_vector_indexes
            
            # Standard indexes
            create_indexes(self.uploader.graph_client)
            
            # Vector indexes
            # Use provided dim or configured dim
            dim = embedding_dim
            if hasattr(self.config, 'embeddings') and hasattr(self.config.embeddings, 'dimension'):
                dim = self.config.embeddings.dimension
                
            create_vector_indexes(self.uploader.graph_client, embedding_dim=dim)
            logger.info("✅ Database schema valid")
            
        finally:
            self.uploader.close()

    def fetch_entity_graph(self) -> nx.Graph:
        """
        Fetch the entity graph from FalkorDB for community detection.
        
        Returns:
            NetworkX graph containing entities and their relationships.
        """
        logger.info("Fetching entity graph from FalkorDB...")
        if not self.uploader.connect():
            raise RuntimeError("Failed to connect to FalkorDB")
        
        try:
            # Match entities and their relationships
            # Removed CONTEXT as requested
            labels_clause = "n:ENTITY_CONCEPT OR n:PLACE OR n:ONTOLOGY_CLASS"
            
            # Fetch nodes with more property fallbacks
            query_nodes = f"""
            MATCH (n) WHERE {labels_clause} 
            RETURN 
                n.id as id, 
                coalesce(n.name, n.title, n.id) as name, 
                coalesce(n.node_type, n.type, 'ENTITY_CONCEPT') as node_type, 
                n.ontology_class as ontology_class
            """
            res_nodes = self.uploader.graph_client.query(query_nodes)
            
            g = nx.Graph() # Undirected for Leiden algorithm usually
            
            for record in res_nodes.result_set:
                node_id, name, node_type, ontology_class = record
                
                # Use node_id from database
                actual_id = node_id if node_id else "UNKNOWN"
                
                # Normalize node_type to uppercase
                if node_type:
                    node_type = node_type.upper()
                
                g.add_node(actual_id, id=actual_id, name=name, node_type=node_type, ontology_class=ontology_class)
            
            logger.info(f"Fetched {g.number_of_nodes()} nodes for community detection")
            
            # Fetch edges
            query_edges = f"""
            MATCH (s)-[r]->(t) 
            WHERE ({labels_clause.replace('n:', 's:')}) 
            AND ({labels_clause.replace('n:', 't:')})
            RETURN coalesce(s.id, 'S_UNKNOWN') as source_id, coalesce(t.id, 'T_UNKNOWN') as target_id, type(r) as type, coalesce(r.weight, 1.0) as weight
            """
            res_edges = self.uploader.graph_client.query(query_edges)
            
            for record in res_edges.result_set:
                source, target, rel_type, weight = record
                if g.has_node(source) and g.has_node(target):
                    g.add_edge(source, target, weight=float(weight), type=rel_type)
                
            logger.info(f"Fetched {g.number_of_edges()} relationships")
            
            return g
            
        finally:
            self.uploader.close()

    def update_communities(self, communities: Dict[str, int]) -> None:
        """
        Update community assignments in FalkorDB.
        
        Args:
            communities: Dictionary mapping node IDs to community IDs
        """
        logger.info(f"Updating community assignments for {len(communities)} nodes...")
        
        if not self.uploader.connect():
            raise RuntimeError("Failed to connect to FalkorDB")
            
        try:
            # Group by community ID to batch updates?
            # Or batch by chunks of nodes.
            # "UNWIND $batch as row MATCH (n:ENTITY_CONCEPT {id: row.id}) SET n.community = row.community"
            
            batch_size = 1000
            updates = [{'id': k, 'community': v} for k, v in communities.items()]
            
            for i in range(0, len(updates), batch_size):
                batch = updates[i:i+batch_size]
                query = """
                UNWIND $batch as row 
                MATCH (n:ENTITY_CONCEPT {id: row.id}) 
                SET n.community = row.community
                """
                self.uploader.graph_client.query(query, {'batch': batch})
                
            logger.info("✅ Community assignments updated")
            
        finally:
            self.uploader.close()

    def apply_merges_to_db(self, merges: List[Tuple[str, str, float]]) -> None:
        """
        Apply merge operations to the database.
        
        Args:
            merges: List of (source_id, target_id, score) tuples.
        """
        if not merges:
            return
            
        logger.info(f"Applying {len(merges)} merges to database...")
        
        if not self.uploader.connect():
             raise RuntimeError("Failed to connect to FalkorDB")
             
        try:
             count = 0
             for source, target, score in merges:
                 # 1. Fetch all edges connected to source
                 # We simply move them to target.
                 query_edges = f"MATCH (s {{id: '{source}'}})-[r]-(n) RETURN type(r) as type, startNode(r).id as start, endNode(r).id as end, properties(r) as props"
                 res = self.uploader.graph_client.query(query_edges)
                 
                 edges_to_create = []
                 if res and res.result_set:
                     for record in res.result_set:
                         rtype, start, end, props = record
                         
                         # Determine new direction
                         if start == source:
                             new_start, new_end = target, end
                         else:
                             new_start, new_end = start, target
                             
                         # Avoid self-loops if not intended
                         if new_start == new_end:
                             continue 
                             
                         edges_to_create.append({
                             'source_id': new_start,
                             'target_id': new_end,
                             'type': rtype,
                             'properties': props
                         })
                 
                 # 2. Create new edges
                 if edges_to_create:
                     self.uploader.merge_relationships(edges_to_create)
                     
                 # 3. Update target aliases (append source name/id)
                 # Using a direct query to update the array
                 query_alias = f"MATCH (s {{id: '{source}'}}), (t {{id: '{target}'}}) SET t.aliases = (CASE WHEN t.aliases IS NULL THEN [] ELSE t.aliases END) + [s.name, s.id]"
                 self.uploader.graph_client.query(query_alias)
                 
                 # 4. Delete source node
                 query_del = f"MATCH (s {{id: '{source}'}}) DETACH DELETE s"
                 self.uploader.graph_client.query(query_del)
                 count += 1
                 
             logger.info(f"✅ Successfully applied {count} merges.")
                 
        except Exception as e:
             logger.error(f"Failed to apply merges: {e}", exc_info=True)
        finally:
             self.uploader.close()

    def fetch_chunks_for_summarization(self) -> nx.DiGraph:
        """
        Fetch chunks and their relationships to entities for summarization.
        
        Returns:
            NetworkX directed graph with CHUNK nodes and HAS_ENTITY edges.
        """
        logger.info("Fetching chunks for summarization context...")
        if not self.uploader.connect():
            raise RuntimeError("Failed to connect to FalkorDB")
        
        g = nx.DiGraph()
        
        try:
            # 1. Fetch chunks with their text and order properties
            query_chunks = """
            MATCH (c:CHUNK) 
            RETURN c.id, c.text, c.speech_order, c.chunk_order, c.name
            """
            res_chunks = self.uploader.graph_client.query(query_chunks)
            
            for record in res_chunks.result_set:
                chunk_id, text, speech_order, chunk_order, name = record
                g.add_node(chunk_id, 
                          node_type='CHUNK', 
                          text=text, 
                          speech_order=speech_order, 
                          chunk_order=chunk_order,
                          name=name)
            
            logger.info(f"Fetched {g.number_of_nodes()} chunk nodes")
            
            # 2. Fetch HAS_ENTITY relationships between CHUNK and ENTITY_CONCEPT
            query_edges = """
            MATCH (c:CHUNK)-[r:HAS_ENTITY]->(e:ENTITY_CONCEPT)
            RETURN c.id, e.id
            """
            res_edges = self.uploader.graph_client.query(query_edges)
            
            edge_count = 0
            for record in res_edges.result_set:
                chunk_id, entity_id = record
                if g.has_node(chunk_id): # Only add if we have the chunk
                    g.add_edge(chunk_id, entity_id, label='HAS_ENTITY')
                    edge_count += 1
                
            logger.info(f"Fetched {edge_count} HAS_ENTITY relationships")
            
            return g
            
        finally:
            self.uploader.close()

    def reset_state(self):
        """Reset processing state (use with caution!)."""
        logger.warning("Resetting processing state!")
        self.state.processed_documents.clear()
        self.state.processed_segments.clear()
        self.state.document_count = 0
        self.state.segment_count = 0
        self.state.save()

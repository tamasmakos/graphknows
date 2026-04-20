"""
FalkorDB Knowledge Graph Uploader.

Uploads a NetworkX knowledge graph to FalkorDB.
"""

import logging
from typing import Dict, List, Any, Optional
import networkx as nx
from falkordb import FalkorDB

from .indexes import create_indexes, create_vector_indexes

logger = logging.getLogger(__name__)

def _escape_cypher_identifier(identifier: str) -> str:
    """
    Escape Cypher identifier (label or relationship type) with backticks if needed.
    
    In Cypher, identifiers with spaces or special characters must be escaped with backticks.
    """
    if not identifier:
        return identifier
    
    # Check if identifier contains spaces or special characters that need escaping
    # Also check if it starts with a digit (Cypher identifiers cannot start with a digit)
    needs_escaping = (
        ' ' in identifier or 
        '-' in identifier or 
        not identifier.replace('_', '').isalnum() or
        identifier[0].isdigit()
    )

    if needs_escaping:
        # Escape with backticks, and escape any backticks in the identifier itself
        escaped = identifier.replace('`', '``')
        return f"`{escaped}`"
    return identifier

def _validate_and_clean_properties(props: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate and clean properties to ensure they're compatible with FalkorDB.
    
    Returns a cleaned dictionary with only valid property types.
    """
    import json
    
    cleaned = {}
    for key, value in props.items():
        # Skip None values
        if value is None:
            continue
        
        # Handle basic types that FalkorDB supports
        if isinstance(value, (int, float, bool)):
            cleaned[key] = value
        elif isinstance(value, str):
            # Strings are fine, but skip empty strings to avoid issues
            if len(value) > 0:
                cleaned[key] = value
        elif isinstance(value, (list, tuple)):
            # Lists/tuples - check if it's an embedding (numeric list) or needs JSON serialization
            if len(value) == 0:
                continue
            # If all elements are numbers, it's likely an embedding - keep as list
            if all(isinstance(x, (int, float)) for x in value):
                cleaned[key] = list(value)
            else:
                # Mixed types or non-numeric - serialize to JSON string
                try:
                    cleaned[key] = json.dumps(value, ensure_ascii=False)
                except (TypeError, ValueError):
                    continue  # Skip if can't serialize
        elif isinstance(value, dict):
            # Dictionaries - serialize to JSON string
            try:
                cleaned[key] = json.dumps(value, ensure_ascii=False)
            except (TypeError, ValueError):
                continue  # Skip if can't serialize
        else:
            # Other types - convert to string
            try:
                str_value = str(value)
                if len(str_value) > 0:
                    cleaned[key] = str_value
            except:
                continue  # Skip if can't convert
    
    return cleaned

from .postgres_store import PostgresVectorStore

class KnowledgeGraphUploader:
    """
    Uploads a knowledge graph to FalkorDB.
    """
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        username: Optional[str] = None,
        password: Optional[str] = None,
        database: str = "kg",
        node_batch_size: int = 50,
        rel_batch_size: int = 100,
        postgres_config: Optional[Dict[str, Any]] = None
    ):
        self.host = host
        self.port = port
        self.username = username
        self.password = password
        self.database_name = database
        self.node_batch_size = node_batch_size
        self.rel_batch_size = rel_batch_size
        self.postgres_config = postgres_config
        self.driver = None
        self.graph_client = None
        self.embedding_dim = None
        
        # Initialize Postgres Store if config is present
        self.pg_store = None
        if self.postgres_config and self.postgres_config.get('enabled', False):
            try:
                self.pg_store = PostgresVectorStore(
                    host=self.postgres_config.get('host', 'localhost'),
                    port=self.postgres_config.get('port', 5432),
                    database=self.postgres_config.get('database', 'graphknows'),
                    user=self.postgres_config.get('user', 'postgres'),
                    password=self.postgres_config.get('password', 'password'),
                    table_name=self.postgres_config.get('table_name', 'hybrid_embeddings')
                )
                logger.info("Enabled Hybrid Storage with Postgres (pgvector)")
            except Exception as e:
                logger.error(f"Failed to initialize PostgresVectorStore: {e}")

    def connect(self) -> bool:
        """Establish connection to FalkorDB."""
        try:
            self.driver = FalkorDB(
                host=self.host, 
                port=self.port,
                username=self.username,
                password=self.password
            )
            self.graph_client = self.driver.select_graph(self.database_name)
            return True
        except Exception as e:
            logger.error(f"Failed to connect to FalkorDB: {e}")
            return False

    def close(self):
        """Close connection."""
        if self.pg_store:
            self.pg_store.close()
        # FalkorDB client doesn't always require explicit close if just using Redis connection pool,
        # but good to have the method.
        pass

    def clear_database(self):
        """Delete the graph."""
        try:
            # delete() deletes the graph key
            self.graph_client.delete()
            logger.info(f"Cleared graph '{self.database_name}'")
            # Should we clear Postgres too?
            # Config says: "In incremental mode: clean_database is controlled by global 'clean_start'"
            # If we clear graph, we should probably clear embeddings too to ensure consistency?
            # But the postgres table might be shared? 'table_name' in config.
            # Safe bet: If we wipe the graph, the pointers become invalid.
            # But leaving orphans in PG is messy.
            # Let's truncate if we are fully clearing.
            if self.pg_store:
                try:
                    with self.pg_store.conn.cursor() as cur:
                         # Truncate is fast but dangerous if shared.
                         # DELETE FROM table WHERE node_type ... ?
                         # For now, let's leave it. The user can recreate.
                         # Or better yet, we just ignore orphans.
                         pass
                except:
                    pass
        except Exception as e:
            logger.warning(f"Failed to clear graph: {e}")

    def _prepare_nodes(self, graph: nx.DiGraph) -> List[Dict]:

        """
        Prepare nodes for upload.
        Extracts properties and detects embedding dimension.
        """
        import json
        
        nodes_for_upload = []
        
        for node_id, data in graph.nodes(data=True):
            # Create a copy of properties
            props = data.copy()
            
            # Extract label
            label = props.get('type', 'Entity')
            # Also check for 'node_type' as fallback
            if label == 'Entity' and 'node_type' in props:
                label = props.get('node_type', 'Entity')
            
            # Sanitize keys (remove spaces, etc.)
            sanitized_props = {}
            
            # Always include id first
            sanitized_props['id'] = str(node_id)
            
            for k, v in props.items():
                # Skip None values
                if v is None:
                    continue
                
                # Sanitize key - replace spaces and hyphens with underscores
                clean_key = k.replace(" ", "_").replace("-", "_")
                
                # Skip empty strings
                if isinstance(v, str) and len(v) == 0:
                    continue
                
                # Handle embeddings - keep as list for vecf32 conversion
                # Handle embeddings - keep as list for vecf32 conversion
                if clean_key.endswith('embedding'):
                    if isinstance(v, (list, tuple)) and len(v) > 0:
                        # Hybrid Storage Offload
                        # Check if we should offload this embedding to Postgres
                        # We target CHUNK, TOPIC, SUBTOPIC for offloading as requested
                        if self.pg_store and clean_key == 'embedding' and label in ['CHUNK', 'TOPIC', 'SUBTOPIC']:
                            try:
                                # Start connection if needed (it handles it internally)
                                pg_id = self.pg_store.store_embedding(str(node_id), label, list(v))
                                sanitized_props['pg_embedding_id'] = pg_id
                                # Do NOT add 'embedding' to sanitized_props, effectively removing it from FalkorDB
                                continue
                            except Exception as e:
                                logger.warning(f"Failed to offload embedding for {node_id} to Postgres: {e}. Fallback to FalkorDB.")
                        
                        # Use first vector to determine dimension if not set
                        if self.embedding_dim is None and clean_key == 'embedding':
                            self.embedding_dim = len(v)
                            
                        # Ensure it's a proper list for FalkorDB
                        sanitized_props[clean_key] = list(v)
                    continue
                
                # Handle basic types
                if isinstance(v, (int, float, bool, str)):
                    sanitized_props[clean_key] = v
                # Handle lists and dicts - serialize to JSON string
                elif isinstance(v, (dict, list)):
                    try:
                        sanitized_props[clean_key] = json.dumps(v, ensure_ascii=False)
                    except (TypeError, ValueError):
                        sanitized_props[clean_key] = str(v)
                # Handle other types - convert to string
                else:
                    try:
                        sanitized_props[clean_key] = str(v)
                    except:
                        continue  # Skip if can't convert
            
            # Ensure we always have at least the id property
            if not sanitized_props:
                sanitized_props['id'] = str(node_id)
            
            # Debug check for missing critical properties on ENTITY_CONCEPT
            if label == 'ENTITY_CONCEPT':
                if 'name' not in sanitized_props:
                    logger.warning(f"Node {node_id} (ENTITY_CONCEPT) is missing 'name' property!")
                if 'ontology_class' not in sanitized_props:
                    logger.warning(f"Node {node_id} (ENTITY_CONCEPT) is missing 'ontology_class' property!")

            nodes_for_upload.append({
                'id': node_id,
                'label': label,
                'properties': sanitized_props
            })
            
        return nodes_for_upload

    def _upload_nodes(self, nodes: List[Dict]):
        """Upload nodes to FalkorDB using bulk upload."""
        if not self.graph_client:
            raise RuntimeError("Not connected to FalkorDB")
        
        # Group nodes by label
        nodes_by_label: Dict[str, List[Dict]] = {}
        for node in nodes:
            label = node['label']
            if label not in nodes_by_label:
                nodes_by_label[label] = []
            nodes_by_label[label].append(node)
        
        for label, label_nodes in nodes_by_label.items():
            logger.info(f"Uploading {len(label_nodes)} {label} nodes...")
            
            # Escape label for Cypher if it contains spaces or special characters
            escaped_label = _escape_cypher_identifier(label)
            
            for i in range(0, len(label_nodes), self.node_batch_size):
                batch = label_nodes[i:i + self.node_batch_size]
                
                # Validate and clean properties for each node in the batch
                batch_props = []
                for node in batch:
                    cleaned_props = _validate_and_clean_properties(node['properties'])
                    # Ensure we always have at least the id
                    if 'id' not in cleaned_props:
                        cleaned_props['id'] = str(node['id'])
                    if cleaned_props:  # Only add if not empty
                        batch_props.append(cleaned_props)
                
                # Skip empty batches
                if not batch_props:
                    logger.warning(f"Skipping empty batch for {label} nodes")
                    continue
                
                # Check if we need embedding conversion
                needs_embedding_conv = False
                if batch_props:
                    first_item = batch_props[0]
                    needs_embedding_conv = 'embedding' in first_item and isinstance(first_item.get('embedding'), (list, tuple))
                
                # Build query - ensure proper Cypher syntax
                # Use SET n += props instead of SET n = props to handle empty dicts better
                # Actually, SET n = props should work, but let's be explicit about properties
                if needs_embedding_conv:
                    # If we need to convert embeddings, set properties first, then convert
                    cypher = f"UNWIND $batch AS props CREATE (n:{escaped_label}) SET n = props SET n.embedding = vecf32(n.embedding)"
                else:
                    # Simple case - no embedding conversion needed
                    cypher = f"UNWIND $batch AS props CREATE (n:{escaped_label}) SET n = props"
                
                try:
                    self.graph_client.query(cypher, {'batch': batch_props})
                    logger.debug(f"Uploaded {len(batch)} {label} nodes (batch {i//self.node_batch_size + 1})")
                except Exception as e:
                    logger.error(f"Failed to upload {label} batch: {e}")
                    # Fallback to individual creation with better error handling
                    for node in batch:
                        try:
                            # Validate and clean properties
                            node_props = _validate_and_clean_properties(node['properties'])
                            # Ensure properties is not empty and has id
                            if not node_props:
                                logger.warning(f"Skipping node {node['id']} with empty properties after cleaning")
                                continue
                            if 'id' not in node_props:
                                node_props['id'] = str(node['id'])
                            
                            # Check for embeddings in this specific node
                            has_embedding = 'embedding' in node_props and isinstance(node_props.get('embedding'), (list, tuple))
                            
                            if has_embedding:
                                # For individual nodes with embeddings, convert inline
                                self.graph_client.query(
                                    f"CREATE (n:{escaped_label}) SET n = $props SET n.embedding = vecf32(n.embedding)",
                                    {'props': node_props}
                                )
                            else:
                                self.graph_client.query(
                                    f"CREATE (n:{escaped_label}) SET n = $props",
                                    {'props': node_props}
                                )
                        except Exception as ind_error:
                            logger.error(f"Failed to create node {node['id']}: {ind_error}")

    def upload(
        self,
        graph: nx.DiGraph,
        clean_database: bool = True,
        create_indexes_flag: bool = True
    ) -> Dict[str, Any]:
        """
        Upload the knowledge graph to FalkorDB.
        """
        logger.info(f"Starting knowledge graph upload to FalkorDB (Clean: {clean_database})...")
        
        if not self.connect():
            raise RuntimeError("Failed to connect to FalkorDB")
        
        try:
            if clean_database:
                self.clear_database()
            
            # Prepare nodes
            nodes = self._prepare_nodes(graph)
            
            if clean_database:
                self._upload_nodes(nodes)
            else:
                self.merge_nodes(nodes)
            
            # Create indexes (standard) BEFORE relationships to speed up matching?
            if clean_database: # Only needed on fresh DB or if we suspect they are missing
                create_indexes(self.graph_client)
            
            # Relationships
            # Separate preparation from upload to handle merge vs create
            edges = self._prepare_relationships(graph)
             
            if clean_database:
                self._upload_relationships(edges)
            else:
                self.merge_relationships(edges)
            
            # Create vector indexes if requested
            if create_indexes_flag and self.embedding_dim:
                # If merging, indexes might already exist, but create_vector_indexes checks usually
                create_vector_indexes(self.graph_client, self.embedding_dim)
            
            stats = {
                'nodes_uploaded': graph.number_of_nodes(),
                'relationships_uploaded': graph.number_of_edges(),
                'embedding_dim': self.embedding_dim,
                'database': self.database_name,
            }
            
            logger.info(f"Upload completed: {stats['nodes_uploaded']} nodes, {stats['relationships_uploaded']} relationships")
            return stats
            
        finally:
            self.close()

    def _prepare_relationships(self, graph: nx.DiGraph) -> List[Dict]:
        """Prepare relationships for upload."""
        edges_to_upload = []
        
        for u, v, data in graph.edges(data=True):
            props = data.copy()
            # Determine relationship type
            rel_type = (
                props.pop('label', None) or
                props.pop('relation_type', None) or
                props.pop('relationship', None) or
                props.pop('type', None) or
                'RELATED_TO'
            )
            
            # Serialize complex types
            for k, v_val in props.items():
                if isinstance(v_val, (dict, list)):
                    import json
                    try:
                        props[k] = json.dumps(v_val)
                    except:
                        props[k] = str(v_val)
            
            edges_to_upload.append({
                'source_id': u,
                'target_id': v,
                'type': rel_type,
                'properties': props
            })
        return edges_to_upload

    def _upload_relationships(self, edges_to_upload: List[Dict]):
        """Upload relationships to FalkorDB."""
        logger.info(f"Uploading {len(edges_to_upload)} relationships...")
        
        # Batch edges
        # Ideally we group by relationship type
        edges_by_type = {}
        for edge in edges_to_upload:
            t = edge['type']
            if t not in edges_by_type:
                edges_by_type[t] = []
            edges_by_type[t].append(edge)
            
        for rel_type, edges in edges_by_type.items():
            # Escape relationship type for Cypher if it contains spaces or special characters
            escaped_rel_type = _escape_cypher_identifier(rel_type)
            
            for i in range(0, len(edges), self.rel_batch_size):
                batch = edges[i:i + self.rel_batch_size]
                
                cypher = f"""
                UNWIND $batch AS rel
                MATCH (source) WHERE source.id = rel.source_id
                MATCH (target) WHERE target.id = rel.target_id
                CREATE (source)-[r:{escaped_rel_type}]->(target)
                SET r = rel.properties
                """
                
                try:
                    self.graph_client.query(cypher, {'batch': batch})
                except Exception as e:
                    logger.error(f"Failed to upload {rel_type} relationships batch: {e}")
    
    def merge_nodes(self, nodes: List[Dict]) -> Dict[str, Any]:
        """
        Merge (upsert) nodes into FalkorDB for incremental updates.
        Uses MERGE instead of CREATE to avoid duplicates.
        
        Args:
            nodes: List of node dictionaries with 'id', 'label', and 'properties'
            
        Returns:
            Statistics about nodes merged
        """
        logger.info(f"🚀 Uploading and Merging {len(nodes)} nodes into FalkorDB...")
        
        if not self.graph_client:
            if not self.connect():
                raise RuntimeError("Failed to connect to FalkorDB")
        
        # Group by label
        nodes_by_label: Dict[str, List[Dict]] = {}
        for node in nodes:
            label = node['label']
            if label not in nodes_by_label:
                nodes_by_label[label] = []
            nodes_by_label[label].append(node)
        
        total_merged = 0
        for label, label_nodes in nodes_by_label.items():
            escaped_label = _escape_cypher_identifier(label)
            
            for i in range(0, len(label_nodes), self.node_batch_size):
                batch = label_nodes[i:i + self.node_batch_size]
                batch_props = []
                
                for node in batch:
                    cleaned_props = _validate_and_clean_properties(node['properties'])
                    if 'id' not in cleaned_props:
                        cleaned_props['id'] = str(node['id'])
                    if cleaned_props:
                        batch_props.append(cleaned_props)
                
                if not batch_props:
                    continue
                
                # Use MERGE instead of CREATE
                needs_embedding_conv = batch_props and 'embedding' in batch_props[0]
                
                if needs_embedding_conv:
                    cypher = f"""
                    UNWIND $batch AS props
                    MERGE (n:{escaped_label} {{id: props.id}})
                    SET n = props
                    SET n.embedding = vecf32(n.embedding)
                    """
                else:
                    cypher = f"""
                    UNWIND $batch AS props
                    MERGE (n:{escaped_label} {{id: props.id}})
                    SET n = props
                    """
                
                try:
                    self.graph_client.query(cypher, {'batch': batch_props})
                    total_merged += len(batch)
                except Exception as e:
                    logger.error(f"Failed to merge {label} batch: {e}")
        
        logger.info(f"✅ Uploaded and Merged {total_merged} nodes")
        return {'nodes_merged': total_merged}
    
    def merge_relationships(self, edges: List[Dict]) -> Dict[str, Any]:
        """
        Merge (upsert) relationships into FalkorDB for incremental updates.
        
        Args:
            edges: List of edge dictionaries with 'source_id', 'target_id', 'type', 'properties'
            
        Returns:
            Statistics about relationships merged
        """
        logger.info(f"🚀 Uploading and Merging {len(edges)} relationships into FalkorDB...")
        
        if not self.graph_client:
            if not self.connect():
                raise RuntimeError("Failed to connect to FalkorDB")
        
        edges_by_type = {}
        for edge in edges:
            t = edge['type']
            if t not in edges_by_type:
                edges_by_type[t] = []
            edges_by_type[t].append(edge)
        
        logger.info(f"DEBUG: merge_relationships types: {list(edges_by_type.keys())}")
        
        total_merged = 0
        for rel_type, type_edges in edges_by_type.items():
            escaped_rel_type = _escape_cypher_identifier(rel_type)
            
            for i in range(0, len(type_edges), self.rel_batch_size):
                batch = type_edges[i:i + self.rel_batch_size]
                
                cypher = f"""
                UNWIND $batch AS rel
                MATCH (source) WHERE source.id = rel.source_id
                MATCH (target) WHERE target.id = rel.target_id
                MERGE (source)-[r:{escaped_rel_type}]->(target)
                SET r = rel.properties
                """
                
                try:
                    self.graph_client.query(cypher, {'batch': batch})
                    total_merged += len(batch)
                except Exception as e:
                    logger.error(f"Failed to merge {rel_type} relationships: {e}")
        
        logger.info(f"✅ Uploaded and Merged {total_merged} relationships")
        return {'relationships_merged': total_merged}
    




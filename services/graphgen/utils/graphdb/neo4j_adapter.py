
import logging
from typing import Dict, List, Any, Optional, Tuple
import networkx as nx
from neo4j import GraphDatabase, Driver
import json

logger = logging.getLogger(__name__)

def _escape_cypher_identifier(identifier: str) -> str:
    """
    Escape Cypher identifier (label or relationship type) with backticks if needed.
    """
    if not identifier:
        return identifier
    
    needs_escaping = (
        ' ' in identifier or 
        '-' in identifier or 
        not identifier.replace('_', '').isalnum() or
        identifier[0].isdigit()
    )

    if needs_escaping:
        escaped = identifier.replace('`', '``')
        return f"`{escaped}`"
    return identifier

def _clean_props_for_neo4j(props: Dict[str, Any]) -> Dict[str, Any]:
    """
    Clean properties for Neo4j.
    Neo4j supports primitives and lists of primitives.
    """
    cleaned = {}
    for key, value in props.items():
        if value is None:
            continue
            
        # Neo4j supports: bool, int, float, str, bytearray, datetime, date, time, localtime, duration, point
        # And lists of these.
        
        if isinstance(value, (bool, int, float, str)):
            cleaned[key] = value
        elif isinstance(value, (list, tuple)):
            if len(value) == 0:
                continue
            # Check if homogeneous
            first = value[0]
            if all(isinstance(x, type(first)) for x in value):
                if isinstance(first, (int, float, str)):
                    cleaned[key] = list(value)
                else:
                    # Complex list -> JSON
                    try:
                        cleaned[key] = json.dumps(value, ensure_ascii=False)
                    except (TypeError, ValueError) as exc:
                        logger.debug("Failed to JSON-encode %s: %s", key, exc)
                        cleaned[key] = str(value)
            else:
                # Mixed list -> JSON
                try:
                    cleaned[key] = json.dumps(value, ensure_ascii=False)
                except (TypeError, ValueError) as exc:
                    logger.debug("Failed to JSON-encode %s: %s", key, exc)
                    cleaned[key] = str(value)
        elif isinstance(value, dict):
            try:
                cleaned[key] = json.dumps(value, ensure_ascii=False)
            except (TypeError, ValueError) as exc:
                logger.debug("Failed to JSON-encode %s: %s", key, exc)
                cleaned[key] = str(value)
        else:
            cleaned[key] = str(value)
            
    return cleaned

class Neo4jGraphUploader:
    """
    Uploads a NetworkX knowledge graph to Neo4j.
    """
    
    def __init__(
        self,
        host: str = "neo4j",
        port: int = 7687,
        username: str = "neo4j",
        password: str = "password",
        database: str = "neo4j", # Default DB
        node_batch_size: int = 1000,
        rel_batch_size: int = 2000
    ):
        self.uri = f"neo4j://{host}:{port}"
        self.auth = (username, password)
        self.database = database
        self.node_batch_size = node_batch_size
        self.rel_batch_size = rel_batch_size
        self.driver: Optional[Driver] = None
        
    def connect(self) -> bool:
        try:
            self.driver = GraphDatabase.driver(self.uri, auth=self.auth)
            self.driver.verify_connectivity()
            return True
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j at {self.uri}: {e}")
            return False
            
    def close(self):
        if self.driver:
            self.driver.close()
            
    def clear_database(self):
        """Wipe the database."""
        logger.warning(f"Clearing Neo4j database '{self.database}'...")
        with self.driver.session(database=self.database) as session:
            session.run("MATCH (n) DETACH DELETE n")
            # Also drop indexes? Maybe better to keep them if we reuse schema, 
            # but usually clear_database implies full reset.
            # Dropping all constraints/indexes is safer for fresh start.
            # For now, just data.
            
    def _create_vector_index(self, label: str, property_key: str, dim: int):
        """Create a vector index on the given label and property."""
        index_name = f"vector_{label}_{property_key}"
        # Cypher for creating vector index
        # Syntax: CREATE VECTOR INDEX index_name IF NOT EXISTS FOR (n:Label) ON (n.embedding)
        # OPTIONS {indexConfig: {`vector.dimensions`: 1536, `vector.similarity_function`: 'cosine'}}
        
        cypher = f"""
        CREATE VECTOR INDEX {index_name} IF NOT EXISTS
        FOR (n:{_escape_cypher_identifier(label)})
        ON (n.{property_key})
        OPTIONS {{indexConfig: {{
            `vector.dimensions`: {dim},
            `vector.similarity_function`: 'cosine'
        }}}}
        """
        
        with self.driver.session(database=self.database) as session:
            try:
                session.run(cypher)
                logger.info(f"Created vector index '{index_name}' for {label}.{property_key} (dim={dim})")
            except Exception as e:
                logger.error(f"Failed to create vector index {index_name}: {e}")

    def _ensure_constraints(self, labels: List[str]):
        """Ensure uniqueness constraints on ID for all labels."""
        with self.driver.session(database=self.database) as session:
            for label in labels:
                escaped = _escape_cypher_identifier(label)
                constraint_name = f"constraint_{label}_id"
                cypher = f"""
                CREATE CONSTRAINT {constraint_name} IF NOT EXISTS
                FOR (n:{escaped}) REQUIRE n.id IS UNIQUE
                """
                try:
                    session.run(cypher)
                except Exception as e:
                    logger.warning(f"Could not create constraint for {label}: {e}")

    def upload(self, graph: nx.DiGraph, clean_database: bool = True, create_indexes_flag: bool = True) -> Dict[str, Any]:
        if not self.driver:
            if not self.connect():
                raise RuntimeError("Not connected to Neo4j")
                
        if clean_database:
            self.clear_database()
            
        nodes_by_label = {}
        # Collect labels and embedding info
        embedding_dims = {} # label -> dim
        
        # Prepare nodes
        for node_id, data in graph.nodes(data=True):
            props = data.copy()
            label = props.pop('type', 'Entity')
            # Fallback
            if label == 'Entity' and 'node_type' in props:
                label = props.get('node_type', 'Entity')
            props['id'] = str(node_id)
            
            # Clean
            cleaned = _clean_props_for_neo4j(props)
            
            # Check embedding
            if 'embedding' in cleaned:
                emb = cleaned['embedding']
                if isinstance(emb, list) and len(emb) > 0 and isinstance(emb[0], (int, float)):
                    embedding_dims[label] = len(emb)
            
            if label not in nodes_by_label:
                nodes_by_label[label] = []
            nodes_by_label[label].append(cleaned)
            
        # Create Constraints first (helps performance and consistency)
        self._ensure_constraints(nodes_by_label.keys())
            
        # Upload Nodes
        with self.driver.session(database=self.database) as session:
            for label, nodes in nodes_by_label.items():
                logger.info(f"Uploading {len(nodes)} {label} nodes to Neo4j...")
                escaped_label = _escape_cypher_identifier(label)
                
                # Batch
                for i in range(0, len(nodes), self.node_batch_size):
                    batch = nodes[i:i + self.node_batch_size]
                    cypher = f"""
                    UNWIND $batch AS props
                    MERGE (n:{escaped_label} {{id: props.id}})
                    SET n += props
                    """
                    session.run(cypher, batch=batch)
                    
        # Upload Relationships
        edges_to_upload = []
        for u, v, data in graph.edges(data=True):
            props = data.copy()
            rel_type = (
                props.pop('label', None) or
                props.pop('relation_type', None) or
                props.pop('relationship', None) or
                props.pop('type', None) or
                'RELATED_TO'
            )
            cleaned_props = _clean_props_for_neo4j(props)
            edges_to_upload.append({
                'source': str(u),
                'target': str(v),
                'type': rel_type,
                'props': cleaned_props
            })
            
        # Group edges
        edges_by_type = {}
        for e in edges_to_upload:
            t = e['type']
            if t not in edges_by_type:
                edges_by_type[t] = []
            edges_by_type[t].append(e)
            
        with self.driver.session(database=self.database) as session:
            for r_type, edges in edges_by_type.items():
                logger.info(f"Uploading {len(edges)} {r_type} relationships...")
                escaped_type = _escape_cypher_identifier(r_type)
                
                for i in range(0, len(edges), self.rel_batch_size):
                    batch = edges[i:i + self.rel_batch_size]
                    cypher = f"""
                    UNWIND $batch AS rel
                    MATCH (s {{id: rel.source}})
                    MATCH (t {{id: rel.target}})
                    MERGE (s)-[r:{escaped_type}]->(t)
                    SET r += rel.props
                    """
                    session.run(cypher, batch=batch)
                    
        # Create Vector Indexes
        if create_indexes_flag:
            for label, dim in embedding_dims.items():
                self._create_vector_index(label, 'embedding', dim)

        stats = {
            'nodes_uploaded': graph.number_of_nodes(),
            'relationships_uploaded': graph.number_of_edges(),
            'database': self.database,
            'backend': 'neo4j'
        }
        return stats

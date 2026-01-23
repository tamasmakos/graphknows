import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple, Union
import os


from falkordb import FalkorDB

from src.infrastructure.config import Config
from src.kg.falkordb.postgres_store import PostgresVectorStore

logger = logging.getLogger(__name__)


class GraphDB(ABC):
    @abstractmethod
    def query(self, cypher: str, params: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Execute a Cypher query and return a list of records (dicts)."""
        pass

    @abstractmethod
    def query_vector(
        self,
        index_name: str,
        embedding: List[float],
        k: int,
        min_score: float = 0.0,
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Execute a vector similarity search.

        Args:
            index_name: The name of the index (e.g. 'community_embeddings_idx')
            embedding: The query vector
            k: Number of nearest neighbors to return
            min_score: Minimum similarity score

        Returns:
            List of (node_data, score) tuples
        """
        pass

    @abstractmethod
    def close(self):
        """Close the connection."""
        pass


class FalkorDBDB(GraphDB):
    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        database: str = "kg",
        username: Optional[str] = None,
        password: Optional[str] = None,
        postgres_config: Optional[Dict[str, Any]] = None
    ):
        self.driver = FalkorDB(host=host, port=port, username=username, password=password)
        self.graph = self.driver.select_graph(database)
        
        self.pg_store = None
        if postgres_config and postgres_config.get('enabled', False):
            try:
                self.pg_store = PostgresVectorStore(
                    host=postgres_config.get('host', 'localhost'),
                    port=postgres_config.get('port', 5432),
                    database=postgres_config.get('database', 'graphknows'),
                    user=postgres_config.get('user', 'postgres'),
                    password=postgres_config.get('password', 'password'),
                    table_name=postgres_config.get('table_name', 'hybrid_embeddings')
                )
                logger.info("FalkorDBDB: Hybrid Search Enabled (Postgres)")
            except Exception as e:
                logger.error(f"FalkorDBDB: Failed to init Postgres: {e}")

    def query(self, cypher: str, params: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Execute a Cypher query and return a list of records (dicts)."""
        try:
            result = self.graph.query(cypher, params)
            if not result or not result.header:
                return []
            
            columns = []
            for h in result.header:
                if isinstance(h, (list, tuple)) and len(h) >= 2:
                    columns.append(h[1])
                else:
                    columns.append(str(h))

            output = []
            for row in result.result_set:
                record = {}
                for i, val in enumerate(row):
                    if i < len(columns):
                        record[columns[i]] = val
                output.append(record)
            return output
        except Exception as e:
            logger.error(f"Query failed: {e}")
            logger.debug(f"Query was: {cypher}")
            return []

    def query_vector(
        self,
        index_name: str,
        embedding: List[float],
        k: int,
        min_score: float = 0.0,
    ) -> List[Tuple[Dict[str, Any], float]]:
        # Check if this index corresponds to an offloaded type
        # Retrieval passes "TOPIC", "SUBTOPIC", "ENTITY_CONCEPT" etc.
        # User requested CHUNK and TOPIC to be offloaded.
        offloaded_types = ['CHUNK', 'TOPIC', 'SUBTOPIC']
        
        if self.pg_store and index_name in offloaded_types:
            try:
                # Hybrid Search
                # 1. Search Postgres for IDs
                pg_results = self.pg_store.search_similar(embedding, node_type=index_name, k=k, min_score=min_score)
                if not pg_results:
                    return []
                
                pg_ids = [r[0] for r in pg_results]
                score_map = {r[0]: r[1] for r in pg_results}
                
                # 2. Fetch Nodes from FalkorDB using pg_embedding_id
                cypher = f"""
                MATCH (n:{index_name})
                WHERE n.pg_embedding_id IN $pg_ids
                RETURN n
                """
                graph_results = self.query(cypher, {'pg_ids': pg_ids})
                
                parsed_results = []
                for record in graph_results:
                     # 'n' is the node object or dict
                     # In query(), we return dicts where keys are columns. 
                     # 'n' column will be the node.
                     node = record.get('n', record) 
                     
                     # Extract properties
                     node_data = {}
                     pg_id_val = None
                     
                     if hasattr(node, "properties"):
                         node_data = dict(node.properties)
                     elif isinstance(node, dict):
                         node_data = node.get('properties', node) # fallback
                     
                     pg_id_val = node_data.get('pg_embedding_id')
                     
                     # Get score from map
                     score = 0.0
                     if pg_id_val is not None:
                         # pg_id might be int or string depending on roundtrip
                         try:
                             score = score_map.get(int(pg_id_val), 0.0)
                         except:
                             score = 0.0
                             
                     parsed_results.append((node_data, score))
                
                # Sort by score descending
                parsed_results.sort(key=lambda x: x[1], reverse=True)
                return parsed_results
                
            except Exception as e:
                logger.error(f"Hybrid vector search failed for {index_name}: {e}")
                return []
        
        # Fallback to standard FalkorDB vector search (e.g. for ENTITY_CONCEPT)
        cypher = f"CALL db.idx.vector.queryNodes('{index_name}', 'embedding', {k}, vecf32($vec)) YIELD node, score"
        
        if min_score > 0:
            cypher += f" WHERE score >= {min_score}"
            
        cypher += " RETURN node, score"
        
        results = self.query(cypher, {"vec": embedding})
        
        parsed_results = []
        for row in results:
            node = row.get("node")
            score = row.get("score")
            
            node_data = {}
            if hasattr(node, "properties"):
                node_data = dict(node.properties)
            elif isinstance(node, dict):
                node_data = node
            
            parsed_results.append((node_data, float(score)))
            
        return parsed_results

    def close(self):
        pass


def get_database_client(config: Config, db_type: str = "falkordb") -> GraphDB:
    if db_type.lower() == "falkordb":
        return FalkorDBDB(
            host=config.falkordb.host,
            port=config.falkordb.port,
            database=config.falkordb.database,
            username=config.falkordb.username,
            password=config.falkordb.password,
            postgres_config=config.to_dict().get('postgres')
        )
    else:
        raise ValueError(f"Unsupported database type: {db_type}")




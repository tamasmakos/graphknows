import logging
import numpy as np
import psycopg2
import os
from pgvector.psycopg2 import register_vector
from typing import List, Tuple, Optional

logger = logging.getLogger(__name__)

class PostgresVectorStore:
    def __init__(
        self,
        host: str = "localhost",
        port: int = 5432,
        database: str = "graphknows",
        user: str = "postgres",
        password: Optional[str] = None,
        table_name: str = "hybrid_embeddings",
        embedding_dim: int = 384
    ):
        self.host = host
        self.port = port
        self.database = database
        self.user = user
        self.password = password or os.getenv("POSTGRES_PASSWORD", "password")
        self.table_name = table_name
        self.embedding_dim = embedding_dim
        self.conn = None
        
    def _connect(self):
        """Establish connection to Postgres and register vector extension."""
        if self.conn and not self.conn.closed:
            return

        try:
            self.conn = psycopg2.connect(
                host=self.host,
                port=self.port,
                dbname=self.database,
                user=self.user,
                password=self.password
            )
            self.conn.autocommit = True
            
            # Ensure extension exists
            with self.conn.cursor() as cur:
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                
            register_vector(self.conn)
            
        except Exception as e:
            logger.error(f"Failed to connect to Postgres: {e}")
            if self.conn:
                try:
                    self.conn.close()
                except Exception:
                    pass
                self.conn = None
            raise

    def search_similar(
        self, 
        query_embedding: List[float], 
        node_type: Optional[str] = None, 
        k: int = 5,
        min_score: float = 0.0
    ) -> List[Tuple[int, float]]:
        """
        Search for similar vectors.
        Returns list of (pg_id, relevance_score).
        """
        self._connect()
        
        with self.conn.cursor() as cur:
            query = np.array(query_embedding)
            
            if node_type:
                sql = f"""
                    SELECT id, embedding <=> %s AS distance 
                    FROM {self.table_name} 
                    WHERE node_type = %s
                    ORDER BY distance ASC
                    LIMIT %s;
                """
                cur.execute(sql, (query, node_type, k))
            else:
                sql = f"""
                    SELECT id, embedding <=> %s AS distance 
                    FROM {self.table_name} 
                    ORDER BY distance ASC
                    LIMIT %s;
                """
                cur.execute(sql, (query, k))
            
            results = []
            for row in cur.fetchall():
                pg_id, distance = row
                similarity = 1.0 - float(distance)
                
                if similarity >= min_score:
                    results.append((pg_id, similarity))
                    
            return results

    def get_stats(self) -> dict:
        """Get statistics about the vector store."""
        self._connect()
        try:
            with self.conn.cursor() as cur:
                # Get total rows
                cur.execute(f"SELECT COUNT(*) FROM {self.table_name}")
                count = cur.fetchone()[0]
                
                # Get table size
                cur.execute(f"SELECT pg_size_pretty(pg_total_relation_size('{self.table_name}'))")
                size = cur.fetchone()[0]
                
                return {
                    "count": count,
                    "size": size,
                    "table": self.table_name
                }
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {"error": str(e)}

    def close(self):
        if self.conn:
            self.conn.close()
            self.conn = None

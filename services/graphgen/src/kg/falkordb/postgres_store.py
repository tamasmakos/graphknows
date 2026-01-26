
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
            # DEBUG: Print connection details (UNMASKED for debugging)
            logger.info(f"Connecting to Postgres: host={self.host}, port={self.port}, db={self.database}, user={self.user}")
            
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
            
            # Register pgvector type
            register_vector(self.conn)
            
            # Ensure table exists
            self._ensure_table()
            
        except Exception as e:
            logger.error(f"Failed to connect to Postgres: {e}")
            if self.conn:
                try:
                    self.conn.close()
                except Exception:
                    pass
                self.conn = None
            raise

    def _ensure_table(self):
        """Create the embeddings table if it doesn't exist."""
        with self.conn.cursor() as cur:
            # Create table
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.table_name} (
                    id SERIAL PRIMARY KEY,
                    node_id TEXT NOT NULL,
                    node_type TEXT NOT NULL,
                    embedding vector({self.embedding_dim})
                );
            """)
            
            # Create HNSW index for performance
            # Check if index exists first to avoid errors (or just use IF NOT EXISTS if PG supports it for indexes)
            # PG 9.5+ supports IF NOT EXISTS for indexes
            
            index_name = f"{self.table_name}_embedding_idx"
            cur.execute(f"""
                CREATE INDEX IF NOT EXISTS {index_name} 
                ON {self.table_name} 
                USING ivfflat (embedding vector_cosine_ops)
                WITH (lists = 100);
            """)
            
            # Index on node_id for lookups
            node_id_idx = f"{self.table_name}_node_id_idx"
            cur.execute(f"""
                CREATE INDEX IF NOT EXISTS {node_id_idx} ON {self.table_name} (node_id);
            """)

    def store_embedding(self, node_id: str, node_type: str, embedding: List[float]) -> int:
        """
        Store an embedding and return its Postgres ID.
        If an entry exists for this node_id, update it (or insert new).
        Actually, let's keep it simple: Insert or Update?
        To keep a 1:1 mapping for node pointers, we probably want to update if exists.
        However, the pointer logic relies on the SERIAL ID. If we update, the ID remains?
        Let's assume we insert fresh or update.
        
        Strategy:
        1. Check if node_id exists.
        2. If so, update embedding and return id.
        3. If not, insert and return id.
        """
        self._connect()
        
        with self.conn.cursor() as cur:
            # Upsert logic
            # We need to know the ID to return it.
            
            # Try update first (returning id)
            cur.execute(f"""
                UPDATE {self.table_name} 
                SET embedding = %s, node_type = %s 
                WHERE node_id = %s
                RETURNING id;
            """, (np.array(embedding), node_type, node_id))
            
            row = cur.fetchone()
            if row:
                return row[0]
            
            # Insert if not updated
            cur.execute(f"""
                INSERT INTO {self.table_name} (node_id, node_type, embedding) 
                VALUES (%s, %s, %s) 
                RETURNING id;
            """, (node_id, node_type, np.array(embedding)))
            
            row = cur.fetchone()
            if row:
                return row[0]
            
            raise RuntimeError("Failed to store embedding")

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
        Note: relevance_score = 1 - cosine_distance for 'vector_cosine_ops' if we treat distance as [0,2]?
        Actually pgvector <-> operator returns cosine distance (0 to 2).
        Score = 1 - distance/2 or just keep distance.
        Usually retrieval wants a score where higher is better.
        Cosine similarity = 1 - cosine distance.
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
                # Convert distance to similarity score
                # cosine_distance is 1 - cosine_similarity
                # So similarity = 1 - distance
                similarity = 1.0 - float(distance)
                
                if similarity >= min_score:
                    results.append((pg_id, similarity))
                    
            return results
            
    def get_embedding_id(self, node_id: str) -> Optional[int]:
        """Get the existing postgres ID for a node_id if it exists."""
        self._connect()
        with self.conn.cursor() as cur:
            cur.execute(f"SELECT id FROM {self.table_name} WHERE node_id = %s", (node_id,))
            row = cur.fetchone()
            return row[0] if row else None

    def close(self):
        if self.conn:
            self.conn.close()
            self.conn = None

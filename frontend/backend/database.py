import logging
import os
from typing import Optional, List, Dict, Any, Tuple
from falkordb import FalkorDB
import psycopg2

logger = logging.getLogger(__name__)

class GraphDB:
    def query(self, cypher: str, params: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        pass

    def get_vector_stats(self) -> Dict[str, Any]:
        pass
    
    def get_graph_stats(self) -> Dict[str, Any]:
        pass

    def close(self):
        pass

class FalkorDBDB(GraphDB):
    def __init__(self, host: str = "localhost", port: int = 6379, database: str = "kg"):
        self.host = host
        self.port = port
        self.database_name = database
        self.driver = FalkorDB(host=host, port=port)
        self.graph = self.driver.select_graph(database)
        
        # Postgres Config
        self.pg_enabled = os.getenv("POSTGRES_ENABLED", "false").lower() == "true"
        self.pg_host = os.getenv("POSTGRES_HOST", "localhost")
        self.pg_port = os.getenv("POSTGRES_PORT", 5432)
        self.pg_db = os.getenv("POSTGRES_DB", "graphknows")
        self.pg_user = os.getenv("POSTGRES_USER", "postgres")
        self.pg_password = os.getenv("POSTGRES_PASSWORD", "password")
        self.pg_table = os.getenv("POSTGRES_TABLE", "hybrid_embeddings")

    def query(self, cypher: str, params: Dict[str, Any] = None) -> List[Dict[str, Any]]:
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
            return []

    def get_vector_stats(self) -> Dict[str, Any]:
        # FalkorDB Stats
        falkor_stats = {
            "nodes": 0,
            "edges": 0,
            "bytes": 0,
            "bytes_fmt": "0 B"
        }
        try:
            # Nodes
            res = self.query("MATCH (n) RETURN count(n) as count")
            if res: falkor_stats["nodes"] = res[0]['count']
            
            # Edges
            res = self.query("MATCH ()-[r]->() RETURN count(r) as count")
            if res: falkor_stats["edges"] = res[0]['count']
            
            # Memory Usage (Requires Redis command access)
            # FalkorDB objects don't expose execute_command directly usually, need the underlying redis connection
            # But the 'driver' object in falkordb-py IS a client that might support it.
            # If not, we skip.
            try:
                # Assuming driver delegates to redis
                size = self.driver.execute_command("MEMORY USAGE", self.database_name)
                if size:
                    falkor_stats["bytes"] = int(size)
                    falkor_stats["bytes_fmt"] = self._sizeof_fmt(int(size))
            except Exception as e:
                logger.warning(f"Could not get FalkorDB memory usage: {e}")
                
        except Exception as e:
            logger.error(f"Failed to get FalkorDB stats: {e}")

        # Postgres Stats
        pg_stats = {
            "rows": 0,
            "size": "0 B",
            "enabled": False
        }
        
        if self.pg_enabled:
            pg_stats["enabled"] = True
            try:
                conn = psycopg2.connect(
                    host=self.pg_host,
                    port=self.pg_port,
                    dbname=self.pg_db,
                    user=self.pg_user,
                    password=self.pg_password
                )
                with conn.cursor() as cur:
                    # Check if table exists
                    cur.execute(f"SELECT to_regclass('{self.pg_table}');")
                    if cur.fetchone()[0]:
                         cur.execute(f"SELECT COUNT(*) FROM {self.pg_table}")
                         pg_stats["rows"] = cur.fetchone()[0]
                         
                         cur.execute(f"SELECT pg_size_pretty(pg_total_relation_size('{self.pg_table}'))")
                         pg_stats["size"] = cur.fetchone()[0]
                    else:
                        pg_stats["size"] = "Table Not Found"
                conn.close()
            except Exception as e:
                logger.error(f"Failed to get PG stats: {e}")
                pg_stats["error"] = str(e)
        
        return {
            "falkordb": falkor_stats,
            "pgvector": pg_stats
        }

    def _sizeof_fmt(self, num, suffix="B"):
        for unit in ["", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi"]:
            if abs(num) < 1024.0:
                return f"{num:3.1f}{unit}{suffix}"
            num /= 1024.0
        return f"{num:.1f}Yi{suffix}"

    def close(self):
        pass

_db_instance: Optional[GraphDB] = None

def get_db() -> GraphDB:
    global _db_instance
    if _db_instance is None:
        host = os.getenv("FALKORDB_HOST", "localhost")
        port = int(os.getenv("FALKORDB_PORT", 6379))
        
        logger.info(f"Connecting to FalkorDB at {host}:{port}")
        
        _db_instance = FalkorDBDB(host=host, port=port, database="kg")
        
        try:
            _db_instance.query("RETURN 1")
        except Exception:
            logger.warning(f"Failed to connect to {host}, trying localhost...")
            _db_instance = FalkorDBDB(host="localhost", port=port, database="kg")
            
    return _db_instance

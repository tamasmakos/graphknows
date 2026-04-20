import logging
import os
from typing import Optional, List, Dict, Any, Tuple
from falkordb import FalkorDB

logger = logging.getLogger(__name__)

class GraphDB:
    def query(self, cypher: str, params: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        pass

    def get_vector_stats(self) -> Dict[str, Any]:
        pass

    def close(self):
        pass

class FalkorDBDB(GraphDB):
    def __init__(self, host: str = "localhost", port: int = 6379, database: str = "kg"):
        self.driver = FalkorDB(host=host, port=port)
        self.graph = self.driver.select_graph(database)

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
        try:
            # Proxy vector stats with node count for now
            res = self.query("MATCH (n) RETURN count(n) as count")
            count = res[0]['count'] if res else 0
            return {
                "row_count": count,
                "table_size": "N/A (FalkorDB)"
            }
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {"row_count": 0, "table_size": "Error"}

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

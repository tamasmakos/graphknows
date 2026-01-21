import sys
import os
import logging
from typing import Optional

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from src.app.infrastructure.graph_db import FalkorDBDB, GraphDB

logger = logging.getLogger(__name__)

_db_instance: Optional[GraphDB] = None

def get_db() -> GraphDB:
    global _db_instance
    if _db_instance is None:
        host = os.getenv("FALKORDB_HOST", "host.docker.internal")
        port = int(os.getenv("FALKORDB_PORT", 6379))
        
        logger.info(f"Connecting to FalkorDB at {host}:{port}")
        
        # FalkorDBDB wraps FalkorDB client
        _db_instance = FalkorDBDB(host=host, port=port, database="kg")
        
        # Test connection by running a simple query
        try:
            _db_instance.query("RETURN 1")
        except Exception:
            logger.warning(f"Failed to connect to {host}, trying localhost...")
            _db_instance = FalkorDBDB(host="localhost", port=port, database="kg")
            
    return _db_instance
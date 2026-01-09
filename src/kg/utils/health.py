import logging
import socket
import time
from typing import Dict, Any, Optional

from falkordb import FalkorDB
import psycopg2

logger = logging.getLogger(__name__)

def check_falkordb(host: str, port: int, timeout: int = 5) -> bool:
    """Check if FalkorDB is reachable and responsive."""
    logger.info(f"Checking FalkorDB at {host}:{port}...")
    try:
        # First check if port is open
        with socket.create_connection((host, port), timeout=timeout):
            pass
        
        # Then try a simple command
        db = FalkorDB(host=host, port=port)
        db.connection.ping()
        logger.info("✅ FalkorDB is UP")
        return True
    except Exception as e:
        logger.error(f"❌ FalkorDB is DOWN or unreachable: {e}")
        return False

def check_postgres(host: str, port: int, user: str, password: str, dbname: str, timeout: int = 5) -> bool:
    """Check if PostgreSQL is reachable and responsive."""
    logger.info(f"Checking PostgreSQL at {host}:{port}...")
    conn = None
    try:
        conn = psycopg2.connect(
            host=host,
            port=port,
            user=user,
            password=password,
            dbname=dbname,
            connect_timeout=timeout
        )
        logger.info("✅ PostgreSQL is UP")
        return True
    except Exception as e:
        logger.error(f"❌ PostgreSQL is DOWN or unreachable: {e}")
        return False
    finally:
        if conn:
            conn.close()

def verify_all_services(config: Any) -> bool:
    """
    Verify all required services based on config.
    Returns True if all enabled services are healthy, False otherwise.
    """
    # Convert Config object to dict if needed
    if hasattr(config, 'to_dict'):
        config_data = config.to_dict()
    else:
        config_data = config

    all_healthy = True
    
    # 1. FalkorDB (Always required by the pipeline)
    falkor_cfg = config_data.get('falkordb', {})
    if not check_falkordb(
        falkor_cfg.get('host', 'localhost'),
        falkor_cfg.get('port', 6379)
    ):
        all_healthy = False
        
    # 2. Postgres (If enabled)
    pg_cfg = config_data.get('postgres', {})
    if pg_cfg.get('enabled', False):
        if not check_postgres(
            host=pg_cfg.get('host', 'localhost'),
            port=pg_cfg.get('port', 5432),
            user=pg_cfg.get('user', 'postgres'),
            password=pg_cfg.get('password', 'password'),
            dbname=pg_cfg.get('database', 'graphknows')
        ):
            all_healthy = False
            
    return all_healthy

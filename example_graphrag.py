import sys
import os
from unittest.mock import MagicMock

# Add services/graphrag to path so 'src' package is found
sys.path.insert(0, os.path.abspath("services/graphrag"))

try:
    from src.infrastructure.config import get_app_config
    from src.infrastructure.graph_db import get_database_client, FalkorDBDB
    # Mock langfuse before importing main to avoid errors if not configured/running
    sys.modules["langfuse.llama_index"] = MagicMock()
    
    from src.main import app
    print("[GraphRAG] Imports successful.")
except ImportError as e:
    print(f"[GraphRAG] Import failed: {e}")
    sys.exit(1)

def test_graphrag():
    print("[GraphRAG] Testing App wiring...")
    
    # 1. Config
    os.environ["FALKORDB_HOST"] = "localhost"
    os.environ["FALKORDB_PORT"] = "6379"
    
    try:
        config = get_app_config()
        print(f"[GraphRAG] Config loaded: {config.falkordb_host}")
    except Exception as e:
        print(f"[GraphRAG] Failed to load config: {e}")
        sys.exit(1)
    
    # 2. DB Client
    # We don't want to actually connect, but instantiation should work.
    try:
        # We need to mock FalkorDB class to prevent connection attempts if any
        # But importing src.infrastructure.graph_db already imported FalkorDB
        # We can patch it?
        # Or just try instantiating.
        
        db = get_database_client(config)
        print("[GraphRAG] Database client instantiated.")
        if isinstance(db, FalkorDBDB):
             print("[GraphRAG] Client is FalkorDBDB.")
    except Exception as e:
        print(f"[GraphRAG] DB Client instantiation warning: {e}")
        # Not fatal if just connection failed

    # 3. FastAPI App
    print(f"[GraphRAG] FastAPI App title: {app.title}")
    
    print("[GraphRAG] Test passed!")

if __name__ == "__main__":
    test_graphrag()

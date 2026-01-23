import sys
import os
import asyncio
from unittest.mock import MagicMock

# Add services/graphgen/src to path so we can import modules as if we are inside src
# But wait, src/main.py uses "from .kg..."
# Ideally we treat 'services/graphgen/src' as a package root?
# If I add 'services/graphgen/src' to path, I can import 'kg'.
# But main.py is a module inside that root.

sys.path.insert(0, os.path.abspath("services/graphgen/src"))

try:
    from kg.config.settings import PipelineSettings
    from kg.falkordb.uploader import KnowledgeGraphUploader
    from kg.pipeline.core import KnowledgePipeline
    print("[GraphGen] Imports successful.")
except ImportError as e:
    print(f"[GraphGen] Import failed: {e}")
    sys.exit(1)

async def test_graphgen():
    print("[GraphGen] Testing KnowledgePipeline wiring...")
    
    # 1. Mock Settings (Env vars)
    os.environ["FALKORDB_HOST"] = "localhost"
    os.environ["FALKORDB_PORT"] = "6379"
    os.environ["OPENAI_API_KEY"] = "sk-test"
    
    try:
        settings = PipelineSettings()
        print(f"[GraphGen] Settings loaded: {settings.falkordb_host}:{settings.falkordb_port}")
    except Exception as e:
        print(f"[GraphGen] Failed to load settings: {e}")
        sys.exit(1)

    # 2. Mock Uploader
    try:
        uploader = KnowledgeGraphUploader(
            host=settings.falkordb_host,
            port=settings.falkordb_port
        )
        uploader.connect = MagicMock(return_value=True)
        uploader.close = MagicMock()
        print("[GraphGen] Uploader instantiated and mocked.")
    except Exception as e:
         print(f"[GraphGen] Failed to instantiate Uploader: {e}")
         sys.exit(1)

    # 3. Instantiate Pipeline
    try:
        pipeline = KnowledgePipeline(
            settings=settings,
            uploader=uploader
        )
        print("[GraphGen] KnowledgePipeline instantiated.")
    except Exception as e:
        print(f"[GraphGen] Failed to instantiate Pipeline: {e}")
        sys.exit(1)

    # 4. Verify IoC
    if hasattr(pipeline, 'uploader') and pipeline.uploader is uploader:
        print("[GraphGen] Dependency Injection verified.")
    else:
        print("[GraphGen] Dependency Injection FAILED.")
        sys.exit(1)

    print("[GraphGen] Test passed!")

if __name__ == "__main__":
    asyncio.run(test_graphgen())

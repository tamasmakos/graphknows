import sys
import os
import asyncio
import shutil
import logging
import dotenv
dotenv.load_dotenv()

# Add services/graphgen/src to path
sys.path.insert(0, os.path.abspath("services/graphgen/src"))


# Configure Logging to show pipeline steps
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

try:
    from kg.config.settings import PipelineSettings
    from kg.falkordb.uploader import KnowledgeGraphUploader
    from kg.pipeline.core import KnowledgePipeline
    from kg.graph.extractors import get_extractor
    print("[GraphGen] Imports successful.")
except ImportError as e:
    print(f"[GraphGen] Import failed: {e}")
    sys.exit(1)

async def test_graphgen():
    print("[GraphGen] Setting up test environment...")
    
    # 1. Setup Input Data
    input_dir = os.path.abspath("input")
    if not os.path.exists(input_dir):
        print(f"[GraphGen] Error: Input directory '{input_dir}' does not exist.")
        sys.exit(1)
    
    print(f"[GraphGen] Reading files from: {input_dir}")

    
    # Important: Set input_dir to our test dir
    os.environ["INPUT_DIR"] = os.path.abspath(input_dir)
    os.environ["OUTPUT_DIR"] = os.path.abspath("output")
    
    try:
        settings = PipelineSettings()
        # Override input_dir explicitly just in case env var didn't pick up
        settings.input_dir = os.path.abspath(input_dir)
        settings.output_dir = os.path.abspath("output")
        print(f"[GraphGen] Settings loaded. Input Dir: {settings.input_dir}")
        print(f"[GraphGen] Settings loaded. Output Dir: {settings.output_dir}")
    except Exception as e:
        print(f"[GraphGen] Failed to load settings: {e}")
        sys.exit(1)

    # 3. Real Uploader
    try:
        uploader = KnowledgeGraphUploader(
            host=settings.falkordb_host,
            port=settings.falkordb_port
        )
        print("[GraphGen] Real Uploader instantiated.")
    except Exception as e:
         print(f"[GraphGen] Failed to instantiate Uploader: {e}")
         sys.exit(1)
         
    # 4. Real Extractor
    try:
        # Pass settings as dict (Pydantic v2 uses model_dump, v1 uses dict)
        config_dict = settings.model_dump() if hasattr(settings, 'model_dump') else settings.dict()
        extractor = get_extractor(config_dict)
        print("[GraphGen] Real Extractor instantiated.")
    except Exception as e:
        print(f"[GraphGen] Failed to instantiate Extractor: {e}")
        sys.exit(1)

    # 5. Instantiate Pipeline
    try:
        pipeline = KnowledgePipeline(
            settings=settings,
            uploader=uploader,
            extractor=extractor
        )
        print("[GraphGen] KnowledgePipeline instantiated.")
    except Exception as e:
        print(f"[GraphGen] Failed to instantiate Pipeline: {e}")
        sys.exit(1)

    # 6. Run Pipeline
    print("[GraphGen] Running pipeline (Real)...")
    try:
        await pipeline.run()
        print("[GraphGen] Pipeline run completed successfully!")
    except Exception as e:
        print(f"[GraphGen] Pipeline run failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
        
    print("[GraphGen] Operations complete.")

if __name__ == "__main__":
    asyncio.run(test_graphgen())


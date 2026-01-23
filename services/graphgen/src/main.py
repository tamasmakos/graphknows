"""
Main Entry Point for GraphGen Service.
Composition Root: Wires dependencies and starts the pipeline.
"""
import asyncio
import logging
from .kg.config.settings import PipelineSettings
from .kg.falkordb.uploader import KnowledgeGraphUploader
from .kg.pipeline.core import KnowledgePipeline

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    logger.info("Starting GraphGen Service...")
    
    # 1. Load Config
    settings = PipelineSettings()
    logger.info(f"Loaded settings for env: {settings.Config.env_file}")

    # 2. Instantiate Dependencies (Infrastructure)
    uploader = KnowledgeGraphUploader(
        host=settings.falkordb_host,
        port=settings.falkordb_port,
        database="kg", # Could be in settings
        username=None,
        password=None,
        postgres_config=None # simplified for now
    )

    # 3. Instantiate Logic (Inject Dependencies)
    pipeline = KnowledgePipeline(
        settings=settings,
        uploader=uploader
    )

        # 4. Run

        # Since existing logic is async, we run it here

        asyncio.run(pipeline.run())

    

    if __name__ == "__main__":

        main()

    
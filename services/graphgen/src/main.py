"""
Main Entry Point for GraphGen Service (API).
Exposes an API to trigger the pipeline.
V2 - Indentation Fix.
"""
import asyncio
import logging
from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any

from .kg.config.settings import PipelineSettings
from .kg.falkordb.uploader import KnowledgeGraphUploader
from .kg.pipeline.core import KnowledgePipeline
from .kg.graph.extractors import get_extractor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="GraphGen Service")

class PipelineRunRequest(BaseModel):
    input_dir: Optional[str] = None
    clean_database: bool = True

PIPELINE_LOCK = asyncio.Lock()

async def run_pipeline_task(request: PipelineRunRequest):
    async with PIPELINE_LOCK:
        logger.info("Starting GraphGen Pipeline Task...")
        try:
            # 1. Load Config
            settings = PipelineSettings()
            if request.input_dir:
                settings.infra.input_dir = request.input_dir
            
            logger.info(f"Loaded settings. Input: {settings.infra.input_dir}, Output: {settings.infra.output_dir}")
            
            # Prepare config dict
            config_dict = settings.model_dump() if hasattr(settings, 'model_dump') else settings.dict()

            # 2. Instantiate Dependencies
            uploader = KnowledgeGraphUploader(
                host=settings.infra.falkordb_host,
                port=settings.infra.falkordb_port,
                database="kg",
                postgres_config={
                    "enabled": settings.infra.postgres_enabled,
                    "host": settings.infra.postgres_host,
                    "port": settings.infra.postgres_port,
                    "database": settings.infra.postgres_db,
                    "user": settings.infra.postgres_user,
                    "password": settings.infra.postgres_password,
                    "table_name": settings.infra.postgres_table
                }
            )
            
            # Instantiate Extractor
            extractor = get_extractor(config_dict)

            # 3. Instantiate Pipeline
            pipeline = KnowledgePipeline(
                settings=settings,
                uploader=uploader,
                extractor=extractor
            )

            # 4. Run
            try:
                await pipeline.run()
                logger.info("GraphGen Pipeline Task Completed Successfully.")
            finally:
                if extractor:
                    await extractor.close()

        except Exception as e:
            logger.error(f"GraphGen Pipeline Failed: {e}", exc_info=True)

@app.post("/run")
async def run_pipeline(request: PipelineRunRequest, background_tasks: BackgroundTasks):
    if PIPELINE_LOCK.locked():
        raise HTTPException(status_code=409, detail="Pipeline is already running")
    
    background_tasks.add_task(run_pipeline_task, request)
    return {"status": "accepted", "message": "Pipeline run started in background"}

@app.get("/health")
async def health_check():
    return {"status": "ok"}

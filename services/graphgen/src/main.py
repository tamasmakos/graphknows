"""
Main Entry Point for GraphGen Service (API).
Exposes an API to trigger the pipeline.
"""
import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import BackgroundTasks, FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional

from .kg.config.settings import PipelineSettings
from .kg.neo4j.driver import create_driver
from .kg.neo4j.uploader import Neo4jUploader
from .kg.neo4j.indexes import create_indexes
from .kg.pipeline.core import KnowledgePipeline
from .kg.graph.extractors import get_extractor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Lifespan ──────────────────────────────────────────────────────────────────
_neo4j_driver = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _neo4j_driver
    settings = PipelineSettings()
    _neo4j_driver = create_driver()
    await _neo4j_driver.verify_connectivity()
    await create_indexes(_neo4j_driver, database=settings.infra.neo4j_database)
    logger.info("Neo4j driver ready.")
    yield
    if _neo4j_driver:
        await _neo4j_driver.close()
        logger.info("Neo4j driver closed.")


app = FastAPI(title="GraphGen Service", lifespan=lifespan)


# ── Models ────────────────────────────────────────────────────────────────────
class PipelineRunRequest(BaseModel):
    input_dir: Optional[str] = None
    clean_database: bool = True
    skip_communities: bool = False


PIPELINE_LOCK = asyncio.Lock()


# ── Pipeline task ─────────────────────────────────────────────────────────────
async def run_pipeline_task(request: PipelineRunRequest) -> None:
    async with PIPELINE_LOCK:
        logger.info("Starting GraphGen Pipeline Task...")
        try:
            settings = PipelineSettings()
            if request.input_dir:
                settings.infra.input_dir = request.input_dir

            config_dict = settings.model_dump()
            extractor = get_extractor(config_dict)

            uploader = Neo4jUploader(
                driver=_neo4j_driver,
                database=settings.infra.neo4j_database,
            )

            pipeline = KnowledgePipeline(
                settings=settings,
                uploader=uploader,
                extractor=extractor,
                clean_database=request.clean_database,
                run_communities=not request.skip_communities,
            )

            try:
                await pipeline.run()
                logger.info("GraphGen Pipeline Task Completed Successfully.")
            finally:
                if extractor:
                    await extractor.close()

        except Exception as e:
            logger.error("GraphGen Pipeline Failed: %s", e, exc_info=True)


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.post("/run")
async def run_pipeline(
    request: PipelineRunRequest, background_tasks: BackgroundTasks
):
    if PIPELINE_LOCK.locked():
        raise HTTPException(status_code=409, detail="Pipeline is already running")
    background_tasks.add_task(run_pipeline_task, request)
    return {"status": "accepted", "message": "Pipeline run started in background"}


@app.get("/health")
async def health_check():
    return {"status": "ok"}

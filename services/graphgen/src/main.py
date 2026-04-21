"""
Main Entry Point for GraphGen Service (API).
Exposes an API to trigger the pipeline.
"""
import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from neo4j.exceptions import ServiceUnavailable, SessionExpired, Neo4jError
from pydantic import BaseModel
from typing import Any, Dict, List, Optional

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
_neo4j_available = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _neo4j_driver, _neo4j_available
    settings = PipelineSettings()
    _neo4j_driver = create_driver()
    try:
        await asyncio.wait_for(_neo4j_driver.verify_connectivity(), timeout=5.0)
        await create_indexes(_neo4j_driver, database=settings.infra.neo4j_database)
        _neo4j_available = True
        logger.info("Neo4j driver ready.")
    except Exception as e:
        logger.warning(f"Neo4j driver connection failed on startup: {e}")
    yield
    if _neo4j_driver:
        await _neo4j_driver.close()
        logger.info("Neo4j driver closed.")


app = FastAPI(title="GraphGen Service", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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
    if not _neo4j_available:
        raise HTTPException(status_code=503, detail="Database not connected")
    if PIPELINE_LOCK.locked():
        raise HTTPException(status_code=409, detail="Pipeline is already running")
    background_tasks.add_task(run_pipeline_task, request)
    return {"status": "accepted", "message": "Pipeline run started in background"}


@app.get("/health")
async def health_check():
    return {"status": "ok"}


# ── Document management endpoints ─────────────────────────────────────────────

@app.get("/documents")
async def list_documents(database: str = "neo4j") -> Dict[str, Any]:
    """Return all Document nodes with chunk counts."""
    if not _neo4j_available:
        return {"documents": [], "error": "Database not connected"}
    cypher = """
    MATCH (d:Document)
    OPTIONAL MATCH (d)-[:CONTAINS]->(c:Chunk)
    RETURN
        d.doc_id      AS doc_id,
        d.title       AS title,
        d.source_path AS source_path,
        d.created_at  AS created_at,
        count(c)      AS chunk_count
    ORDER BY d.created_at DESC
    """
    try:
        async with _neo4j_driver.session(database=database) as session:
            result = await session.run(cypher)
            rows = await result.data()
        return {"documents": rows}
    except (ServiceUnavailable, SessionExpired, Neo4jError) as e:
        raise HTTPException(status_code=503, detail=f"Database unavailable: {e}")


@app.post("/documents")
async def upload_document(
    file: UploadFile = File(...),
    database: str = "neo4j",
) -> Dict[str, Any]:
    """
    Accept an uploaded file, parse it with the matching parser,
    and store the Document + Chunks in Neo4j.
    Does not run entity extraction — use /run for full ETL.
    """
    if not _neo4j_available:
        raise HTTPException(status_code=503, detail="Database not connected")

    import tempfile, shutil
    from pathlib import Path
    from .kg.parser import get_parser
    # Ensure all parsers are registered
    import importlib; importlib.import_module(".kg.parser.registry", package=__name__.rsplit(".", 1)[0])
    from .kg.neo4j.uploader import Neo4jUploader

    suffix = Path(file.filename or "upload").suffix
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = Path(tmp.name)

    try:
        parser = get_parser(tmp_path)
        parsed = parser.parse(tmp_path)
        # Override title with original filename
        parsed.title = file.filename or parsed.title

        uploader = Neo4jUploader(driver=_neo4j_driver, database=database)
        await uploader.upload_parsed_document(parsed)

        return {"doc_id": parsed.doc_id, "chunks": len(parsed.chunks), "title": parsed.title}
    except (ServiceUnavailable, SessionExpired, Neo4jError) as e:
        raise HTTPException(status_code=503, detail=f"Database unavailable: {e}")
    finally:
        tmp_path.unlink(missing_ok=True)


@app.get("/documents/{doc_id}")
async def get_document(doc_id: str, database: str = "neo4j") -> Dict[str, Any]:
    """Return a document with all its chunks."""
    if not _neo4j_available:
        raise HTTPException(status_code=503, detail="Database not connected")
    cypher = """
    MATCH (d:Document {doc_id: $doc_id})
    OPTIONAL MATCH (d)-[:CONTAINS]->(c:Chunk)
    WITH d, c ORDER BY c.position
    RETURN
        d.doc_id      AS doc_id,
        d.title       AS title,
        d.source_path AS source_path,
        d.created_at  AS created_at,
        collect({
            chunk_id: c.chunk_id,
            position: c.position,
            text: c.text,
            heading_path: c.heading_path
        }) AS chunks
    LIMIT 1
    """
    try:
        async with _neo4j_driver.session(database=database) as session:
            result = await session.run(cypher, {"doc_id": doc_id})
            record = await result.single()
    except (ServiceUnavailable, SessionExpired, Neo4jError) as e:
        raise HTTPException(status_code=503, detail=f"Database unavailable: {e}")
    if not record:
        raise HTTPException(status_code=404, detail="Document not found")
    return dict(record)


@app.delete("/documents/{doc_id}", status_code=204)
async def delete_document(doc_id: str, database: str = "neo4j") -> None:
    """Delete a Document and all its Chunks from Neo4j."""
    if not _neo4j_available:
        raise HTTPException(status_code=503, detail="Database not connected")
    cypher = """
    MATCH (d:Document {doc_id: $doc_id})
    OPTIONAL MATCH (d)-[:CONTAINS]->(c:Chunk)
    DETACH DELETE d, c
    """
    try:
        async with _neo4j_driver.session(database=database) as session:
            await session.run(cypher, {"doc_id": doc_id})
    except (ServiceUnavailable, SessionExpired, Neo4jError) as e:
        raise HTTPException(status_code=503, detail=f"Database unavailable: {e}")


@app.post("/documents/{doc_id}/reprocess")
async def reprocess_document(
    doc_id: str,
    background_tasks: BackgroundTasks,
    database: str = "neo4j",
) -> Dict[str, Any]:
    """Re-run entity extraction + community detection for a single document."""
    if _neo4j_driver is None:
        raise HTTPException(status_code=503, detail="Database not connected")

    async def _task() -> None:
        logger.info("Reprocessing document %s", doc_id)
        # TODO: per-document pipeline run

    background_tasks.add_task(_task)
    return {"status": "accepted", "doc_id": doc_id}


# ── Analytics ─────────────────────────────────────────────────────────────────

@app.get("/analytics")
async def get_analytics(database: str = "neo4j") -> Dict[str, Any]:
    """Return aggregate node/relationship counts."""
    if not _neo4j_available:
        return {"documents": 0, "chunks": 0, "entities": 0, "relationships": 0}
    cypher = """
    MATCH (d:Document) WITH count(d) AS documents
    MATCH (c:Chunk)    WITH documents, count(c) AS chunks
    MATCH (e:Entity)   WITH documents, chunks, count(e) AS entities
    MATCH ()-[r]->()   WITH documents, chunks, entities, count(r) AS relationships
    RETURN documents, chunks, entities, relationships
    """
    try:
        async with _neo4j_driver.session(database=database) as session:
            result = await session.run(cypher)
            record = await result.single()
        return dict(record) if record else {"documents": 0, "chunks": 0, "entities": 0, "relationships": 0}
    except (ServiceUnavailable, SessionExpired, Neo4jError):
        return {"documents": 0, "chunks": 0, "entities": 0, "relationships": 0}


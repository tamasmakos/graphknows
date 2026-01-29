import logging
import sys
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from dashboard.backend.routers import graph, pipeline, retrieval
from dashboard.backend.database import get_db

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Dashboard starting up...")
    db = get_db()
    if db:
        try:
            db.query("RETURN 1")
            logger.info("Connected to FalkorDB")
        except Exception as e:
            logger.error(f"Failed to connect to FalkorDB: {e}")
    yield
    # Shutdown
    if db:
        db.close()
        logger.info("Closed database connection.")

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(graph.router, prefix="/api/graph", tags=["graph"])
app.include_router(pipeline.router, prefix="/api/pipeline", tags=["pipeline"])
app.include_router(retrieval.router, prefix="/api/retrieval", tags=["retrieval"])

# Explicitly serve index.html at root
static_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../static"))

@app.get("/")
async def read_index():
    index_path = os.path.join(static_dir, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"error": f"index.html not found at {index_path}"}

# Mount static files at /static
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")
else:
    print(f"Warning: Static directory not found at {static_dir}")

# Fallback for other paths to serve index.html (SPA support)
@app.get("/{full_path:path}")
async def serve_spa(full_path: str):
    if full_path.startswith("api/"):
        return {"error": "Not Found"}
    index_path = os.path.join(static_dir, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"error": f"index.html not found"}
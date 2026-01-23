import sys
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# Add project root to path to allow importing src.kg
# Root is ../../../ from src/dashboard/backend
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from src.dashboard.backend.routers import graph, retrieval
from src.dashboard.backend.database import get_db

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    db = get_db()
    if db:
        try:
            # Simple health check query
            db.query("RETURN 1")
            print("Connected to FalkorDB")
        except Exception as e:
            print(f"Failed to connect to FalkorDB: {e}")
    else:
        print("Failed to initialize database client")
    yield
    # Shutdown
    if db:
        db.close()

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For dev
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(graph.router, prefix="/api/graph", tags=["graph"])
app.include_router(retrieval.router, prefix="/api/retrieval", tags=["retrieval"])

# Mount static files (Simplified Dashboard)
static_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../static"))
if os.path.exists(static_dir):
    app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")
else:
    print(f"Warning: Static directory not found at {static_dir}")

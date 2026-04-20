from fastapi import APIRouter, HTTPException, BackgroundTasks
import subprocess
import os
import logging

logger = logging.getLogger(__name__)

router = APIRouter()

from fastapi import APIRouter, HTTPException, BackgroundTasks
import requests
import os
import logging
import asyncio

logger = logging.getLogger(__name__)

router = APIRouter()

GRAPHGEN_URL = os.getenv("GRAPHGEN_URL", "http://graphgen:8000")

@router.post("/run")
async def run_pipeline():
    try:
        # Trigger remote pipeline
        response = requests.post(f"{GRAPHGEN_URL}/run", json={"clean_database": True}, timeout=5)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to trigger pipeline at {GRAPHGEN_URL}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to trigger pipeline: {e}")

@router.get("/status")
async def get_status():
    # Simple check if service is up, not exact pipeline status (needs more complex API on graphgen side)
    try:
        response = requests.get(f"{GRAPHGEN_URL}/health", timeout=2)
        return {"service_status": "ok" if response.status_code == 200 else "error"}
    except:
        return {"service_status": "unreachable"}


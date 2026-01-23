from fastapi import APIRouter, HTTPException, BackgroundTasks
import subprocess
import os
import logging

logger = logging.getLogger(__name__)

router = APIRouter()

PIPELINE_RUNNING = False

def run_pipeline_task():
    global PIPELINE_RUNNING
    PIPELINE_RUNNING = True
    try:
        # Assuming we are at PROJECT_ROOT
        logger.info("Starting GraphGen pipeline...")
        env = os.environ.copy()
        # Set PYTHONPATH to include project root
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
        env["PYTHONPATH"] = project_root
        
        # Run module
        result = subprocess.run(
            ["python", "-m", "services.graphgen.src.main"],
            cwd=project_root,
            env=env,
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            logger.info("GraphGen pipeline completed successfully.")
            logger.info(result.stdout)
        else:
            logger.error(f"GraphGen pipeline failed with code {result.returncode}")
            logger.error(result.stderr)
            
    except Exception as e:
        logger.error(f"Failed to run pipeline: {e}")
    finally:
        PIPELINE_RUNNING = False

@router.post("/run")
async def run_pipeline(background_tasks: BackgroundTasks):
    global PIPELINE_RUNNING
    if PIPELINE_RUNNING:
        raise HTTPException(status_code=400, detail="Pipeline is already running")
    
    background_tasks.add_task(run_pipeline_task)
    return {"status": "started", "message": "GraphGen pipeline started in background"}

@router.get("/status")
async def get_status():
    global PIPELINE_RUNNING
    return {"running": PIPELINE_RUNNING}

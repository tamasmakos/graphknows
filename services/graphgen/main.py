"""GraphGen pipeline entry point."""

import asyncio
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from graphgen.config.settings import PipelineSettings
from graphgen.orchestrator import KnowledgePipeline
from graphgen.utils.logging import configure_logging

logger = logging.getLogger(__name__)


def resolve_env_file() -> str:
    """Resolve the project .env file deterministically.

    Preference order:
    1. GRAPHGEN_ENV_FILE override
    2. repo-root .env
    3. package-dir graphgen/.env
    4. default repo-root .env path (even if absent)
    """
    override = os.environ.get("GRAPHGEN_ENV_FILE")
    if override:
        return override

    repo_root = Path(__file__).resolve().parents[1]
    candidates = [
        repo_root / ".env",
        repo_root / "graphgen" / ".env",
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return str(candidates[0])


ENV_FILE = resolve_env_file()
load_dotenv(ENV_FILE)


async def run_pipeline() -> None:
    configure_logging()
    try:
        settings = PipelineSettings.load(env_file=ENV_FILE) # loads from config.yaml and resolved .env
        configure_logging(debug=settings.debug)

        logger.info("Initializing GraphGen Pipeline...")
        
        logger.info(f"Configuration loaded. Input: {settings.infra.input_dir}, Output: {settings.infra.output_dir}")
        
        from graphgen.utils.graphdb.neo4j_adapter import Neo4jGraphUploader
        logger.info("Initializing Neo4j Backend...")
        uploader = Neo4jGraphUploader(
            host=settings.infra.neo4j_host,
            port=settings.infra.neo4j_port,
            username=settings.infra.neo4j_user,
            password=settings.infra.neo4j_password,
            database="neo4j"
        )
            
        # Initialize Extractor
        from graphgen.pipeline.entity_relation.extractors import get_extractor
        extractor = get_extractor(settings.model_dump())
        
        # Active runtime always uses the single-run knowledge pipeline.
        # Iterative experimentation, if retained in the repo for historical or
        # auxiliary use, is no longer part of the main execution path.
        pipeline = KnowledgePipeline(settings, uploader, extractor)

        # Run
        await pipeline.run()
        
    except Exception as e:
        logger.critical(f"Pipeline failed: {e}", exc_info=True)
        sys.exit(1)

def main() -> None:
    """Entry point for the console script."""
    asyncio.run(run_pipeline())

if __name__ == "__main__":
    main()


import os
import shutil
import json
import asyncio
import logging
from typing import List, Dict, Any
from datetime import datetime, timedelta

from .adapter import generate_single_day
from .metrics import GraphMetricTracker
from ..kg.config.settings import PipelineSettings
from ..kg.falkordb.uploader import KnowledgeGraphUploader
from ..kg.pipeline.core import KnowledgePipeline
from ..kg.graph.extractors import get_extractor

logger = logging.getLogger(__name__)

class LifeSimulation:
    """
    Orchestrates the Life Simulation:
    Generating Data -> Daily Ingestion -> Graph Evolution -> Metric Tracking.
    """
    
    def __init__(self, days_to_simulate: int, start_date: str):
        self.days = days_to_simulate
        self.start_date = start_date
        self.settings = PipelineSettings()
        # Ensure we control the input directory
        # We will use the configured input_dir from settings or default to /app/input
        self.input_dir = self.settings.infra.input_dir or "/app/input"
        self.tracker = GraphMetricTracker(self.settings)
        self.results = []
        
    async def run(self):
        start_time = datetime.now()
        logger.info(f"🚀 [Orchestrator] Starting Life Simulation for {self.days} days from {self.start_date}...")
        
        buffer_dir = os.path.join(self.settings.infra.output_dir, "simulation_buffer")
        os.makedirs(buffer_dir, exist_ok=True)
        
        start_dt = datetime.strptime(self.start_date, "%Y-%m-%d")
        
        # Phase 2: The Time Loop (Now Interleaved)
        for days_passed in range(self.days):
            current_date = start_dt + timedelta(days=days_passed)
            day_num = days_passed + 1
            
            print(f"\n🔄 --- Processing Simulation Day {day_num}/{self.days} [{current_date.date()}] ---")
            logger.info(f"📅 [Orchestrator] Processing Day {day_num}/{self.days} | Date: {current_date.date()}")
            
            # A. Generate Data for THIS Day
            try:
                # Generate single day file
                file_path = generate_single_day(current_date, output_dir=buffer_dir)
                if not file_path or not os.path.exists(file_path):
                     raise RuntimeError(f"Failed to generate file for {current_date.date()}")
                     
            except Exception as e:
                logger.critical(f"❌ [Orchestrator] Data Generation Failed for Day {day_num}: {e}", exc_info=True)
                return # Stop simulation if generation fails

            # B. Stage Data
            try:
                self._stage_input_file(file_path)
            except Exception as e:
                logger.error(f"❌ [Orchestrator] Failed to stage file {file_path}: {e}", exc_info=True)
                continue
            
            # C. Run GraphGen Pipeline
            # Clean DB only on first day (Genesis)
            is_genesis = (days_passed == 0)
            logger.info(f"▶ [Orchestrator] Starting Pipeline Task (Clean DB: {is_genesis})...")
            
            try:
                # Direct instantiation to avoid lock deadlock
                # 1. Config
                config_dict = self.settings.model_dump() if hasattr(self.settings, 'model_dump') else self.settings.dict()
                
                # 2. Uploader
                uploader = KnowledgeGraphUploader(
                    host=self.settings.infra.falkordb_host,
                    port=self.settings.infra.falkordb_port,
                    database="kg",
                    postgres_config={
                        "enabled": self.settings.infra.postgres_enabled,
                        "host": self.settings.infra.postgres_host,
                        "port": self.settings.infra.postgres_port,
                        "database": self.settings.infra.postgres_db,
                        "user": self.settings.infra.postgres_user,
                        "password": self.settings.infra.postgres_password,
                        "table_name": self.settings.infra.postgres_table
                    }
                )
                
                # 3. Extractor
                extractor = get_extractor(config_dict)
                
                # 4. Pipeline
                pipeline = KnowledgePipeline(
                    settings=self.settings,
                    uploader=uploader,
                    extractor=extractor,
                    clean_database=is_genesis,
                    run_communities=False # Skip communities, done later in step D
                )
                
                await pipeline.run()
                
                if extractor:
                    await extractor.close()
                    
                logger.info(f"✅ [Orchestrator] Pipeline Ingestion Completed for Day {day_num}.")
            except Exception as e:
                logger.error(f"❌ [Orchestrator] Pipeline failed for Day {day_num}: {e}", exc_info=True)
                raise
            
            # D. Analyze Evolution (Community Detection & Summarization)
            logger.info(f"🧠 [Orchestrator] Starting Graph Evolution Analysis (Global Communities)...")
            try:
                stats = await self.tracker.analyze_evolution(day_index=days_passed)
                
                # Add timestamp
                stats['timestamp'] = datetime.now().isoformat()
                self.results.append(stats)
                
                logger.info(f"📊 [Orchestrator] Day {day_num} Metrics: Modularity={stats.get('modularity', 0):.2f}, Nodes={stats.get('node_count')}, Top Entity={stats.get('top_entity')}")
                print(f"   📊 Day {day_num} Stats: Modularity={stats.get('modularity', 0):.2f}, Nodes={stats.get('node_count')}")
                
            except Exception as e:
                logger.error(f"❌ [Orchestrator] Metrics calculation failed for Day {day_num}: {e}", exc_info=True)
        
        # Phase 3: Final Report
        self._save_report()
        duration = datetime.now() - start_time
        logger.info(f"✅ [Orchestrator] Simulation Complete in {duration}. Report saved.")

    def _stage_input_file(self, source_path: str):
        """Prepare the input directory with ONLY the current day's file."""
        # Ensure target dir exists
        os.makedirs(self.input_dir, exist_ok=True)
        
        # Clean target dir
        logger.info(f"🧹 [Orchestrator] Cleaning input directory: {self.input_dir}")
        for filename in os.listdir(self.input_dir):
            file_path = os.path.join(self.input_dir, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                logger.warning(f"⚠️ Failed to delete {file_path}: {e}")
        
        # Copy new file
        dest_path = os.path.join(self.input_dir, os.path.basename(source_path))
        shutil.copy2(source_path, dest_path)
        logger.info(f"📥 [Orchestrator] Staged file: {dest_path}")

    def _save_report(self):
        output_path = os.path.join(self.settings.infra.output_dir or "output", "simulation_report.json")
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(self.results, f, indent=2)
            logger.info(f"📝 [Orchestrator] Report saved to {output_path}")
        except Exception as e:
            logger.error(f"❌ [Orchestrator] Failed to save report: {e}")

if __name__ == "__main__":
    # Simple CLI for testing
    import argparse
    import asyncio
    
    # Configure logging to stdout
    logging.basicConfig(level=logging.INFO)
    
    parser = argparse.ArgumentParser(description="Run Life Simulation")
    parser.add_argument("--days", type=int, default=3, help="Number of days to simulate")
    parser.add_argument("--start-date", type=str, default="2025-01-01", help="Start date YYYY-MM-DD")
    
    args = parser.parse_args()
    
    sim = LifeSimulation(args.days, args.start_date)
    asyncio.run(sim.run())

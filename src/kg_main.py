#!/usr/bin/env python3
"""
Main entry point for Knowledge Graph pipeline.

Supports both batch and incremental processing modes.
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    logging.warning("python-dotenv not found, assuming environment variables are set")

from kg.config.loader import load_config
from kg.service import IterativeGraphBuilder
from kg.utils.health import verify_all_services


# Configure logging
# Configure logging
class CustomFormatter(logging.Formatter):
    """Custom formatter to add colors and better formatting."""
    
    grey = "\x1b[38;20m"
    blue = "\x1b[34;20m"
    green = "\x1b[32;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    format_str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    FORMATS = {
        logging.DEBUG: grey + format_str + reset,
        logging.INFO: blue + "%(message)s" + reset,  # Simplified info for better readability
        logging.WARNING: yellow + format_str + reset,
        logging.ERROR: red + format_str + reset,
        logging.CRITICAL: bold_red + format_str + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt, datefmt='%H:%M:%S')
        return formatter.format(record)

# Create console handler with custom formatter
console_handler = logging.StreamHandler()
console_handler.setFormatter(CustomFormatter())

# Configure root logger
logging.basicConfig(
    level=logging.INFO,
    handlers=[console_handler]
)
logger = logging.getLogger(__name__)


async def main_async(config_path: str, reset: bool = False):
    """Main async entry point."""
    logger.info("=" * 80)
    logger.info("Knowledge Graph Pipeline")
    logger.info("=" * 80)
    
    if reset:
        logger.warning("⚠️  Reset requested - will clear all data before running")
    
    # Load config to check services
    config_data = load_config(config_path)
    
    # Pre-flight health checks
    logger.info("Performing pre-flight health checks...")
    if not verify_all_services(config_data):
        logger.error("Pre-flight health checks failed. Aborting pipeline.")
        sys.exit(1)
    
    # Run the iterative pipeline
    from kg.pipeline.iterative import run_iterative_pipeline
    results = await run_iterative_pipeline(config_path, reset=reset)
    
    logger.info("=" * 80)
    logger.info("PIPELINE EXECUTION COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Documents processed: {results.get('documents_processed', 0)}")
    logger.info(f"Nodes merged: {results.get('nodes_merged', 0)}")
    logger.info(f"Relationships merged: {results.get('relationships_merged', 0)}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Knowledge Graph Pipeline - Batch or Incremental Processing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default config (respects mode in config.yaml)
  python src/kg_main.py
  
  # Use specific config file
  python src/kg_main.py --config my_config.yaml
  
  # Clean start (clear all data before running)
  python src/kg_main.py --clean-start
        """
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to configuration file (default: config.yaml)'
    )
    
    parser.add_argument(
        '--clean-start',
        action='store_true',
        help='Clear FalkorDB and processing state before running (overrides config)'
    )
    
    args = parser.parse_args()
    
    # If --clean-start is provided, temporarily modify config
    if args.clean_start:
        logger.warning("--clean-start flag provided, will clear all data")
        # We'll handle this in main_async
    
    try:
        asyncio.run(main_async(args.config, reset=args.clean_start))
    except KeyboardInterrupt:
        logger.info("\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()

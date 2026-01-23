import os
import sys
from pathlib import Path

from src.common.config.loader import load_config
from src.common.config.schema import Config

# Add project root (workspace root) to Python path.
# This file lives at `src/infrastructure/config.py` (inside the container's `/app` root), so we need to go
# three levels up: infrastructure -> src -> app (project root).
project_root = str(Path(__file__).parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)


def get_app_config(config_file: str | None = None) -> Config:
    """
    Load the application configuration for the web/agent app.

    Resolution order (all paths are relative to the project root):
    1. Explicit `config_file` argument
    2. `APP_CONFIG_FILE` environment variable
    3. Top-level `config.yaml` shared with the KG pipeline
    """
    # Determine which config file path to use (relative to project root).
    # Default to the shared top-level `config.yaml` so DB/LLM settings come
    # from the same source as the KG pipeline configuration.
    relative_path = config_file or os.environ.get("APP_CONFIG_FILE") or "config.yaml"

    config_path = os.path.join(project_root, relative_path)
    return load_config(config_path)




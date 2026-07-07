"""Shared pytest configuration.

sys.path is managed via pytest.ini ``pythonpath`` entries; the project-root
insert below is belt-and-suspenders for direct invocations.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Ensure project root is in sys.path (belt-and-suspenders alongside pytest.ini pythonpath)
_project_root = str(Path(__file__).parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)


@pytest.fixture(autouse=True)
def _reset_settings_singletons():
    """Reset module-level _settings singletons so env-patching tests stay isolated."""
    yield
    for mod_name in ("graphknows.mcp.stdio_server",):
        mod = sys.modules.get(mod_name)
        if mod is not None and hasattr(mod, "_settings"):
            mod._settings = None

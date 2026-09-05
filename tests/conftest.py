"""Shared pytest configuration.

sys.path is managed via pytest.ini ``pythonpath`` entries; the project-root
insert below is belt-and-suspenders for direct invocations.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

# Ensure project root is in sys.path (belt-and-suspenders alongside pytest.ini pythonpath)
_project_root = str(Path(__file__).parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# Keep unit tests hermetic: the relation verifier is ON in production but would
# pull a ~500MB DeBERTa model on first extract. setdefault so a test that wants
# it can still opt in.
os.environ.setdefault("GRAPHKNOWS_RELATION_VERIFIER", "false")

# The air-gapped node (tests/test_offline.py) sets these through monkeypatch,
# which is too late once an earlier test has imported huggingface_hub: its
# offline constant is read at import, so a later env change is ignored and a
# cached model still revalidates against the Hub. The dev image deliberately
# does not bake the switches (tests/test_repo_hygiene.py), so the session sets
# them here, before any import — the node then measures the guarded call path
# rather than test order. setdefault: an operator who wants the Hub keeps it.
for _offline_switch in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
    os.environ.setdefault(_offline_switch, "1")


@pytest.fixture(autouse=True)
def _reset_settings_singletons():
    """Reset module-level _settings singletons so env-patching tests stay isolated."""
    yield
    for mod_name in ("graphknows.server.mcp.stdio_server",):
        mod = sys.modules.get(mod_name)
        if mod is not None and hasattr(mod, "_settings"):
            mod._settings = None

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
    """Drop the process settings around each test, so env-patching stays isolated.

    ``get_settings`` is an ``lru_cache``; without this a cached instance built
    under one environment would leak into the next test, and tests would go back
    to poking module globals to get a fresh one.
    """
    from graphknows.settings import get_settings

    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


# ---------------------------------------------------------------------------
# Live-ArcadeDB guard, shared by every integration module
# ---------------------------------------------------------------------------


def _arcadedb_reachable() -> bool:
    """Return True when ArcadeDB answers at the configured URL."""
    import httpx

    from graphknows.settings import GraphKnowsSettings

    try:
        s = GraphKnowsSettings()
        r = httpx.get(
            f"{s.arcadedb_url}/api/v1/exists/mem",
            auth=(s.arcadedb_user, s.arcadedb_password.get_secret_value()),
            timeout=3,
        )
    except Exception:
        return False
    else:
        return r.status_code == 200


def require_arcadedb() -> None:
    """Skip the caller when ArcadeDB is down — fail when the run demands it.

    ``GRAPHKNOWS_REQUIRE_ARCADEDB=1`` is CI's promise that a service container
    is up; a skip there would report a suite that never ran as green.
    """
    if _arcadedb_reachable():
        return
    if os.environ.get("GRAPHKNOWS_REQUIRE_ARCADEDB") == "1":
        pytest.fail("ArcadeDB not reachable and GRAPHKNOWS_REQUIRE_ARCADEDB=1")
    pytest.skip("ArcadeDB not reachable")


@pytest.fixture
def arcadedb_required() -> None:
    """Fixture form of :func:`require_arcadedb`."""
    require_arcadedb()

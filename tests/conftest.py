import pytest
import os
import datetime
import json
from pathlib import Path

# Create a reports directory
REPORT_DIR = Path("/workspaces/graphknows/test_reports")
REPORT_DIR.mkdir(exist_ok=True)

@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    rep = outcome.get_result()
    
    # We only care about the actual call, not setup/teardown
    if rep.when == "call":
        mode = "a" if (REPORT_DIR / "latest_run.jsonl").exists() else "w"
        with open(REPORT_DIR / "latest_run.jsonl", mode) as f:
            result = {
                "node_id": item.nodeid,
                "outcome": rep.outcome,
                "duration": rep.duration,
                "timestamp": datetime.datetime.now().isoformat()
            }
            f.write(json.dumps(result) + "\n")

@pytest.fixture(scope="session", autouse=True)
def setup_test_env():
    """Setup test environment variables for the session."""
    # Ensure no real API calls are made if not mocked
    os.environ["TESTING"] = "true"
    yield
    # Cleanup if needed

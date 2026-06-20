"""Ensure services/agent is importable when running ontology tests."""

from __future__ import annotations

import sys
from pathlib import Path

# Make services/agent importable (ontology package lives there)
_agent_path = str(Path(__file__).parent.parent.parent / "services" / "agent")
if _agent_path not in sys.path:
    sys.path.insert(0, _agent_path)

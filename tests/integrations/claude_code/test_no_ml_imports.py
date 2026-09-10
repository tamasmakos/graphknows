"""The Claude Code hooks are stdlib-only: importing them pulls no ML stack (SC-010).

A hook runs on every stop and pre-compaction event, inside the agent's own
session, so its import cost is paid by the user each time. Loading torch or a
spaCy pipeline there would cost seconds per event.
"""

from __future__ import annotations

import subprocess
import sys

_ML = ["torch", "transformers", "spacy", "sentence_transformers", "gliner"]


def test_importing_the_hooks_loads_no_ml_module() -> None:
    code = (
        "import sys, graphknows.integrations.claude_code.hooks; "
        f"print(','.join(m for m in {_ML!r} if m in sys.modules))"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == ""

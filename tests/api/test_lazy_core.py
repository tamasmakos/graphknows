"""The core import surface must not pull the heavy ML/LLM stack.

``import graphknows`` and ``import graphknows.integrations.client`` must work on a core-only
install (no local-embeddings/extraction/analytics/assisted extras) and must not
eagerly import sentence-transformers, gliner2, spacy, igraph, leidenalg, torch,
or dspy. This test fails if an eager import creeps back onto the core path.

``dateparser`` is on the same list for a different reason: it is a declared core
dependency, but building its language loader costs seconds at import time, and
every importer paid that whether or not it ever parsed a date.
"""

from __future__ import annotations

import subprocess
import sys

_HEAVY = ["sentence_transformers", "gliner2", "spacy", "igraph", "leidenalg", "torch", "dspy"]


def _heavy_modules_after(import_line: str) -> list[str]:
    """Import in a fresh interpreter and report which heavy modules loaded."""
    code = f"import sys; {import_line}; print(','.join(m for m in {_HEAVY!r} if m in sys.modules))"
    out = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
    )
    line = out.stdout.strip().splitlines()[-1] if out.stdout.strip() else ""
    return [m for m in line.split(",") if m]


def test_import_graphknows_is_light() -> None:
    assert _heavy_modules_after("import graphknows") == []


def test_import_client_is_light() -> None:
    assert _heavy_modules_after("import graphknows.integrations.client") == []


def test_import_does_not_pull_dateparser() -> None:
    """dateparser is deferred to the call sites that parse; importing must not load it."""
    code = "import sys, graphknows; print('dateparser' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip().splitlines()[-1] == "False"

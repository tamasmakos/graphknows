"""Check that optional service dependencies are not imported unconditionally.

This script scans graphknows/ for bare top-level imports of packages that
are declared as optional extras (e.g. neo4j, asyncpg, rdflib) and flags any
file that imports them without a try/except guard.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

# Packages that must only be imported inside a try/except block
# because they are optional extras not required for all deployments.
OPTIONAL_PACKAGES: set[str] = {
    "neo4j",
    "asyncpg",
    "rdflib",
    "pgvector",
    "gliner",
    "dspy",
}

ROOT = Path(__file__).parent.parent / "graphknows"


def _is_guarded(node: ast.stmt, tree: ast.Module) -> bool:
    """Return True if *node* is nested inside a try/except block."""
    for parent in ast.walk(tree):
        if isinstance(parent, ast.Try):
            for child in ast.walk(parent):
                if child is node:
                    return True
    return False


def check_file(path: Path) -> list[str]:
    source = path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError:
        return []

    # Collect all top-level import statements (direct children of Module)
    errors: list[str] = []
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.Import | ast.ImportFrom):
            if isinstance(node, ast.Import):
                names = [alias.name.split(".")[0] for alias in node.names]
            else:
                names = [node.module.split(".")[0]] if node.module else []

            for name in names:
                if name in OPTIONAL_PACKAGES and not _is_guarded(node, tree):
                    errors.append(
                        f"{path}:{node.lineno}: bare import of optional package '{name}'"
                        " — wrap in try/except ImportError"
                    )
    return errors


def main() -> int:
    all_errors: list[str] = []
    for py_file in sorted(ROOT.rglob("*.py")):
        # Skip vendored / generated directories and dedicated store adapters
        # (store modules are expected to import their own optional dependency)
        if any(part in py_file.parts for part in (".venv", "__pycache__", "graphgen", "stores")):
            continue
        all_errors.extend(check_file(py_file))

    if all_errors:
        for err in all_errors:
            print(err, file=sys.stderr)
        print(f"\n{len(all_errors)} undeclared service import(s) found.", file=sys.stderr)
        return 1

    print("Service import check: clean")
    return 0


if __name__ == "__main__":
    sys.exit(main())

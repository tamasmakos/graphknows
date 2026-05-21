#!/usr/bin/env python3
"""
Pre-commit import validator.

Statically walks each service's source tree, collects all top-level
(non-relative) imports, and checks that every third-party package is
declared in that service's pyproject.toml dependencies.

Also checks for any imports of sentence_transformers (banned).

Usage:
    python3 scripts/check_imports.py          # check all services
    python3 scripts/check_imports.py agent    # check one service

Exit 0 = clean. Exit 1 = undeclared imports found.
"""

import ast
import re
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).parent.parent

# ── stdlib module names (Python 3.11) ────────────────────────────────────────
STDLIB = frozenset(sys.stdlib_module_names) | {
    # a few extras that appear in older lists
    "posixpath",
    "ntpath",
    "fnmatch",
    "antigravity",
    "this",
    "typing_extensions",  # bundled with Python ≥3.11
}

# ── package name → importable name mapping (where they differ) ───────────────
# pyproject dep name → the `import X` name it provides
DIST_TO_IMPORT: dict[str, str] = {
    "scikit-learn": "sklearn",
    "python-dotenv": "dotenv",
    "pyyaml": "yaml",
    "pillow": "PIL",
    "beautifulsoup4": "bs4",
    "opencv-python": "cv2",
    "opencv-python-headless": "cv2",
    "python-multipart": "multipart",
    "python-jose": "jose",
    "psycopg2-binary": "psycopg2",
    "pymupdf": "fitz",
    "pdfminer.six": "pdfminer",
    "tavily-python": "tavily",
    "beautifulsoup4": "bs4",
    "langchain-core": "langchain_core",
    "langchain-openai": "langchain_openai",
    "langchain-text-splitters": "langchain_text_splitters",
    "langchain-experimental": "langchain_experimental",
    "langchain-community": "langchain_community",
    "langchain-groq": "langchain_groq",
    "langfuse": "langfuse",
    "python-json-logger": "pythonjsonlogger",
    "neo4j": "neo4j",
    "asyncpg": "asyncpg",
    "sentence-transformers": "sentence_transformers",  # banned
}

# Services to check: name → (source dirs, pyproject path)
SERVICES: dict[str, tuple[list[Path], Path]] = {
    "graphknows": (
        [ROOT / "services/graphknows/graphknows"],
        ROOT / "services/graphknows/pyproject.toml",
    ),
    "agent": (
        [ROOT / "services/agent"],
        ROOT / "services/agent/pyproject.toml",
    ),
    "search": (
        [ROOT / "services/search"],
        ROOT / "services/search/pyproject.toml",
    ),
}

BANNED = {"sentence_transformers"}

# Imports that are intentionally optional (extras / research / heavy ML deps).
# Files matching these path fragments are allowed to import them without
# declaring them as hard deps — they're expected to be guarded by try/except
# or only used in offline research pipelines.
OPTIONAL_PATH_FRAGMENTS = {
    "analytics",
    "visualizer",
    "thesis_plots",
    "plot_style",
    "node2vec",
    "community/detection",
    "iterative_orchestrator",
}

# Packages that are transitive deps of declared packages (fastapi → starlette)
# and safe to import without explicit declaration.
TRANSITIVE_OK = {"starlette"}


def parse_declared_imports(pyproject: Path) -> set[str]:
    """Return the set of importable names declared in a pyproject.toml."""
    with open(pyproject, "rb") as f:
        data = tomllib.load(f)

    raw_deps: list[str] = []
    raw_deps += data.get("project", {}).get("dependencies", [])
    for extra in data.get("project", {}).get("optional-dependencies", {}).values():
        raw_deps += extra

    importable: set[str] = set()
    for dep in raw_deps:
        # strip version specifiers: "numpy>=1.24" → "numpy"
        name = re.split(r"[>=<!;\s\[]", dep.strip())[0].lower().replace("-", "_")
        # map dist name → import name
        canonical = DIST_TO_IMPORT.get(name.replace("_", "-"), name)
        importable.add(canonical)
        importable.add(name)  # also accept the dist name itself as import

    return importable


def collect_imports(src_dirs: list[Path]) -> dict[str, set[str]]:
    """
    Walk source dirs, parse each .py file with ast, return
    {file_path: {top_level_import_names}}.
    """
    results: dict[str, set[str]] = {}
    for src in src_dirs:
        for py in src.rglob("*.py"):
            if any(p in {".venv", "__pycache__", "research"} for p in py.parts):
                continue
            try:
                tree = ast.parse(py.read_text(encoding="utf-8", errors="replace"))
            except SyntaxError:
                continue
            names: set[str] = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        names.add(alias.name.split(".")[0])
                elif isinstance(node, ast.ImportFrom):
                    if node.level == 0 and node.module:  # absolute import only
                        names.add(node.module.split(".")[0])
            results[str(py.relative_to(ROOT))] = names
    return results


def check_service(name: str, src_dirs: list[Path], pyproject: Path) -> list[str]:
    declared = parse_declared_imports(pyproject)
    file_imports = collect_imports(src_dirs)

    # also include workspace siblings as declared
    # (graphknows imports graphknows itself, agent imports agent, etc.)
    internal = {name, "graphknows", "graphgen", "agent", "search", "services"}
    declared |= internal

    errors: list[str] = []
    for filepath, imports in sorted(file_imports.items()):
        # Allow optional path fragments (analytics, viz, thesis, etc.)
        is_optional_file = any(frag in filepath for frag in OPTIONAL_PATH_FRAGMENTS)
        for mod in sorted(imports):
            if mod in STDLIB:
                continue
            if mod.startswith("_"):
                continue
            if mod in declared:
                continue
            if mod in TRANSITIVE_OK:
                continue
            if mod in BANNED:
                errors.append(f"  BANNED  {filepath}: import {mod}")
            elif is_optional_file:
                continue  # optional — not a hard error
            else:
                errors.append(f"  MISSING {filepath}: import {mod}")
    return errors


def main() -> None:
    filter_services = sys.argv[1:] if len(sys.argv) > 1 else list(SERVICES)
    total_errors: list[str] = []

    for svc_name in filter_services:
        if svc_name not in SERVICES:
            print(f"Unknown service: {svc_name}. Known: {list(SERVICES)}")
            sys.exit(1)
        src_dirs, pyproject = SERVICES[svc_name]
        errors = check_service(svc_name, src_dirs, pyproject)
        if errors:
            print(f"\n[{svc_name}] {len(errors)} issue(s):")
            for e in errors:
                print(e)
            total_errors.extend(errors)
        else:
            print(f"[{svc_name}] OK")

    if total_errors:
        print(
            f"\n{len(total_errors)} total issue(s). Declare missing deps in pyproject.toml before committing."
        )
        sys.exit(1)
    else:
        print("\nAll services clean.")


if __name__ == "__main__":
    main()

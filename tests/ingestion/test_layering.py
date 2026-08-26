"""Pin the intra-`ingestion` dependency direction: extraction must not import stm.

`extraction` mints entity/relation spans; `stm` writes them to storage. That
makes `stm -> extraction` the correct and only permitted direction. The
.importlinter `layers` contract only constrains top-level packages -- both
`extraction` and `stm` sit inside the single `graphknows.ingestion` layer, so a
module-level cycle between them is invisible to it, and file-level cycle
detection sees two files importing different files rather than two packages
importing each other.

Parses `extraction`'s source with `ast` instead of importing it: the package
transitively pulls in GLiNER/torch at import time, which is unacceptable for a
unit test.
"""

from __future__ import annotations

import ast
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_EXTRACTION_PKG = _REPO_ROOT / "graphknows" / "ingestion" / "extraction"
_FORBIDDEN_MODULE = "graphknows.ingestion.stm"


def _imported_module_names(node: ast.Import | ast.ImportFrom) -> list[str]:
    if isinstance(node, ast.Import):
        return [alias.name for alias in node.names]
    if node.module is None:
        return []
    return [node.module]


def _stm_imports(path: Path) -> list[tuple[int, str]]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    violations: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Import, ast.ImportFrom)):
            continue
        for module in _imported_module_names(node):
            if module == _FORBIDDEN_MODULE or module.startswith(f"{_FORBIDDEN_MODULE}."):
                violations.append((node.lineno, module))
    return violations


class TestExtractionDoesNotImportStm:
    def test_extraction_never_imports_stm(self) -> None:
        assert _EXTRACTION_PKG.is_dir(), f"expected extraction package at {_EXTRACTION_PKG}"

        offenders: list[str] = []
        for py_file in sorted(_EXTRACTION_PKG.rglob("*.py")):
            rel_path = py_file.relative_to(_REPO_ROOT)
            for lineno, module in _stm_imports(py_file):
                offenders.append(f"{rel_path}:{lineno} -> {module}")

        assert not offenders, (
            "extraction must not import stm (stm -> extraction is the "
            "permitted direction); found:\n" + "\n".join(offenders)
        )

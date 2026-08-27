"""Refuse to start when the image's dependencies are stale relative to the source.

This is the guard for the failure that made a stale image invisible for seven
weeks: `docker-compose.yaml` bind-mounts the working tree over `/app` and puts it
first on `PYTHONPATH`, so a container runs **live source against image
dependencies**. Code is always current; site-packages are frozen at whenever
someone last ran `docker compose build`. Nothing detects the gap, and it surfaces
only as a `ModuleNotFoundError` in whichever code path a user happens to hit
first — or, worse, as a silently degraded feature whose import-guard fallback
swallows the evidence.

One check, fast enough to run on every container start (~20ms): every
dependency `pyproject.toml` declares must be installed. This is the precise
form of the bug and has no false positives.

Hard-fails. Set GRAPHKNOWS_SKIP_PREFLIGHT=1 to bypass — needed to get a shell
in a container that is *already* wedged, and for nothing else.

In a deployment there is no bind-mount, so there is no source to compare against
and the check no-ops: the image is the truth by definition.
"""

from __future__ import annotations

import contextlib
import os
import re
import sys
import tomllib
from importlib.metadata import PackageNotFoundError, distribution
from pathlib import Path

APP = Path(os.environ.get("GRAPHKNOWS_APP_DIR", "/app"))


def _fail(title: str, detail: str) -> None:
    bar = "#" * 74
    print(f"\n{bar}", file=sys.stderr)
    print(f"#  {title}".ljust(73) + "#", file=sys.stderr)
    print(f"{bar}", file=sys.stderr)
    print(detail.rstrip(), file=sys.stderr)
    print(
        "\n  Fix:   docker compose build\n"
        "  Bypass (to get a shell in an already-wedged container):\n"
        "         GRAPHKNOWS_SKIP_PREFLIGHT=1\n"
        f"{bar}\n",
        file=sys.stderr,
    )


def _declared_requirements(pyproject: Path) -> set[str]:
    """Core dependency names from *pyproject*. Extras are excluded deliberately.

    An absent extra is a legitimate configuration; an absent core dependency is a
    broken install.
    """
    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    names = set()
    for spec in data.get("project", {}).get("dependencies", []):
        name = re.split(r"[<>=!~\[; ]", spec.strip(), maxsplit=1)[0]
        if name:
            names.add(name)
    return names


def _is_installed(name: str) -> bool:
    installed = False
    with contextlib.suppress(PackageNotFoundError):
        distribution(name)
        installed = True
    return installed


def check_declared_dependencies_installed() -> str:
    """Return an error string when a declared core dependency is not installed."""
    pyproject = APP / "pyproject.toml"
    if not pyproject.is_file():
        return ""  # no mounted source; nothing to compare

    missing = [
        name for name in sorted(_declared_requirements(pyproject)) if not _is_installed(name)
    ]

    if not missing:
        return ""
    listed = "\n".join(f"    - {m}" for m in missing)
    return (
        f"\n  {len(missing)} dependency/ies declared in pyproject.toml are NOT installed\n"
        f"  in this image:\n\n{listed}\n\n"
        "  The image was built before these were declared. Because the working tree\n"
        "  is bind-mounted over /app, the code running here already expects them."
    )


def main() -> int:
    if os.environ.get("GRAPHKNOWS_SKIP_PREFLIGHT") == "1":
        print("preflight: SKIPPED (GRAPHKNOWS_SKIP_PREFLIGHT=1)", file=sys.stderr)
        return 0

    problem = check_declared_dependencies_installed()
    if problem:
        _fail("STALE IMAGE: declared dependencies are missing", problem)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

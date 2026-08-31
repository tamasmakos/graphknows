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

It also checks optional extras: GRAPHKNOWS_IMAGE_EXTRAS records which extras
the image was *built* to carry (set by the Dockerfile), and each one's
distributions must actually *import* — an extra installed out-of-band
(`--no-deps`, no lock entry) can be dropped by a later layer or `uv sync`
prune, or lose a transitive dependency its own `--no-deps` closure needs,
without anything noticing otherwise.
"""

from __future__ import annotations

import contextlib
import importlib
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


def _distribution_name(spec: str) -> str:
    """Strip a PEP 508 requirement spec down to its bare distribution name."""
    return re.split(r"[<>=!~\[; ]", spec.strip(), maxsplit=1)[0]


def _declared_requirements(pyproject: Path) -> set[str]:
    """Core dependency names from *pyproject*. Extras are excluded deliberately.

    An absent extra is a legitimate configuration; an absent core dependency is a
    broken install.
    """
    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    names = {_distribution_name(spec) for spec in data.get("project", {}).get("dependencies", [])}
    names.discard("")
    return names


def _is_installed(name: str) -> bool:
    installed = False
    with contextlib.suppress(PackageNotFoundError):
        distribution(name)
        installed = True
    return installed


def _is_importable(name: str) -> bool:
    """Whether *name* actually imports, not just has metadata on disk.

    An extra installed out-of-band with ``--no-deps`` can have its own
    metadata present while a transitive dependency it needs is missing (e.g.
    the `srl` extra's nlpaug -> gdown chain) — the import raises where mere
    metadata presence would not catch it.
    """
    imported = False
    with contextlib.suppress(ImportError):
        importlib.import_module(name.replace("-", "_"))
        imported = True
    return imported


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


def check_declared_extras_installed() -> str:
    """Return an error string when a claimed extra's distributions are missing.

    GRAPHKNOWS_IMAGE_EXTRAS is a comma-separated list of extras the image was
    built to carry. Unset or empty means the image claims none — a legitimate
    configuration, not a check to perform.
    """
    claimed = [e for e in os.environ.get("GRAPHKNOWS_IMAGE_EXTRAS", "").split(",") if e]
    if not claimed:
        return ""

    pyproject = APP / "pyproject.toml"
    if not pyproject.is_file():
        return ""  # no mounted source; nothing to compare

    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    optional = data.get("project", {}).get("optional-dependencies", {})

    missing: list[str] = []
    for extra in claimed:
        specs = optional.get(extra, [])
        for spec in specs:
            name = _distribution_name(spec)
            if name and not _is_importable(name):
                missing.append(f"{extra}: {name}")

    if not missing:
        return ""
    listed = "\n".join(f"    - {m}" for m in missing)
    return (
        f"\n  {len(missing)} distribution/s from extras this image claims to carry\n"
        f"  (GRAPHKNOWS_IMAGE_EXTRAS) do NOT import:\n\n{listed}\n\n"
        "  The layer that installs this extra was dropped, failed, was pruned by a\n"
        "  later `uv sync`, or lost a transitive dependency its own out-of-band\n"
        "  install needs. The image is silently carrying less than it claims."
    )


def main() -> int:
    if os.environ.get("GRAPHKNOWS_SKIP_PREFLIGHT") == "1":
        print("preflight: SKIPPED (GRAPHKNOWS_SKIP_PREFLIGHT=1)", file=sys.stderr)
        return 0

    problem = check_declared_dependencies_installed()
    if problem:
        _fail("STALE IMAGE: declared dependencies are missing", problem)
        return 1

    problem = check_declared_extras_installed()
    if problem:
        _fail("INCOMPLETE IMAGE: a declared extra is missing", problem)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

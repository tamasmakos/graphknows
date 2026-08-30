"""README install section must tell a `pip install graphknows` user what they get.

The section documents the ArcadeDB requirement only as `docker compose up -d
arcadedb` and never names the setting that points at an ArcadeDB the user
already runs (`GRAPHKNOWS_ARCADEDB_URL`, default `http://localhost:2480` —
graphknows/settings.py). It also tells that pip user to run
`python scripts/bake_models.py`, but the wheel ships only the `graphknows`
package (`[tool.hatch.build.targets.wheel] packages = ["graphknows"]`), so
`scripts/` does not exist after `pip install graphknows` — the instruction is
executable only from a repo checkout. Nothing states that a pip install is a
library plus its console entry points, not a standalone-runnable app without
a reachable graph database.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = REPO_ROOT / "pyproject.toml"


def _install_section() -> str:
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    match = re.search(r"## Install\b(.*?)(?=\n## )", readme, re.DOTALL)
    assert match, "README.md has no '## Install' section followed by another '## ' heading"
    return match.group(1)


def _pyproject() -> dict:
    return tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))


def test_states_arcadedb_url_setting_and_default() -> None:
    section = _install_section()
    assert "GRAPHKNOWS_ARCADEDB_URL" in section, (
        "Install section never names the setting (GRAPHKNOWS_ARCADEDB_URL) that "
        "points the library at an ArcadeDB instance"
    )
    assert "http://localhost:2480" in section, (
        "Install section never states GRAPHKNOWS_ARCADEDB_URL's default "
        "(http://localhost:2480, see graphknows/settings.py field arcadedb_url)"
    )


def test_states_console_entry_points() -> None:
    section = _install_section()
    scripts = _pyproject()["project"]["scripts"]
    assert scripts, "pyproject.toml [project.scripts] is empty"
    missing = [name for name in scripts if name not in section]
    assert not missing, (
        f"Install section never mentions console script(s) {missing} declared in "
        "pyproject.toml [project.scripts], so it doesn't say what a pip install actually gives you"
    )


def test_arcadedb_requirement_is_satisfiable_without_compose() -> None:
    section = _install_section()
    lowered = section.lower()
    assert "arcadedb" in lowered, "Install section never mentions arcadedb"
    phrases = ("existing", "already running", "already have", "reachable")
    assert any(p in lowered for p in phrases), (
        "Install section presents ArcadeDB only via 'docker compose up -d arcadedb' and "
        f"never states it can point at an {phrases[1]}/{phrases[2]} ArcadeDB instead "
        "(no mention of: " + ", ".join(phrases) + ")"
    )


def test_links_are_absolute() -> None:
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    targets = re.findall(r"\[[^\]]*\]\(([^)\s]+)\)", readme)
    relative = [t for t in targets if not t.startswith(("http://", "https://", "mailto:", "#"))]
    assert not relative, (
        'README.md is the PyPI long description (readme = "README.md" in '
        "pyproject.toml); relative links resolve against pypi.org and 404 there. "
        "Internal targets must be absolute GitHub blob URLs under "
        "https://github.com/tamasmakos/graphknows/blob/main/ . Offending "
        f"targets: {relative}"
    )


def test_scripts_path_is_qualified_as_checkout_only() -> None:
    section = _install_section()
    if "scripts/" not in section:
        return
    lowered = section.lower()
    qualifiers = ("clone", "checkout", "repo")
    assert any(q in lowered for q in qualifiers), (
        "Install section references a scripts/ path (e.g. 'python scripts/bake_models.py'), "
        "but the wheel ships only the graphknows package "
        '([tool.hatch.build.targets.wheel] packages = ["graphknows"]), so scripts/ does not '
        "exist after `pip install graphknows` — the section must qualify this as requiring a "
        "repo clone/checkout (no mention of: " + ", ".join(qualifiers) + ")"
    )

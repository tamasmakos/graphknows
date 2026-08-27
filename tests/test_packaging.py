"""Packaging invariants: what the code imports must be what the package declares.

These are the cheap seam for a class of bug that otherwise only ever surfaces as
a ModuleNotFoundError inside a container, weeks later, in a code path nobody
runs locally:

  * `gliner` was imported by the default extraction path and declared nowhere —
    invisible because docker-compose bind-mounts the repo over the image, so
    live source ran against months-old site-packages.
  * `nltk` was imported by the default relation path but declared only in the
    `eval` dependency-group, so `pip install graphknows` produced a broken
    install and only the Docker image (which installed dev+eval groups) worked.
  * Six of seven `MissingExtraError` call sites named extras that do not exist,
    so the remedy they printed installed nothing.

A Docker build is the integration-level check for the same class
(`scripts/preflight.py`, plus `scripts/bake_models.py --check` as the image
HEALTHCHECK); these run in milliseconds.
"""

from __future__ import annotations

import ast
import re
import shutil
import subprocess
import sys
import tomllib
import zipfile
from importlib.metadata import packages_distributions
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
PKG_ROOT = REPO_ROOT / "graphknows"
PYPROJECT = REPO_ROOT / "pyproject.toml"


def _pyproject() -> dict:
    return tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))


def _requirement_names(specs: list[str]) -> set[str]:
    """Normalise PEP 508 specs to comparable distribution names."""
    out = set()
    for spec in specs:
        name = re.split(r"[<>=!~\[; ]", spec.strip(), maxsplit=1)[0]
        if name:
            out.add(name.lower().replace("_", "-"))
    return out


def _declared_distributions() -> set[str]:
    """Every distribution the *installed package* declares: core + all extras."""
    proj = _pyproject()["project"]
    declared = _requirement_names(proj.get("dependencies", []))
    for extra_specs in proj.get("optional-dependencies", {}).values():
        declared |= _requirement_names(extra_specs)
    return declared


def _source_files() -> list[Path]:
    return list(PKG_ROOT.rglob("*.py"))


def _top_level_imports(path: Path) -> set[str]:
    """Top-level module names imported by *path*, including inside functions.

    Lazy imports inside a `try:` or a function body are the norm in this
    codebase (that is how extras are guarded), so a module-header-only scan
    would miss exactly the imports that break at runtime.
    """
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except SyntaxError:  # pragma: no cover - vendored/broken files are excluded
        return set()

    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module:  # skip relative imports
                names.add(node.module.split(".")[0])
    return names


def _is_first_party(mod: str) -> bool:
    return mod == "graphknows" or (PKG_ROOT / mod).exists()


def test_missing_extra_call_sites_name_declared_extras() -> None:
    """Every MissingExtraError must name an extra that actually exists.

    Naming a non-existent extra tells the user to run a pip command that
    installs nothing. For a missing *core* dependency, raise BrokenInstallError.
    """
    declared_extras = set(_pyproject()["project"].get("optional-dependencies", {}))
    pattern = re.compile(r'MissingExtraError\(\s*["\'][^"\']*["\']\s*,\s*["\']([^"\']+)["\']')

    offenders: list[str] = []
    for path in _source_files():
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            for extra in pattern.findall(line):
                if extra not in declared_extras:
                    rel = path.relative_to(REPO_ROOT).as_posix()
                    offenders.append(f"{rel}:{lineno} names extra '{extra}'")

    assert not offenders, (
        "MissingExtraError call sites naming extras that are not declared in "
        f"pyproject.toml (declared: {sorted(declared_extras)}):\n  " + "\n  ".join(offenders)
    )


def test_known_extras_constant_matches_pyproject() -> None:
    """MissingExtraError.KNOWN_EXTRAS must track the real extras."""
    from graphknows.exceptions import MissingExtraError

    declared = set(_pyproject()["project"].get("optional-dependencies", {}))
    assert set(MissingExtraError.KNOWN_EXTRAS) == declared


@pytest.mark.parametrize(
    "distribution",
    [
        # Regression guards: each of these was imported by a default-mode code
        # path while undeclared or declared only in a dev group.
        "gliner",  # entities/gliner_model.py -> knowledgator/gliner-relex-large-v1.0
        "nltk",  # relations/_lingfeatures.py, frames/framenet.py
        "sentence-transformers",  # storage/embedder.py, ranking/rerank.py
        "spacy",  # entities/extractor.py
    ],
)
def test_default_path_dependencies_are_core(distribution: str) -> None:
    """Dependencies of the default (llm_free) path must be core, not extras.

    `pip install graphknows` promises a working llm_free stack. A dependency of
    that path living in an extra or a dependency-group breaks the promise, and
    the Docker image hides it whenever the image happens to install that group.
    """
    core = _requirement_names(_pyproject()["project"].get("dependencies", []))
    assert distribution in core, (
        f"'{distribution}' is imported by a default-mode code path but is not a "
        f"core dependency in pyproject.toml [project.dependencies]"
    )


def test_bundled_default_ontology_ships_in_the_wheel() -> None:
    """The always-on ontology is a data file, and data files are easy to lose.

    ``[tool.hatch.build.targets.wheel] packages = ["graphknows"]`` ships the
    package directory, but hatchling excludes anything the VCS ignores and
    honours any `exclude` entry — either would produce an installed package
    whose default ``ontology_source`` points at a file that is not there.
    """
    from graphknows.settings import GraphKnowsSettings

    asset = Path(GraphKnowsSettings().ontology_source)
    assert asset.is_file(), f"bundled ontology missing: {asset}"
    assert asset.is_relative_to(PKG_ROOT), "asset must live inside the shipped package"

    build = _pyproject()["tool"]["hatch"]["build"]
    assert build["targets"]["wheel"]["packages"] == ["graphknows"]
    assert not build.get("exclude"), "a global exclude could drop the asset"

    # Not VCS-ignored: hatchling's default `ignore-vcs = false` drops such files
    # from the build even though they sit inside the package directory.
    ignored = subprocess.run(
        ["git", "check-ignore", str(asset)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if ignored.returncode == 128:  # e.g. "dubious ownership" inside the container
        pytest.skip("git cannot read this repo here; the VCS check needs a host run")
    assert ignored.returncode == 1, f"gitignored, so hatchling will not ship it: {asset}"


def test_built_wheel_ships_the_default_ontology_assets(tmp_path: Path) -> None:
    """Build a real wheel with `uv build` and assert the ontology assets are
    actually inside the archive — the condition proxy above checks the
    source file exists and is not gitignored, but never proves hatchling's
    packaging config actually puts it in the wheel.

    These three archive paths mirror `_BUNDLED_ONTOLOGY`, `_BUNDLED_OVERLAY`
    and `_BUNDLED_PERSONAL_PROFILE` in graphknows/settings.py. The first two
    are what `GraphKnowsSettings().ontology_source` / `overlay_source`
    resolve to by default; the personal profile is only reached via
    `GRAPHKNOWS_RELATION_ONTOLOGY=personal` (the default relation ontology
    is `_BUNDLED_ONTOLOGY`, same file as `_BUNDLED_RELATION_PROFILE`).
    """
    if shutil.which("uv") is None:
        pytest.skip("uv not on PATH; cannot build a wheel here")

    result = subprocess.run(
        ["uv", "build", "--wheel", "--out-dir", str(tmp_path)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, f"uv build --wheel failed:\n{result.stderr}"

    wheels = sorted(tmp_path.glob("*.whl"))
    assert len(wheels) == 1, f"expected exactly one built wheel in {tmp_path}, found {wheels}"

    names = zipfile.ZipFile(wheels[0]).namelist()
    for expected in (
        "graphknows/symbolic/ontology/assets/cco/cco.json",
        "graphknows/symbolic/ontology/assets/cco/conversational.json",
        "graphknows/symbolic/ontology/assets/personal/personal-profile.json",
    ):
        assert expected in names, f"{wheels[0].name}: missing bundled ontology asset {expected!r}"


def test_every_imported_third_party_module_is_declared() -> None:
    """No graphknows module may import a distribution the package never declares.

    This is the check that would have caught `gliner`. It resolves each imported
    top-level module to its installed distribution and compares against the
    union of core dependencies and all extras.
    """
    declared = _declared_distributions()
    stdlib = sys.stdlib_module_names
    mod_to_dists = packages_distributions()

    undeclared: dict[str, set[str]] = {}
    for path in _source_files():
        rel = path.relative_to(REPO_ROOT).as_posix()
        for mod in _top_level_imports(path):
            if mod in stdlib or _is_first_party(mod) or mod.startswith("_"):
                continue
            dists = {d.lower().replace("_", "-") for d in mod_to_dists.get(mod, [])}
            if not dists:
                # Not installed at all: we cannot map it to a distribution, so
                # fall back to the module name (right for the common case, and
                # a false positive here is still a real thing to look at).
                dists = {mod.lower().replace("_", "-")}
            if not (dists & declared):
                undeclared.setdefault(sorted(dists)[0], set()).add(rel)

    assert not undeclared, "Imported but not declared in pyproject.toml:\n  " + "\n  ".join(
        f"{dist}  <- {', '.join(sorted(files))}" for dist, files in sorted(undeclared.items())
    )

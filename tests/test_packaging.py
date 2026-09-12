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


def _missing_extra_call_sites() -> dict[str, set[str]]:
    """{repo-relative source file: the extras its MissingExtraError calls name}.

    An AST walk rather than a line scan: it survives a call reflowed across
    lines, reads the extra whether it is positional or `extra=`, and never
    matches the extra's name where it only appears in prose.
    """
    sites: dict[str, set[str]] = {}
    for path in _source_files():
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:  # pragma: no cover - vendored/broken files are excluded
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            called = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
            if called != "MissingExtraError":
                continue
            extra = node.args[1] if len(node.args) > 1 else None
            for keyword in node.keywords:
                if keyword.arg == "extra":
                    extra = keyword.value
            if isinstance(extra, ast.Constant) and isinstance(extra.value, str):
                rel = path.relative_to(REPO_ROOT).as_posix()
                sites.setdefault(rel, set()).add(extra.value)
    return sites


#: The extra each MissingExtraError call site must name. Every one is now an RDF
#: path: the decoder's dspy import used to name `assisted`, and once dspy became
#: core (ADR 0003) a missing dspy stopped being an extra to install and became a
#: BrokenInstallError — so the decoder is deliberately absent from this table.
ONTOLOGY_SPLIT_CALL_SITES = {
    "graphknows/symbolic/ontology/rdf_io.py": "ontology",
    "graphknows/symbolic/ontology/rdf/taxonomy.py": "ontology",
    "graphknows/symbolic/ontology/rdf/projection.py": "ontology",
    "graphknows/symbolic/ontology/rdf/prefixes.py": "ontology",
    "graphknows/symbolic/ontology/digest.py": "ontology",
}


def test_ontology_call_sites_name_the_ontology_extra() -> None:
    """The RDF paths must send the user to `ontology`.

    An ontology call site naming anything else tells someone who only wants to
    load their own OWL file to install a stack they asked not to have.
    """
    found = _missing_extra_call_sites()
    actual = {rel: sorted(found.get(rel, set())) for rel in ONTOLOGY_SPLIT_CALL_SITES}
    expected = {rel: [extra] for rel, extra in ONTOLOGY_SPLIT_CALL_SITES.items()}
    assert actual == expected, (
        "MissingExtraError call sites name the wrong extra (or the file moved, "
        f"leaving none to find):\n  expected {expected}\n  found    {actual}"
    )

    # A call site added to the ontology package later must not silently revive
    # `assisted`: the split is a property of the package, not of six lines.
    strays = {
        rel: sorted(extras)
        for rel, extras in found.items()
        if rel.startswith("graphknows/symbolic/ontology/") and extras != {"ontology"}
    }
    assert not strays, f"ontology modules naming an extra other than 'ontology': {strays}"


def test_known_extras_constant_matches_pyproject() -> None:
    """MissingExtraError.KNOWN_EXTRAS must track the real extras."""
    from graphknows.exceptions import MissingExtraError

    declared = set(_pyproject()["project"].get("optional-dependencies", {}))
    assert set(MissingExtraError.KNOWN_EXTRAS) == declared


def test_ontology_extra_excludes_dspy() -> None:
    """`[ontology]` gives RDF support without the LLM-extraction stack.

    This is the whole reason the extra exists: a client who brings their own OWL
    ontology must not be made to reach for the LLM stack to parse a file. It
    stays true now that dspy is core — the extra carries the RDF stack and
    nothing else, so what it names is still only what an RDF user needs.
    """
    extras = _pyproject()["project"].get("optional-dependencies", {})
    assert "ontology" in extras, "the `ontology` extra is not declared"

    ontology = _requirement_names(extras["ontology"])
    assert {"rdflib", "networkx"} <= ontology, (
        f"[ontology] must carry the RDF stack: {sorted(ontology)}"
    )
    assert "dspy" not in ontology, f"[ontology] must not pull the LLM stack: {sorted(ontology)}"


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
    resolve to by default; the personal profile is `_BUNDLED_RELATION_PROFILE`
    — the default relation vocabulary — and is also reached explicitly via
    `GRAPHKNOWS_RELATION_ONTOLOGY=personal` (`=cco` reaches `_BUNDLED_ONTOLOGY`
    instead).
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


def test_pack_data_ships_in_the_wheel(tmp_path: Path) -> None:
    """Domain packs are JSON data, and data files are easy to lose.

    `[tool.hatch.build.targets.wheel] artifacts` declares them explicitly, so a
    future ignore rule matching `*.json` cannot silently produce an installed
    package whose pack loader has nothing to load. The archive check is what
    proves the declaration reaches the wheel.
    """
    wheel_target = _pyproject()["tool"]["hatch"]["build"]["targets"]["wheel"]
    assert "graphknows/packs/data/**/*.json" in wheel_target.get("artifacts", []), (
        "pack data is not declared to the build backend"
    )

    packs = sorted((PKG_ROOT / "packs" / "data").rglob("*.json"))
    assert packs, "no pack data files on disk to ship"

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

    names = set(zipfile.ZipFile(wheels[0]).namelist())
    for pack in packs:
        expected = pack.relative_to(REPO_ROOT).as_posix()
        assert expected in names, f"{wheels[0].name}: missing pack data {expected!r}"


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


def test_assisted_extra_removed_and_dspy_core() -> None:
    """`llm_assisted` is contract surface, so its stack is core and has no extra.

    An extra that gates half the contract surface is a documentation bug waiting
    to happen: the mode is documented, the install that runs it is a footnote,
    and the error a user meets is an ImportError from inside a library call
    (ADR 0003). Three things must hold together — remove any one and the mode
    becomes optional again:
    """
    project = _pyproject()["project"]
    extras = project.get("optional-dependencies", {})

    assert "assisted" not in extras, (
        f"the `assisted` extra is back: {sorted(extras)} — dspy is core now, so an "
        "extra naming it can only install what is already installed"
    )
    assert "ontology" in extras, "removing `assisted` must not take `ontology` with it"

    core = _requirement_names(project.get("dependencies", []))
    for distribution in ("dspy", "litellm"):
        assert distribution in core, (
            f"'{distribution}' is imported by the LLM decoder but is not a core "
            f"dependency — `pip install graphknows` would not run llm_assisted"
        )

    specs = {
        re.split(r"[<>=!~\[]", spec, maxsplit=1)[0].strip().lower(): spec
        for spec in project.get("dependencies", [])
    }
    assert ">=" in specs["dspy"], (
        f"dspy is declared without a lower bound ({specs['dspy']!r}): the decoder "
        "pins a JSONAdapter and calls litellm.register_model, neither of which "
        "every past dspy has"
    )

    named = {extra for extras_ in _missing_extra_call_sites().values() for extra in extras_}
    assert "assisted" not in named, (
        "a MissingExtraError call site still names `assisted`, so it sends the user "
        "to `pip install graphknows[assisted]` — which now fails outright"
    )


@pytest.mark.slow
def test_bare_install_runs_decoder(tmp_path: Path) -> None:
    """A no-extras install of the built wheel runs the LLM decoder end to end.

    The metadata assertions above can only see what pyproject *declares*. This
    one catches what it *omits*: an undeclared transitive import on the decoder
    path is invisible in this repo's own environment, where every dev and eval
    group is installed, and shows up for the first time on a user's machine.
    So the wheel is built, installed alone, and driven against a stub provider —
    no network, no database, no extras.
    """
    if shutil.which("uv") is None:
        pytest.skip("uv not on PATH; cannot build or install a wheel here")

    dist = tmp_path / "dist"
    build = subprocess.run(
        ["uv", "build", "--wheel", "--out-dir", str(dist)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert build.returncode == 0, f"uv build --wheel failed:\n{build.stderr}"
    wheels = list(dist.glob("*.whl"))
    assert wheels, f"no wheel produced in {dist}"

    venv = tmp_path / "venv"
    assert (
        subprocess.run(
            ["uv", "venv", str(venv)], capture_output=True, text=True, check=False
        ).returncode
        == 0
    ), "uv venv failed"
    python = venv / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    install = subprocess.run(
        ["uv", "pip", "install", "--python", str(python), str(wheels[0])],
        capture_output=True,
        text=True,
        check=False,
    )
    assert install.returncode == 0, f"installing the bare wheel failed:\n{install.stderr}"

    # Run from tmp_path, not the repo: `.env` and the source tree are both on the
    # doorstep here, and either would let the check pass on something the wheel
    # does not actually carry.
    drive = subprocess.run(
        [str(python), "-c", _BARE_DECODER_SCRIPT],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert drive.returncode == 0, (
        f"the bare install could not run the decoder:\n{drive.stdout}\n{drive.stderr}"
    )
    assert "DECODED 2" in drive.stdout, f"decoder returned nothing usable:\n{drive.stdout}"


#: Driven in the bare venv's interpreter, so it may import nothing from this repo
#: and defines its own stub provider rather than reusing the one in
#: tests/extraction/test_llm_decoder.py.
_BARE_DECODER_SCRIPT = """
import json
import dspy
from graphknows.ingestion.extraction.llm.decoder import LLMDecoder
from graphknows.settings import GraphKnowsSettings

REPLY = json.dumps(
    {
        "entities": [
            {"surface": "Melanie", "label": "person"},
            {"surface": "Acme", "label": "organization"},
        ],
        "relations": [
            {
                "head": "Melanie",
                "predicate": "worksFor",
                "tail": "Acme",
                "evidence": "Melanie works for Acme",
                "confidence": 0.9,
            }
        ],
        "frames": [],
    }
)


class Stub(dspy.LM):
    def __init__(self):
        super().__init__(model="openrouter/stub/decoder-v1", api_key="stub", cache=False)

    def __call__(self, prompt=None, messages=None, **kwargs):
        return [REPLY]


decoder = LLMDecoder(GraphKnowsSettings(), lm=Stub())
result = decoder.extract(
    "Melanie works for Acme.",
    ["person", "organization"],
    {"worksFor": "the subject is employed by the object"},
)
print("DECODED", len(result.entities))
"""

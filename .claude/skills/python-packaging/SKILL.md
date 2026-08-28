---
name: python-packaging
description: How to package and distribute Python — pyproject.toml metadata, build backends, sdists and wheels, versioning, dependency specifiers, and publishing to PyPI. Use when the user wants to package a project, publish or release to PyPI, pick or configure a build backend, write or fix pyproject.toml, ship binary extensions, or freeze an application; and when another skill needs the packaging vocabulary.
---

Two artifacts, always publish both: the **sdist** (`.tar.gz`, source + `PKG-INFO`, needs a build step) and the **wheel** (`.whl`, a ZIP with `.dist-info/`, unpacked straight into place). pip prefers wheels; a pure-Python project still ships one — `py3-none-any`.

Two roles, kept apart by PEP 517: the **frontend** the user runs (`build`, `pip`) and the **backend** it delegates to (`hatchling`, `setuptools`, …). You never call a backend, and you never call `setup.py`.

## The three branches

- **Publishing a library or tool** — the default path below.
- **Binary extensions** (C/C++/Fortran/Rust) — the default path, plus [`WHEEL-TAGS.md`](WHEEL-TAGS.md) for the compatibility-tag matrix, `abi3`, manylinux/musllinux, and cibuildwheel.
- **Shipping an application** — a different game entirely; read [`APPLICATIONS.md`](APPLICATIONS.md).

Full `[project]` key table and the PyPA spec-to-PEP index live in [`REFERENCE.md`](REFERENCE.md).

## The default path

1. Build from a tagged VCS checkout.
2. Write `pyproject.toml`: `[build-system]` (PEP 518) + `[project]` (PEP 621) + any `[tool.*]`.
3. Pick a layout — prefer **src layout** (`src/pkg/`) *for new projects*. It forces `pip install -e .` for dev, which is the point: cwd sits first on `sys.path`, so a flat layout silently imports the in-tree copy and whatever else is lying around. An existing flat-layout repo stays flat — never migrate layout as a side effect of another change; its hazard is managed by testing the built wheel from outside the source tree.
4. `python -m build` → sdist + wheel in `dist/`.
5. Inspect the wheel: `.dist-info/METADATA` and `RECORD`. `twine check` for README rendering.
6. Rehearse on **TestPyPI** (`twine upload --repository testpypi dist/*`), then PyPI.
7. Authenticate with **Trusted Publishing** (OIDC, no stored secret) from CI — GitHub Actions, GitLab CI/CD, Google Cloud, ActiveState. API tokens otherwise.

### Choosing a backend

`[build-system]` needs `requires` + `build-backend`. PyPA recommends no single one.

```toml
[build-system]
requires = ["hatchling >= 1.26"]      # pure-Python, extensible
build-backend = "hatchling.build"
```

Others: `setuptools.build_meta` (general, C/C++), `flit_core.buildapi` (minimal, no plugins), `pdm.backend`, `uv_build`, `poetry.core.masonry.api` (uses `[tool.poetry]`, not `[project]`). Extensions: `meson-python`, `scikit-build-core` (CMake), `maturin` (Rust). Never `distutils` — removed from the stdlib in 3.12.

### Metadata that bites

- Prefer **static** values. Anything the backend computes goes in `dynamic = [...]`; `name` may never appear there.
- `license` is an **SPDX expression string** (PEP 639) — `"MIT AND (Apache-2.0 OR BSD-2-Clause)"`. The table form and the `License ::` classifiers are deprecated as of Metadata 2.4.
- `requires-python = ">=3.9"` is what actually gates installs. Classifiers only drive search and browse.
- Single-source the version: from the VCS tag, from `pyproject.toml`, or from source the backend reads. Add a test asserting `pkg.__version__ == importlib.metadata.version("dist")`.

## Versions

`[N!]N(.N)*[{a|b|rc}N][.postN][.devN][+local]` (PEP 440). Ordering inside a release: `.devN` < `aN` < `bN` < `rcN` < final < `.postN`.

Post-releases are for non-code fixes, never bugfixes. Local versions (`1.0+ubuntu.1`) are for downstream patches and PyPI rejects them. Epochs (`1!1.0`) are discouraged — jump to a higher number instead. pip ignores pre-releases unless `--pre`. Comparisons normalize case, a `v` prefix, and `._-` separators; `~=` is compatible-release. To retire a bad release without deleting it, **yank** it (PEP 592).

## Dependencies: abstract vs concrete

`[project].dependencies` is **abstract** — names and bounds only (`A>=1,<2`), no pins, no sub-dependencies, no index. A requirements/lock file is **concrete** — exhaustive pins for one environment, and the only place `--index-url` belongs.

Specifier grammar (PEP 508): `name [extras] version-specifier ; marker`, or `name @ URL`.

- **Extras** = optional dependency sets published in the metadata (`requests[security]`).
- **Dependency Groups** (PEP 735, `[dependency-groups]`) = dev/test/lint requirements that stay local, never reach the built metadata, and support `{include-group = "..."}`. Reach for these over a `dev` extra.
- Markers: `python_version`, `sys_platform`, `platform_machine`, `implementation_name`, `extra`, joined with `and`/`or`.
- Reproducibility: pip-tools / Pipenv, or `pylock.toml` (PEP 751); hash-checking mode for secure installs.

## Retired practices

| Reach for | Not |
|---|---|
| `python -m pip install .` | `python setup.py install` |
| `python -m pip install -e .` | `python setup.py develop` |
| `python -m build` | `setup.py sdist` / `bdist_wheel` |
| twine, Trusted Publishing | `setup.py upload` (insecure) |
| pip | `easy_install` |
| wheels | eggs (PyPI stopped accepting `.egg` on 2023-08-01, PEP 715) |
| `[build-system].requires` | `setup_requires` |

`setup.py` and `setup.cfg` remain valid *configuration*; only invoking `setup.py` as a CLI is deprecated. Keep a `setup.py` solely for programmatic config, such as C extension definitions.

## Elsewhere

- Wheels carry only what gets installed — never tests or docs; sdists may carry both.
- Entry points go in `[project.scripts]` / `[project.gui-scripts]` / `[project.entry-points."group"]` as `name = "module:function"`; installers write wrappers into the scripts dir. `gui-scripts` differs only on Windows. Plugins discover each other through `importlib.metadata`.
- Namespace packages: native PEP 420 — omit `__init__.py`. The `pkgutil`/`pkg_resources` styles are legacy and cannot be mixed.
- A **PEP 668** `EXTERNALLY-MANAGED` system Python blocks pip on purpose. Work in a venv.
- Project names compare case-insensitively with runs of `._-` collapsed: `re.sub(r"[-_.]+", "-", name).lower()`. Artifact filenames use `_`.
- Multi-version testing: tox or nox. Publishing from CI: `pypa/gh-action-pypi-publish` on a tag.

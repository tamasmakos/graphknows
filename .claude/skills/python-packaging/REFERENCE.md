# Lookup tables

## Every `[project]` key

| Key | Purpose | Notes |
|---|---|---|
| `name` | PyPI project name | Required; the only key that can never be dynamic |
| `version` | PEP 440 version | Required; static or dynamic |
| `description` | One-line summary | → `Summary` |
| `readme` | Long description | `.md` → GFM, `.rst` → reStructuredText |
| `requires-python` | e.g. `>=3.9` | Gates installs; avoid upper bounds |
| `license` | SPDX expression string (PEP 639) | Table form deprecated |
| `license-files` | Glob list, e.g. `["LICEN[CS]E*"]` | |
| `authors` / `maintainers` | Lists of `{name, email}` | |
| `keywords` | Search keywords | |
| `classifiers` | Trove classifiers | `Private :: Do Not Upload` blocks PyPI upload |
| `urls` | Labeled project URLs | Well-known labels (PEP 753) get special rendering |
| `scripts` | Console entry points | `name = "pkg:func"` |
| `gui-scripts` | GUI entry points | Differs from `scripts` only on Windows |
| `entry-points` | `[project.entry-points."group"]` plugins | |
| `dependencies` | Dependency specifiers | → `Requires-Dist` |
| `optional-dependencies` | Extras | → `Requires-Dist` / `Provides-Extra` |
| `import-names` | Import names provided (PEP 794) | |
| `import-namespaces` | Namespaces contributed (PEP 794) | |
| `dynamic` | Keys the backend fills | `name` must not appear |

List and table keys may be both static and dynamic — a backend may only *append* (PEP 808).

## Sdist contents

`.tar.gz` with a single top-level `{name}-{version}/` holding `pyproject.toml` and an email-format `PKG-INFO` (Core Metadata ≥ 2.2, generated — never hand-written). Nothing else is required; tests and docs are allowed. Files or links resolving outside the destination, and device files/pipes, are invalid; leading slashes are dropped and `..` components discouraged.

## What the backend ships (hatchling)

By default hatchling ships only the Python package tree; everything else must be declared.

```toml
[tool.hatch.build.targets.wheel]
packages = ["graphknows"]              # which import package(s) go in the wheel
artifacts = ["*.json"]                 # gitignored files that must ship anyway
force-include = { "data/onto.owl" = "graphknows/data/onto.owl" }  # explicit extra files

[tool.hatch.build.targets.sdist]
include = ["graphknows", "tests", "README.md"]
exclude = ["*.tmp"]
```

- Non-`.py` files inside the package tree ship only if not gitignored (hatchling respects .gitignore); `artifacts` overrides that.
- `py.typed` must sit inside the package directory to ship — verify with `python -m zipfile -l dist/*.whl`.
- A new import package directory needs `__init__.py`, or it may silently not ship as expected.
- Extras (`[project.optional-dependencies]`) reach the built metadata; `[dependency-groups]` (PEP 735) stay local and never ship — dev/test tools belong there.

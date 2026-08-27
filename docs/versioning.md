# Versioning & deprecation policy

graphknows follows [Semantic Versioning](https://semver.org). From 1.0.0 the
version is single-sourced from `graphknows/_version.py` (`__version__`) — the
file `[tool.hatch.version]` in `pyproject.toml` points the build backend at,
and the only place the version is written by hand. `graphknows/__init__.py`
merely re-exports it.

## What "public" means

Semver applies to the **documented public surface** only:

- the names in `graphknows.__all__` (the package root);
- `graphknows.integrations.client` (the out-of-process SDK);
- `graphknows.integrations.*` (framework adapters);
- the `graphknows-mcp` CLI entry point and the MCP tool contract.

Everything else — including underscore-free names inside internal modules such
as `graphknows.memory.*`, `graphknows.runtime` internals, and the ArcadeDB
store classes — is implementation detail and may change in any release. The
public surface is pinned by `tests/api/test_public_surface.py`.

## Deprecation

A deprecated public name keeps working for **at least one minor release** and
emits a `DeprecationWarning` pointing at the replacement, then is removed in the
next major. Current deprecations (removed in 2.0):

| Deprecated | Use instead |
| --- | --- |
| `graphknows.GraphKnowsRuntime` | `graphknows.Memory` |
| `graphknows.ports.*` | `graphknows.storage.arcadedb.graph_store` (`GraphStore`) |
| `graphknows.config.llm._get_openrouter_api_key` | `graphknows.llm.get_openrouter_api_key` |

## Releasing

1. Bump `__version__` in `graphknows/_version.py`.
2. Update `CHANGELOG.md` (Keep a Changelog format) with an entry for the new version.
3. `make ci` (lint, typecheck, import-linter, unit tests, wheel size + py.typed).
4. `make eval-smoke` against a live ArcadeDB for end-to-end acceptance.
5. Commit, then `git tag vX.Y.Z` and push the tag: `git push origin vX.Y.Z`.

Pushing the tag is what triggers `.github/workflows/release.yml` — the rest is
automated. It reruns the full CI gate (`ci.yml`, called by reference), then
asserts that the pushed tag
matches the `__version__` literal in `graphknows/_version.py`. If they
disagree, the run fails before any upload, reporting both values. Only then
does it publish the wheel and sdist that CI itself built — no separate build
step, no hand-run `uv build` from a laptop.

### One-time: Trusted Publisher registration

Before the first tagged release can publish, a maintainer must register this
repository as a Trusted Publisher in PyPI's web interface (project name, this
repository, workflow filename `release.yml`, environment field left empty).
This is maintainer configuration outside the repository — nothing here can
verify it has been done. Until it has, the publish step fails with an OIDC
identity error; that error means the registration is missing, not that the
workflow is broken.

### Recovering from a failed release

PyPI refuses to accept a re-upload of a filename that already exists, and a
pushed tag is not cleanly re-pushable. Recover **fix-forward**: bump to the
next version, add a CHANGELOG entry, and tag again. Never delete and re-push
a tag, and never attempt to re-upload a filename — neither is recoverable
the way a normal bump-and-retag is.

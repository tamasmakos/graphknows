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
next major. There are no current deprecations.

### Removed in 2.0

| Removed | Use instead |
| --- | --- |
| `graphknows.GraphKnowsRuntime` | `graphknows.Memory` |
| `graphknows.ports.*` | `graphknows.storage.arcadedb.graph_store` (`GraphStore`) |
| `graphknows.llm._get_openrouter_api_key` | `graphknows.llm.get_openrouter_api_key` |

## Releasing

1. Bump `__version__` in `graphknows/_version.py`.
2. Update `CHANGELOG.md` (Keep a Changelog format) with an entry for the new version.
3. `make ci` (the pre-push gate — ruff, `mypy --strict`, import-linter, bandit,
   unit tests at the coverage floor, dependency audit — plus wheel size + py.typed).
4. `make eval-smoke` against a live ArcadeDB for end-to-end acceptance.
5. Commit, then `git tag vX.Y.Z` and push the tag: `git push origin vX.Y.Z`.

Pushing the tag is what triggers `.github/workflows/release.yml` — the rest is
automated. It reruns the full CI gate (`ci.yml`, called by reference), then
asserts that the pushed tag
matches the `__version__` literal in `graphknows/_version.py`. If they
disagree, the run fails before any upload, reporting both values. Only then
does it attach the wheel that CI itself built — no separate build step, no
hand-run `uv build` from a laptop.

### Where a release goes

**Nowhere public.** The wheel is attached as an asset on this private
repository's GitHub Release for the pushed tag; it is not published to PyPI or
any other public index. The Release is the private channel, and access to the
repository is what gates access to the artifact.

Two assets are attached:

| Asset | What it is |
| --- | --- |
| `graphknows-<version>-py3-none-any.whl` | the library, the same bytes CI tested |
| `graphknows-deploy-<version>.tar.gz` | the deployment bundle: `deploy/` and `scripts/` beside the wheel, in the repository's own layout, so `deploy/Dockerfile` builds identically from the unpacked bundle or from a checkout |

The bundle exists because a recipient must be able to build the deployment
without access to this repository. `scripts/` ships in no wheel, so the bundle
carries it rather than the client cloning for three files.

There is no one-time registration step. Attaching a Release asset needs only
`contents: write`, scoped to the publish job, and no stored token: earlier
releases used PyPI's Trusted Publisher over OIDC, and that machinery is gone
along with the public index it served.

### Recovering from a failed release

A pushed tag is not cleanly re-pushable. Recover **fix-forward**: bump to the
next version, add a CHANGELOG entry, and tag again. Never delete and re-push a
tag — it is not recoverable the way a normal bump-and-retag is.

Re-running the upload for the *same* tag is safe (assets are attached with
`--clobber`), which is the one thing that got easier: a public index would have
refused a re-upload of a filename it had already seen.

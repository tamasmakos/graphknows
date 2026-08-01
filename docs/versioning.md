# Versioning & deprecation policy

graphknows follows [Semantic Versioning](https://semver.org). From 1.0.0 the
version is single-sourced from `graphknows/__init__.py` (`__version__`).

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

1. Update `CHANGELOG.md` (Keep a Changelog format) and bump `__version__`.
2. `make ci` (lint, typecheck, import-linter, unit tests, wheel size + py.typed).
3. `make eval-smoke` against a live ArcadeDB for end-to-end acceptance.
4. Tag `vX.Y.Z`; optionally publish to TestPyPI before PyPI.

# Contributing to GraphKnows

GraphKnows is a single Python package (`graphknows/`) backed by ArcadeDB. There
is no monorepo, no web UI and no TypeScript. If you are looking for the shape of
the system before changing it, read [docs/services.md](docs/services.md).

## Development setup

**Prerequisites**: Python 3.11+, [uv](https://docs.astral.sh/uv/), Docker and
Docker Compose v2.

```bash
git clone https://github.com/your-org/graphknows.git
cd graphknows

make setup          # copies .env.example → .env if missing
uv sync             # install the package + dev tooling
pre-commit install                             # commit hooks
pre-commit install --hook-type pre-push        # pre-push gate
```

GraphKnows needs a running ArcadeDB:

```bash
make infra-up       # start only ArcadeDB
# or
make up             # start the full compose stack
```

`docker-compose.yaml` is the **development** stack — it mounts the working tree
over the image and puts it first on `PYTHONPATH`, so the container runs live
source against image dependencies.

Models (GLiNER2, sentence-transformers, spaCy, NLTK) download on first use, into
a compose named volume so a rebuild never re-downloads them. `python
scripts/bake_models.py` warms the volume ahead of time.

## Quality gates

`make ci` is the full local gate and mirrors CI: it runs `scripts/gate.sh`
(via `make gate`) and then `make build-wheel`. The targets below are the
narrower quick checks it supersedes:

| Command | Checks |
| --- | --- |
| `make lint` | `ruff check graphknows tests` |
| `make typecheck` | `mypy graphknows` |
| `make arch` | `lint-imports` — the layer contracts in `.importlinter` |
| `make test` | unit suite (`pytest -m "not integration"`) |
| `make build-wheel` | the wheel stays small and ships `py.typed` |

`make gate` runs the pre-push hook exactly as the hook runs it, both locally
and in CI: a wider ruff scope (`graphknows evaluation tests`, not just
`graphknows tests`), plus `mypy --strict`, `bandit`, a dependency audit and a
65% coverage floor that `make lint` and `make test` above don't cover. `make
lint` and `make test` are quick local checks, not a substitute for `make gate`
before you push.

### The merge gate

A green `make gate` alone is not acceptance for a change to retrieval, ingestion
or the packs: it says the code is clean, not that memory got better. The merge
gate is all three of

```bash
make gate
python -m evaluation panel --compare evaluation/results/baseline
python -m evaluation deadweight
```

The panel is the evidence, and no single row is the headline: the comparison
fails if any row falls below `median − band`, if identity precision dropped, or
if the run left dead weight — a layer that was written but never read.

Integration tests need a live ArcadeDB and are excluded from the unit run:

```bash
make infra-up
make test-integration
```

Test layout mirrors the package (`tests/api/`, `tests/ingestion/`,
`tests/retrieval/`, `tests/storage/`, …). `pyproject.toml`'s
`[tool.pytest.ini_options]` is authoritative: `testpaths = ["tests"]`,
`asyncio_mode = "auto"`, markers `unit` and `integration`. Ruff's config lives
in `[tool.ruff]`, same file.

A diff coverage check now runs in CI on pull requests, comparing the branch against
the base branch; it is report-only until its threshold is set from measured
numbers. One known gap remains: there is no secret scan at all now that the
analysis CLI that ran one is gone.
[gitleaks](https://github.com/gitleaks/gitleaks) is the candidate replacement.

A dependency-drift check (`deptry`) now runs in both `make gate` and CI,
report-only for now. Its suppressions live in
`[tool.deptry.per_rule_ignores]` in `pyproject.toml`, and every one must
carry a dated rationale — if that ignore list grows without corresponding
fixes, the tool is deleted. This is not a new policy: `import-linter` is the
only architecture checker left for exactly this reason, after a second one
(`tach`) was deleted for the same failure mode, recorded in `pyproject.toml`'s
`import-linter` comment: "a gate that documents its own bypass is not a gate."

## Architecture rules

The layer stack in [`.importlinter`](.importlinter) is enforced, not advisory —
`make arch` fails the build on a violation. Two contracts are easy to trip over:

- **`ingestion` and `retrieval` must not import each other.** They meet only at
  `storage`.
- **`integrations.client` stays stdlib-only.** It may not import `ingestion`,
  `retrieval`, `storage`, `channels` or `memory`, so consumers can drive the MCP
  server without the ML dependency tree.

Adding a package means adding it to the stack in `.importlinter`. See
[docs/services.md](docs/services.md) for what each layer owns.

## Extension points

### Adding a document parser

There is one parser — `PlainTextParser` in
`graphknows/ingestion/parsers/text.py` — and it handles plain text and Markdown,
which is what the ingest path feeds it. It is a plain class, not a registered
plugin: there is no `BaseParser` ABC, no extension registry and no
auto-discovery, because one implementation never needed them.

To support another format, add a function to that module that turns the source
into `TextSegment`s and call it from `parse()`. Reach for a second class only
once a second format actually exists.

### Adding a retrieval channel

A **channel** is one memory signal with symmetric hooks: `ingest()` to tag a
chunk on the write path, `collect()` to contribute ranked candidates on the read
path. A channel that only writes is dead weight — if nothing consumes it, do not
add it.

1. Implement the `Channel` protocol from `graphknows/channels/base.py`;
   `collect(ctx, rt, top_k)` returns `{chunk_id: (score, info)}`.
2. Register it in `graphknows/channels/registry.py::default_channels`, gated on
   the setting that enables it. Insertion order defines fusion order.
3. Measure it. Channels are added on evidence recall, not on principle — both the
   topic and PageRank channels were removed after measurement showed they did not
   pay.

### Adding an MCP tool

Tools live in `graphknows/server/mcp/tools/`, grouped by concern (`stm.py`,
`query.py`, `ltm.py`, `corpus.py`, `admin.py`).

1. Add an `async def` to the right module, decorated on the shared FastMCP app.
2. Accept an optional `namespace` argument — every tool does — and resolve it
   through the standard path in `graphknows/server/mcp/_state.py`.
3. Keep it a thin wrapper over one `Memory` method. Logic belongs in `memory` or
   below, never in the transport.
4. Export it from `tools/__init__.py`, which is the authoritative inventory.

## Code style

- **Formatting and lint**: `ruff` + `ruff format`. Run `make lint`.
- **Types**: type hints everywhere; `make typecheck` must stay clean.
- **ArcadeDB**: Cypher has no DDL surface, so schema is SQL. Embedding values
  must be inline SQL literals — LIST params conflict with the `LSM_VECTOR`
  index — and openCypher has no `IN $list`, so **every interpolated string goes
  through `graphknows/storage/arcadedb/_sql.py`**. Never hand-format a value
  into a query.
- **Async**: the public surface is async; `Memory` is an async context manager.
- **Errors**: no silent fallbacks. A failing channel raises rather than quietly
  degrading recall.
- **Tests**: add a test for every new parser, channel or tool.

## Public surface and releases

Semver applies to the documented public surface only, and it is pinned by
`tests/api/test_public_surface.py`. Read
[docs/versioning.md](docs/versioning.md) before changing anything exported from
`graphknows.__all__`, `graphknows.integrations.*`, or the MCP tool contract.

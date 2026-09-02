# Phase 0 Research: Air-Gapped Delivery

Every finding below was verified against the tree at `f8d362d` on the date of writing, not taken
from a prior design document. Three earlier design attempts failed because they inherited claims
from a docstring or a run report without checking them; R1 and R2 are exactly those claims, and
both were false.

## R1 — `scripts/` is not in the wheel

**Decision**: The release image installs the library from the wheel **and separately `COPY`s**
`scripts/bake_models.py`, `scripts/preflight.py` and `scripts/docker-entrypoint.sh` from the build
context.

**Rationale**: `pyproject.toml` declares `[tool.hatch.build.targets.wheel] packages = ["graphknows"]`.
Nothing under `scripts/` ships. An image built only from the wheel therefore has no bake script and
no entrypoint — the earlier design's `RUN python scripts/bake_models.py` would fail on a path that
does not exist.

**The build context is the release bundle.** FR-010 promises a build from the artifact and the
deployment definition *alone*, so "the build context" must not mean "a checkout of this repo". The
release attaches a bundle whose layout **mirrors the repository**: `deploy/` and `scripts/` both at
its root, plus the wheel. The same `deploy/Dockerfile` then works unchanged from either a bundle or
a checkout, because the relative paths are identical — no second Dockerfile, and no copy of any
script committed twice.

**Alternatives considered**: Adding `scripts/` to the wheel. Rejected — `tests/test_readme_install.py`
pins wheel contents, and shipping operational scripts to every installing user to serve one
deployment path is exactly the "build only what was asked for" failure. Committing duplicate copies
under `deploy/` was also rejected: two files that must stay identical are a sync problem, and the
bundle can simply carry the originals.

## R2 — Nothing checks baked models at boot today

**Decision**: Add one guarded step to `scripts/docker-entrypoint.sh`, running
`bake_models.py --present` only when `GRAPHKNOWS_REQUIRE_BAKED=1`.

**Rationale**: Verified — `docker-entrypoint.sh` runs `python preflight.py` and then `exec "$@"`,
nothing else. `grep` finds exactly one caller of `bake_models.py` in the repo, in the `Makefile`.
`preflight.py` cannot serve as the guard: it early-returns when there is no bind-mounted source,
which is always true inside an image. So the "fails loudly on missing weights" guarantee has **no
implementation** and must be built. The claim that it exists comes from `bake_models.py`'s own
docstring and is wrong.

**Why guarded**: The dev image shares this entrypoint and holds no weights at first start — they
arrive lazily into a named volume. An unguarded check would break the dev container's cold boot,
which is a working setup that this feature has no mandate to disturb.

**Why `--present` and not `--check`**: `--present` is a files-on-disk test measured in seconds;
`--check` deep-loads every model. `_present()`'s own docstring records a ~5× ingest tax when a
model probe runs on a health-check duty cycle. `set -e` is already active in the entrypoint, so a
non-zero exit aborts before `exec "$@"` with no extra plumbing.

## R3 — `_CORE_DDL` is a list of tuples, not strings

**Decision**: Derive `SCHEMA_VERSION` from the **statement text only**, e.g. a short digest over
`"\n".join(stmt for _lang, stmt in _CORE_DDL)`, and include the embedding dimension in the digest
input.

**Rationale**: `_schema.py` declares `_CORE_DDL: list[tuple[str, str]]`, entries shaped
`("sql", "CREATE VERTEX TYPE SESSION IF NOT EXISTS")`. A digest over `"\n".join(_CORE_DDL)` — as
earlier design notes proposed — raises `TypeError`. Including `dims` is not incidental: the DDL
interpolates it into the `LSM_VECTOR` indexes, so a graph built at one embedding dimension genuinely
cannot serve another. A stamp that ignored `dims` would let exactly that mismatch through.

**Alternatives considered**: A hand-maintained integer version. Rejected — it relies on someone
remembering to bump it on every DDL edit, and the failure mode of forgetting is silent corruption,
which Principle V forbids. A content digest cannot be forgotten.

**Known consequence, to be stated in the ticket**: a whitespace or comment edit inside a DDL
statement changes the version and strands existing client graphs. That is acceptable only because
upgrades are re-ingest by decision (FR-022), but a reviewer must know that editing `_CORE_DDL` is
now a client-visible event.

## R4 — `ensure_schema` is the single seam

**Decision**: Read and write the stamp inside `GraphStore.ensure_schema`, after `_apply_ddl`.

**Rationale**: Verified — `ensure_schema(dims)` calls `_apply_ddl(_CORE_DDL, dims=dims)` and then
clears caches. Its production callers are `memory.py:155` (via `_build_retriever`) and
`ingestion/stm/service.py:77`, which between them cover every path that opens a namespace's
database for reading or writing. One guard at this seam beats two guards at the callers.

**Sequencing is forced**: `_apply_ddl` creates the database before running DDL, so the stamp cannot
be read before the database exists. Order must be: apply DDL (idempotent, as today) → read stamp →
absent means insert, present-and-equal means proceed, present-and-different means raise.

**Open edge the ticket must handle**: `graphknows/symbolic/ontology/rdf/store.py` has its own
`ensure_schema(client, db)`, and `graphknows/storage/__init__.py` builds stores directly. The
single-seam claim holds for `Memory` and the STM service; whether the ontology store path also needs
the guard was **not** verified here and must be checked by the implementer rather than assumed.

## R5 — The egress allowlist must include the database host

**Decision**: The offline test's socket guard allows loopback **plus the host resolved from
`GRAPHKNOWS_ARCADEDB_URL`**, and blocks everything else.

**Rationale**: Under compose the database is reached at `http://arcadedb:2480`, which is not
loopback. An allowlist of `127.0.0.1`/`::1` alone would block the database and make the test fail
for a reason that has nothing to do with egress.

**Known limitation to record in the ticket**: the guard resolves that host at test setup, so the
DNS lookup happens before the block is installed. On a genuinely DNS-less host the test could pass
locally and behave differently in the field. Worth a comment at the call site rather than silent.

**Alternatives considered**: `pytest-socket`. Rejected — a monkeypatched `socket.socket.connect` is
about fifteen lines of stdlib and adds no dependency, per Principle IV.

## R6 — A `## Verify` must name a test node, never an existing file

**Decision**: Acceptance commands point at a **new test node** in an existing file
(`tests/test_packaging.py::test_ontology_extra_excludes_dspy`), or at a file that does not exist yet
(`tests/storage/test_schema_stamp.py`, `tests/test_offline.py`, `deploy/acceptance.sh`).

**Rationale**: This is not a style note; it is what caused three prior ticket-drafting attempts to
be refused. `pytest tests/test_packaging.py -q` passes today (8 passed) and
`pytest tests/test_release_workflow.py -q` passes today (11 passed), so as acceptance they prove
nothing — nobody ever establishes the check failed first, which Principle I requires. A node that
does not exist yet exits non-zero with "no tests ran" today and passes only once the work lands.

## R7 — The release workflow's shape

**Decision**: In the `publish` job, drop `permissions: id-token: write` and the
`pypa/gh-action-pypi-publish` step; upload the same `dist/` wheel as an asset on the private repo's
Release for the pushed tag.

**Rationale**: Verified — the job scopes `id-token: write` to itself, checks out with
`persist-credentials: false`, asserts the tag matches the `__version__` literal by reading the source
text, downloads the `dist` artifact CI built, and publishes with the action pinned to a full commit
SHA. Everything except the final publish step is worth keeping verbatim: the tag/version equality
check, the no-stored-token property, and "publish builds nothing, it consumes CI's artifact".

**Repository visibility confirmed private**, so a Release asset is a genuinely private channel. This
was an open question in an earlier attempt and is now closed.

## R8 — Image transfer: the client builds

**Decision**: The client builds the deployment image locally from the wheel. `deploy/compose.yaml`
carries a `build:` directive and never an `image:` reference.

**Rationale**: The dev image is ~10.1 GB before baked weights, and GitHub Release assets cap at 2 GB
on many plans, so shipping a `docker save` archive through the same channel as the wheel does not
work. Settled by the requester; recorded in
`docs/adr/0002-air-gapped-delivery-wheel-plus-client-built-image.md`.

**Consequence**: the build host needs network for the bake step, so the air-gap is a **runtime**
guarantee only. Since the exact tested bytes are not what the client runs, the build must be pinned
— base image by digest, wheel by exact version — so two builds of one release agree. FR-020 requires
the docs to say this plainly, so an operator watching the build reach Hugging Face does not conclude
the air-gap is broken.

**Alternatives considered**: a private container registry (fewest moving parts, no size ceiling,
one `docker pull`); a split `docker save` archive; a slim image with a client-seeded weights volume.
All rejected by the requester in favour of local build.

## R9 — `pip install <glob>[extra]` is invalid in a shell `RUN`

**Decision**: Install by exact filename, or glob into a quoted shell variable first.

**Rationale**: `[ontology]` is a glob character class, so `pip install /tmp/graphknows-*.whl[ontology]`
matches nothing and pip receives the literal string. This is a silent build failure that looks like a
typo in a log, and it was present in an earlier design.

## R10 — Governance context

**Decision**: Proceed under Constitution v1.0.0, ratified 2026-09-02.

**Rationale**: The constitution was an unfilled template until immediately before this plan. It now
encodes rules already in force via `CLAUDE.md`, `docs/agents/definition-of-done.md` and
`CONTRIBUTING.md`, so the Constitution Check in plan.md is a real gate rather than a formality. The
one deviation it surfaced — rewriting a green test — is recorded in Complexity Tracking rather than
waved through.

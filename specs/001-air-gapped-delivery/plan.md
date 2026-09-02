# Implementation Plan: Air-Gapped Delivery

**Branch**: `001-air-gapped-delivery` | **Date**: 2026-09-02 | **Spec**: [spec.md](./spec.md)

**Input**: Feature specification from `/specs/001-air-gapped-delivery/spec.md`

## Summary

Deliver GraphKnows to a client as a private versioned wheel plus a deployment definition they
build and run themselves, with no outbound network at runtime. Four separable changes, all
landing at seams that already exist:

1. **Delivery** — the release workflow stops publishing to PyPI and attaches the wheel to a
   private GitHub Release instead.
2. **Deployment** — a new `deploy/` directory holds a release image built *from the wheel* and a
   deployment compose stack, both separate from the dev-only root `Dockerfile`/`docker-compose.yaml`.
3. **Offline enforcement** — the image's environment is the single source of truth, plus a
   guarded boot check that fails the container when a baked model is missing.
4. **Schema safety** — a version stamp written at `GraphStore.ensure_schema`, so a graph written
   by a different schema refuses to open rather than serving stale ids.

Plus a packaging split: an `ontology` extra carved out of `assisted`, so bring-your-own-ontology
does not drag in the LLM extraction stack an air-gapped client cannot use.

## Technical Context

**Language/Version**: Python ≥3.11 (package supports 3.11–3.13); release image base
`python:3.13-slim`, to be pinned by digest

**Primary Dependencies**: pydantic, httpx, numpy, GLiNER/GLiNER2, spaCy, sentence-transformers,
torch, rank-bm25, nltk, mcp. Moving: `rdflib`, `networkx` → new `ontology` extra; `dspy` stays
the whole of `assisted`

**Storage**: ArcadeDB (`arcadedata/arcadedb:26.5.1` in the dev stack), one physical database per
namespace

**Testing**: pytest; `make gate` is the pre-push gate — ruff over `graphknows evaluation tests`,
mypy, `lint-imports`, bandit, unit suite, 65% coverage floor (raise-only)

**Target Platform**: Linux container, built by the client; host may have no outbound network at
run time

**Project Type**: Distributed Python library plus an MCP stdio server and a container deployment.
No HTTP surface.

**Performance Goals**: The boot model-presence check adds seconds, not minutes, and runs once per
start. `bake_models.py --present` is a files-on-disk check; `--check` is the deep load and must
never run at boot or on a health-check tick — its own docstring records a measured ~5× ingest tax
from running a model probe on a health-check duty cycle.

**Constraints**: Runtime air-gap (no egress except the graph database); no public package index;
no forward migrations — upgrades are re-ingest; the root `Dockerfile` and root
`docker-compose.yaml` must not be modified.

**Scale/Scope**: Five workstreams, ~10 files touched, 3 new test files. Single client deployment;
multi-namespace within it.

No `NEEDS CLARIFICATION` remain — every open question was closed in issue #326 before the spec was
written, and the two false premises that misled earlier design attempts are corrected in
[research.md](./research.md).

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-checked after Phase 1 design.*

Evaluated against Constitution v1.0.0.

| Principle | Verdict | Evidence |
|---|---|---|
| **I. Acceptance Is a Runnable Command** | PASS | Every workstream lands a check that fails on the tip today: `tests/storage/test_schema_stamp.py` and `tests/test_offline.py` do not exist; `deploy/acceptance.sh` does not exist; the extras and release checks are **new nodes** in existing files (`tests/test_packaging.py`, `tests/test_release_workflow.py`), so they fail collection with "no tests ran" rather than passing. Naming a whole existing file is what sank three prior attempts — see research.md R6. |
| **II. Never Weaken a Check** | **DEVIATION — justified** | `tests/test_release_workflow.py` currently asserts the PyPI trusted-publishing path and will fail the moment the workflow changes. See Complexity Tracking. |
| **III. Fix the Defect, Not the Symptom** | PASS | Missing weights are an environment defect and are fixed in the image and entrypoint, never by library code fetching them. No library module gains a download path. |
| **IV. Build Only What Was Asked For** | PASS | No migration framework (upgrades are re-ingest, FR-022). No new abstraction over the store. The stamp is a helper called from the one seam every path already crosses, not a new module. `deploy/` holds two files plus one script. |
| **V. Errors Never Pass Silently** | PASS — and strengthened | The stamp *raises* on mismatch rather than reading past it (FR-012). The boot check exits non-zero rather than starting degraded (FR-004). The egress test fails loudly rather than tolerating a fetch. |
| **VI. Evidence Over Vibes** | N/A — justified | This feature changes delivery and startup, not retrieval or extraction. No recall metric can observe it; the measurements that apply are the acceptance runs in SC-001…SC-007. Principle VI governs retrieval/extraction changes, and none are made here. |
| **VII. A Package Installed by Strangers** | PASS | The new `ontology` extra is declared in `pyproject.toml`, added to `MissingExtraError.KNOWN_EXTRAS`, and enforced by `tests/test_packaging.py`. `scripts/` is **not** added to the wheel — the image copies it from the build context instead, so `tests/test_readme_install.py`'s pin on wheel contents is untouched. |

**Post-Phase-1 re-check**: unchanged. The Phase 1 design introduced no new abstraction, no new
dependency, and no additional test rewrite. The single deviation below stands as the only one.

## Project Structure

### Documentation (this feature)

```text
specs/001-air-gapped-delivery/
├── plan.md              # This file
├── spec.md              # Feature specification
├── research.md          # Phase 0 output
├── data-model.md        # Phase 1 output
├── quickstart.md        # Phase 1 output
├── contracts/           # Phase 1 output
│   ├── extras.md
│   ├── schema-stamp.md
│   └── deployment-environment.md
└── checklists/
    └── requirements.md
```

### Source Code (repository root)

```text
deploy/                              # NEW — client-facing deployment, nothing dev-only
├── Dockerfile                       # release image, built FROM the wheel
├── compose.yaml                     # build: directive, never image:
└── acceptance.sh                    # the offline acceptance run (SC-001..SC-003)

graphknows/
├── exceptions.py                    # + SchemaVersionMismatchError, + "ontology" in KNOWN_EXTRAS
├── storage/arcadedb/
│   ├── _schema.py                   # + SCHEMA_STAMP DDL, + SCHEMA_VERSION derivation
│   └── graph_store.py               # ensure_schema() reads/writes the stamp
└── symbolic/ontology/               # 5 call sites: "assisted" -> "ontology"
    ├── digest.py                    #   :324
    ├── rdf_io.py                    #   :109
    └── rdf/{prefixes,projection,taxonomy}.py   # :170, :531, :274

tests/
├── test_offline.py                  # NEW — egress guard + image/compose assertions
├── test_packaging.py                # + new node for the ontology extra
├── test_release_workflow.py         # REWRITTEN for the private-asset path
└── storage/test_schema_stamp.py     # NEW

pyproject.toml                       # + [ontology] extra; fix stale comment at :119
.github/workflows/release.yml        # publish job: PyPI -> private Release asset
scripts/docker-entrypoint.sh         # + guarded bake_models.py --present
```

**Structure Decision**: Single project — this is one installable library, not a multi-app repo, so
no `backend/`/`frontend/` split applies. The one new top-level directory is `deploy/`, and its
placement is load-bearing rather than cosmetic: `tests/test_repo_hygiene.py` scopes its assertions
to the **root** `Dockerfile` and a root-level `docker-compose*.y*ml` glob. Putting the release image
and deployment stack under `deploy/` keeps every one of those assertions true as written, instead of
relaxing them to accommodate new files — which Principle II forbids.

## Complexity Tracking

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| Rewriting three nodes in `tests/test_release_workflow.py`, all green today (Principle II) | The test asserts the *current* requirement — publish to PyPI over OIDC trusted publishing, with `id-token: write` scoped to the publish job and the action pinned by SHA. FR-007 reverses that requirement: the artifact must never reach a public index. A test encoding a requirement that no longer holds cannot stay. | Leaving it green is impossible — it asserts the presence of the very step being removed, so it fails whatever we do. Deleting it loses the protections worth keeping (tag/version equality, no stored token, no build in the publish job). The alternative chosen is to **rewrite it to pin the new path with equivalent strictness**, in the same ticket as the workflow change, so the two never diverge. This is a requirement change, not a check weakened to buy a green suite — and it must land as its own ticket rather than a drive-by edit. **Three** nodes are in scope, not two: besides the two publish-path assertions, `test_docs_describe_the_tag_push_release_mechanism` hard-asserts `"Trusted Publisher" in docs/versioning.md`, so the doc and its pin move with the workflow. Missing that third node would have made it an *undeclared* deviation, which is the failure this table exists to prevent. |

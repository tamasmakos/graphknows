---
description: "Task list for Air-Gapped Delivery"
---

# Tasks: Air-Gapped Delivery

**Input**: Design documents from `/specs/001-air-gapped-delivery/`

**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/, quickstart.md

**Tests**: Test tasks are included. The spec's success criteria (SC-001…SC-007) are stated as
runnable outcomes and Constitution Principle I requires every acceptance to be a command that fails
on the tip today, so the checks are part of the work rather than optional extras.

**Organization**: Grouped by user story. Each `- Verify:` command fails on the tree as it stands and
passes once its task lands.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (US1…US4)
- Every task carries exactly one `- Verify:` sub-bullet: one shell command, no chains. Python
  commands are written as they run at `/app` in the workspace container.

## Path Conventions

Single project. Library at `graphknows/`, tests at `tests/`, operational scripts at `scripts/`, the
new client-facing deployment at `deploy/`, all from the repository root.

---

## Phase 1: Setup (Shared Infrastructure)

**None.** Nothing in this feature needs project initialisation — the package, the test suite, the
dev container and the release workflow all exist. Inventing a setup task here would be scaffolding
for its own sake (Principle IV).

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: The one item that blocks a user story other than its own. `deploy/Dockerfile` (T002)
installs `graphknows[ontology]`, so the extra has to be declared before the release image can build.

- [ ] T001 Declare the `ontology` extra in `pyproject.toml` (move `rdflib>=7.0.0` and `networkx>=3.0` out of `assisted`, leaving `assisted = ["dspy>=2.5.0"]`), add `"ontology"` to `MissingExtraError.KNOWN_EXTRAS` in `graphknows/exceptions.py` **in the same commit** (`tests/test_packaging.py::test_known_extras_constant_matches_pyproject` asserts they agree), correct the stale `tests/test_extras.py` reference in the `pyproject.toml` comment to `tests/test_packaging.py`, and add the new node `test_ontology_extra_excludes_dspy` to `tests/test_packaging.py` asserting the extra is declared, holds `rdflib` and `networkx`, and holds no `dspy`
  - Verify: `pytest -q tests/test_packaging.py::test_ontology_extra_excludes_dspy`

**Checkpoint**: the installable surface is split; user story work can begin.

---

## Phase 3: User Story 1 - Run the memory service with no network (Priority: P1) 🎯 MVP

**Goal**: A deployment the client builds once with network and then runs with none — ingest and
search complete with zero outbound connections except the graph database, and a missing baked model
stops the container at boot instead of mid-ingest.

**Independent Test**: Bring `deploy/compose.yaml` up on a host with egress blocked, ingest a
document, search for a phrase from it, get hits, and observe that no external request was attempted.

**Note on parallelism**: T002–T006 each touch a different production file but all add nodes to the
one new `tests/test_offline.py`, so none are marked `[P]`. Run them in order.

- [X] T002 [US1] Create `deploy/Dockerfile`: base `python:3.13-slim` pinned **by digest**, `pip install` the release wheel by **exact filename** with the `[ontology]` extra (never a glob — `[ontology]` is a glob character class, research.md R9), `COPY` `scripts/bake_models.py`, `scripts/preflight.py` and `scripts/docker-entrypoint.sh` from the build context (they are not in the wheel, research.md R1), run the bake at build time, and `ENV` the six variables from `contracts/deployment-environment.md` (`HF_HUB_OFFLINE`, `TRANSFORMERS_OFFLINE`, `HF_DATASETS_OFFLINE`, `HF_HOME`, `NLTK_DATA`, `GRAPHKNOWS_REQUIRE_BAKED=1`); add node `test_release_image_is_pinned_offline_and_self_contained` to the new `tests/test_offline.py` pinning all of it. Do not modify the root `Dockerfile`
  - Verify: `pytest -q tests/test_offline.py::test_release_image_is_pinned_offline_and_self_contained`

- [X] T003 [US1] Create `deploy/compose.yaml`: a `build:` directive and never an `image:` reference (research.md R8), an ArcadeDB service with a healthcheck plus `depends_on: condition: service_healthy` on the app service so a stopped database fails inside the healthcheck window (SC-003), no workspace/eval/GPU service and no published application port; add node `test_compose_overrides_no_offline_variable` to `tests/test_offline.py` asserting the compose file overrides none of the six image ENV variables and carries no `image:` key for the app service. Do not modify the root `docker-compose.yaml`
  - Verify: `pytest -q tests/test_offline.py::test_compose_overrides_no_offline_variable`

- [ ] T004 [US1] Add the guarded boot check to `scripts/docker-entrypoint.sh` — `[ "${GRAPHKNOWS_REQUIRE_BAKED:-}" = "1" ] && python bake_models.py --present`, after the existing `preflight.py` call and before `exec "$@"`, relying on the `set -e` already present; `--present` never `--check`, because `--check` deep-loads every model and carries a measured ~5× ingest tax (FR-005, research.md R2); add node `test_entrypoint_checks_baked_models_only_when_required` to `tests/test_offline.py` asserting the check is present, is gated on `GRAPHKNOWS_REQUIRE_BAKED`, uses `--present`, and runs before `exec` — the gate is what keeps the dev container's lazy cold start working (FR-006)
  - Verify: `pytest -q tests/test_offline.py::test_entrypoint_checks_baked_models_only_when_required`

- [ ] T005 [US1] Add the egress guard to `tests/test_offline.py` as node `test_llm_free_ingest_and_recall_attempt_no_egress`: monkeypatch `socket.socket.connect` to raise on any address outside an allowlist of loopback plus the host resolved from `GRAPHKNOWS_ARCADEDB_URL` (research.md R5 — the database is not on loopback under compose), wrap an `llm_free` ingest and recall in it, and leave a comment at the call site recording that the host is resolved before the block is installed so the DNS lookup precedes it. Use stdlib monkeypatching, not a new `pytest-socket` dependency
  - Verify: `pytest -q tests/test_offline.py::test_llm_free_ingest_and_recall_attempt_no_egress`

- [ ] T006 [US1] Create `deploy/acceptance.sh`, the SC-001…SC-003 run: ingest a document into the running stack and search for a phrase from it, failing loudly on an empty result rather than reporting it as an answer, and **state in a comment at the top how egress is actually blocked** (compose `internal: true`, `--network none`, or host firewall rules) so a reviewer can tell whether a run was a real test; add node `test_acceptance_script_names_how_egress_is_blocked` to `tests/test_offline.py` asserting the script exists and names its blocking method
  - Verify: `pytest -q tests/test_offline.py::test_acceptance_script_names_how_egress_is_blocked`

**Checkpoint**: the client-buildable offline deployment exists and its guarantees are pinned by tests.

---

## Phase 4: User Story 2 - Receive the software privately (Priority: P2)

**Goal**: A tagged release attaches the wheel to a private GitHub Release instead of publishing it
to PyPI, with the protections of the old path carried over intact.

**Independent Test**: Cut a release, confirm the versioned wheel appears on the private Release and
that a PyPI search for the project returns nothing.

- [ ] T007 [P] [US2] Rewrite the `publish` job in `.github/workflows/release.yml`: drop `permissions: id-token: write` and the `pypa/gh-action-pypi-publish` step, and upload the `dist/` wheel CI built as an asset on the Release for the pushed tag; keep verbatim the tag/version equality check that reads the literal out of `graphknows/_version.py`, `persist-credentials: false`, and the property that the publish job builds nothing. Rewrite `tests/test_release_workflow.py` in the **same** commit to pin the new path with the strictness the old file applied to PyPI — replacing `test_publishes_via_trusted_publisher_sha_pinned_no_stored_token` and `test_id_token_write_is_scoped_to_the_publish_job_only` with `test_publishes_wheel_as_private_release_asset` and keeping every other node. Also update `docs/versioning.md` to describe the Release-asset mechanism instead of the Trusted Publisher registration, and re-pin `test_docs_describe_the_tag_push_release_mechanism` against the new text — that node asserts `"Trusted Publisher" in docs/versioning.md` (tests/test_release_workflow.py:256) and breaks the moment the workflow changes, so it is part of this same rewrite and not a separate one. This is the single Principle II deviation, justified in plan.md Complexity Tracking: these tests encode a requirement FR-007 reverses
  - Verify: `pytest -q tests/test_release_workflow.py::test_publishes_wheel_as_private_release_asset`

**Checkpoint**: delivery is private; nothing reaches a public index.

---

## Phase 5: User Story 3 - Be stopped before a stale graph is served (Priority: P3)

**Goal**: Every namespace records the schema shape that created it, and opening one written by a
different shape is refused with a message an operator can recover from unaided.

**Independent Test**: Build a graph on one version, change `_CORE_DDL`, reopen the same namespace,
and read the error — it must name the stored version, the running version, and the remedy.

- [ ] T008 [P] [US3] Add `SchemaVersionMismatch` to `graphknows/exceptions.py`, subclassing the existing `StoreError` so callers already handling store failures keep working; the message must name the namespace/database, the recorded version, the running version, and the remedy — a rebuild from original sources, referencing a gesture that actually exists rather than a flag that does not (FR-013)
  - Verify: `python -c "from graphknows.exceptions import SchemaVersionMismatch"`

- [ ] T009 [P] [US3] Add the `SCHEMA_STAMP` type, its `stamp_id`/`version` properties and the UNIQUE index on `stamp_id` to `_CORE_DDL` in `graphknows/storage/arcadedb/_schema.py`, following that file's existing type → properties → index idiom, and derive `SCHEMA_VERSION` as a short digest over `"\n".join(stmt for _lang, stmt in _CORE_DDL)` combined with `dims` — `_CORE_DDL` is `list[tuple[str, str]]`, so the statement must be projected out of each pair (research.md R3), and `dims` participates because the DDL interpolates it into the `LSM_VECTOR` indexes; record in a comment that editing `_CORE_DDL` is now a client-visible event that strands existing graphs. Create `tests/storage/test_schema_stamp.py` with node `test_schema_version_changes_with_ddl_text_and_dims`
  - Verify: `pytest -q tests/storage/test_schema_stamp.py::test_schema_version_changes_with_ddl_text_and_dims`

- [ ] T010 [US3] Implement the decision table from `contracts/schema-stamp.md` in `GraphStore.ensure_schema` (`graphknows/storage/arcadedb/graph_store.py`): record `client.database_exists()` **before** `_apply_ddl` (which creates the database, so afterwards the question is unanswerable), then read the stamp — absent on a fresh database inserts the current version, absent on a pre-existing one raises (FR-014), equal proceeds silently, differing raises `SchemaVersionMismatch` before any query the caller asked for. Check, rather than assume, whether `graphknows/symbolic/ontology/rdf/store.py`'s own `ensure_schema` and the direct store construction in `graphknows/storage/__init__.py` need the same guard (research.md R4 open edge), and state what you found in the commit message. Add nodes `test_fresh_namespace_stamps_and_proceeds`, `test_matching_stamp_proceeds`, `test_pre_existing_unstamped_database_refuses` and `test_mismatched_stamp_refuses_naming_both_versions_and_the_remedy` to `tests/storage/test_schema_stamp.py`
  - Verify: `pytest -q tests/storage/test_schema_stamp.py::test_mismatched_stamp_refuses_naming_both_versions_and_the_remedy`

- [ ] T011 [US3] Handle concurrent first open in `GraphStore.ensure_schema` (`graphknows/storage/arcadedb/graph_store.py`): catch the UNIQUE-index rejection the way `_apply_ddl` already catches benign re-creation (lowercased message containing `already exists` / `duplicate`), then re-read the stamp and apply the decision table — exactly one stamp survives and neither process fails (FR-016); add node `test_concurrent_first_open_leaves_exactly_one_stamp` to `tests/storage/test_schema_stamp.py`
  - Verify: `pytest -q tests/storage/test_schema_stamp.py::test_concurrent_first_open_leaves_exactly_one_stamp`

**Checkpoint**: a mis-keyed graph is never read from or written to.

---

## Phase 6: User Story 4 - Bring your own ontology without the LLM stack (Priority: P4)

**Goal**: An operator installing ontology support gets it without the LLM-extraction stack, and the
error they see when it is missing names the right extra.

**Independent Test**: Install `graphknows[ontology]` in a clean environment, import
`graphknows.symbolic.ontology.rdf_io`, and confirm `pip show dspy` finds nothing.

- [ ] T012 [US4] Change the five ontology `MissingExtraError` call sites from `"assisted"` to `"ontology"` — `graphknows/symbolic/ontology/rdf_io.py:109`, `graphknows/symbolic/ontology/rdf/taxonomy.py:274`, `graphknows/symbolic/ontology/rdf/projection.py:531`, `graphknows/symbolic/ontology/rdf/prefixes.py:170`, `graphknows/symbolic/ontology/digest.py:324` — leaving `graphknows/ingestion/extraction/relations/llm_assisted/dspy.py:34` on `"assisted"`, which is the real dspy path (FR-019); add node `test_ontology_call_sites_name_the_ontology_extra` to `tests/test_packaging.py` pinning all six
  - Verify: `pytest -q tests/test_packaging.py::test_ontology_call_sites_name_the_ontology_extra`

**Checkpoint**: ontology support installs and errors independently of the LLM stack.

---

## Phase 7: Polish & Cross-Cutting Concerns

- [ ] T013 Write the deployment documentation (`docs/deployment.md`, linked from `docs/README.md`): the build needs network and the offline guarantee covers running, not building, so an operator watching the build reach Hugging Face does not conclude the air-gap is broken (FR-020); the capabilities outside the guarantee — LLM-assisted extraction and the remote embedding option — named so they can be avoided deliberately (FR-021); and the upgrade path, including that a schema change requires rebuilding the graph from its original sources with no migration offered (FR-022). Add node `test_deployment_doc_states_the_guarantee_boundaries` to `tests/test_offline.py`
  - Verify: `pytest -q tests/test_offline.py::test_deployment_doc_states_the_guarantee_boundaries`

- [ ] T014 [US2] Assemble and attach the deployment bundle in `.github/workflows/release.yml`, alongside the wheel: an archive whose layout **mirrors the repository** — `deploy/` and `scripts/` at its root plus the wheel — so `deploy/Dockerfile` builds identically from an unpacked bundle or a checkout, and the client never needs repository access (FR-010, FR-010a). Copy `scripts/bake_models.py`, `scripts/preflight.py` and `scripts/docker-entrypoint.sh` into the archive at assembly time rather than committing second copies under `deploy/`: two files that must stay identical are a sync problem, and there is nothing to keep in sync if only one copy exists. Add node `test_release_attaches_a_self_contained_deployment_bundle` to `tests/test_release_workflow.py` asserting the bundle step exists, names all three scripts and the `deploy/` directory, and preserves the repo-relative layout
  - Verify: `pytest -q tests/test_release_workflow.py::test_release_attaches_a_self_contained_deployment_bundle`

**Not a task**: running `make gate`. It is green today, so it cannot be a task under the
fails-first rule — it is the pre-push gate every task above must leave green.

---

## Dependencies & Execution Order

### Phase Dependencies

- **Phase 1 (Setup)**: empty.
- **Phase 2 (Foundational, T001)**: blocks T002, because `deploy/Dockerfile` installs
  `graphknows[ontology]` and the extra must exist first. Nothing else depends on it.
- **Phase 3 (US1)**: after T001. T002 → T003 → T004 → T005 → T006 in order — they share
  `tests/test_offline.py`.
- **Phase 4 (US2, T007)**: independent of every other phase. Touches only
  `.github/workflows/release.yml` and `tests/test_release_workflow.py`.
- **Phase 5 (US3)**: independent of US1/US2. T008 ∥ T009, then T010, then T011.
- **Phase 6 (US4, T012)**: after T001 (`"ontology"` must be in `KNOWN_EXTRAS` before a call site
  names it, or `test_missing_extra_call_sites_name_declared_extras` goes red).
- **Phase 7 (T013)**: after T006, sharing `tests/test_offline.py`; content depends on US1 and US3
  being decided.

### Parallel Opportunities

- T007 (US2) runs alongside the whole of US1 and US3 — no shared file.
- T008 ∥ T009 — `graphknows/exceptions.py` versus `graphknows/storage/arcadedb/_schema.py`.
- US3 as a whole runs alongside US1: `deploy/` and `scripts/` never meet `graphknows/storage/`.
- US1's five tasks are **not** parallel with each other despite touching five different production
  files, because each adds a node to the same new `tests/test_offline.py`.

```bash
# After T001, three tracks can run at once:
Track A: T002 → T003 → T004 → T005 → T006 → T013   # offline deployment
Track B: T007                                       # private delivery
Track C: (T008 ∥ T009) → T010 → T011                # schema safety
```

---

## Implementation Strategy

### MVP (User Story 1)

1. T001 — the extra the image installs.
2. T002…T006 — image, compose, boot check, egress guard, acceptance run.
3. **STOP and VALIDATE**: build on a networked host, cut egress, run `deploy/acceptance.sh`.
   SC-001…SC-003 are met and the product promise is demonstrable at a client site.

### Incremental Delivery

- **+ US2 (T007)** — the MVP becomes deliverable privately rather than by hand.
- **+ US3 (T008…T011)** — the only silent-and-wrong failure mode in this feature is closed.
- **+ US4 (T012)** — the error messages stop pointing at the wrong extra.
- **+ T013** — the recipient can reach a working deployment without contacting the maintainers
  (SC-007).

### Notes

- Every `- Verify:` above exits non-zero on the tip today: `tests/test_offline.py` and
  `tests/storage/test_schema_stamp.py` do not exist, the named nodes in `tests/test_packaging.py`
  and `tests/test_release_workflow.py` do not exist (so pytest exits with "no tests ran"), and
  `SchemaVersionMismatch` does not import. Naming a whole already-passing file as acceptance is what
  sank three prior drafting attempts (research.md R6).
- The root `Dockerfile` and root `docker-compose.yaml` are not modified by any task. That is what
  keeps `tests/test_repo_hygiene.py`'s assertions — `HF_HUB_OFFLINE` and `HEALTHCHECK` absent from
  the root `Dockerfile`, exactly one root-level `docker-compose*.y*ml` — true as written instead of
  relaxed.
- One Principle II deviation exists in this list, T007, and it is justified in plan.md. No other
  task may weaken an existing check.

# Quickstart: Validating Air-Gapped Delivery

How to prove this feature works, end to end. Every command below **fails on today's tip** — that is
the point (Principle I). Contract details live in [contracts/](./contracts/); this file is the run
guide.

## Prerequisites

- Docker with compose, and enough disk for a multi-gigabyte image
- Network **once**, for the build only
- The release bundle for the version under test, unpacked — it carries `deploy/`, `scripts/` and the
  wheel in the same relative layout as this repository, so every command below is identical whether
  you run it from an unpacked bundle or from a checkout

## Scenario 1 — Ingest and search with no network (SC-001)

Covers User Story 1, the core promise.

```bash
# Build. This is the one step allowed to reach the network.
docker compose -f deploy/compose.yaml build

# Cut the host off, then bring the stack up.
docker compose -f deploy/compose.yaml up -d

# Ingest and recall.
sh deploy/acceptance.sh
```

**Expected**: every service reaches healthy; the ingest completes; the search returns hits from the
ingested document; and no external connection is attempted at any point.

**How egress is actually blocked** matters — a test that merely *asserts* offline while the host has
network proves nothing. Use one of: a compose network with `internal: true`, `--network none` on a
one-shot run, or host firewall rules. Whichever the ticket picks, it must be stated in the script,
because a reviewer cannot otherwise tell whether the run was a real test.

## Scenario 2 — Fail loudly when a model is missing (SC-002)

```bash
# Start the release image with the baked-model path emptied or masked.
docker compose -f deploy/compose.yaml run --rm \
  -v /dev/null:/opt/models/marker app true
```

**Expected**: the container exits non-zero **before** serving anything, naming the missing resource.
It must not start healthy and fail later mid-ingest — that is the failure mode the boot check
exists to prevent.

## Scenario 3 — Fail loudly when the database is down (SC-003)

```bash
docker compose -f deploy/compose.yaml up -d
docker compose -f deploy/compose.yaml stop arcadedb
sh deploy/acceptance.sh   # expected to fail
```

**Expected**: failure inside the healthcheck window with a message naming the database. Not a hang,
and not an empty search result presented as a valid answer — an empty result here is the bug.

## Scenario 4 — Refuse a stale graph (SC-005)

Covers User Story 3, the only silent-and-wrong failure in this feature.

```bash
pytest tests/storage/test_schema_stamp.py -q
```

**Expected today**: fails — the file does not exist.
**Expected after**: passes, including a graph written by a different version being refused with a
message naming the stored version, the running version, and the remedy.

Manual equivalent: ingest into a namespace, change any statement in `_CORE_DDL`, reopen the same
namespace, and read the error. If it does not tell you what to do next without opening the source,
FR-013 is not met.

## Scenario 5 — Ontology without the LLM stack (SC-006)

```bash
python -m venv /tmp/ontology-check && /tmp/ontology-check/bin/pip install "dist/graphknows-<version>-py3-none-any.whl[ontology]"
/tmp/ontology-check/bin/python -c "from graphknows.symbolic.ontology import rdf_io; print('ok')"
/tmp/ontology-check/bin/pip show dspy    # expected: not found
```

**Expected**: the import succeeds and `dspy` is absent. Then the contract test:

```bash
pytest tests/test_packaging.py::test_ontology_extra_excludes_dspy -q
```

**Expected today**: exits non-zero with "no tests ran" — the node does not exist. Note the *file*
passes today, which is why the node must be named (research.md R6).

## Scenario 6 — Private delivery (SC-004)

```bash
pytest tests/test_release_workflow.py::test_publishes_wheel_as_private_release_asset -q
```

**Expected today**: "no tests ran".
**Expected after**: passes, and the rest of that file — rewritten in the same ticket — pins the new
path with the strictness the old one applied to PyPI: tag/version equality, no stored token, and no
build in the publish job.

## Full gate

Before pushing anything:

```bash
make gate
```

Runs ruff over `graphknows evaluation tests`, mypy, `lint-imports`, bandit, the unit suite and the
coverage floor. `make lint` and `make test` are quick local checks and are not a substitute.

## What "done" looks like

All six scenarios pass, `make gate` is green, and no existing test was weakened to get there —
with the single documented exception of `tests/test_release_workflow.py`, whose rewrite is justified
in [plan.md](./plan.md) under Complexity Tracking because the requirement it encoded was reversed.

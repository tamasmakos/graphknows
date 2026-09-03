# 1. Ship an air-gapped release image, not a public wheel

Status: accepted

(Supersedes the first draft of this file, written against two premises that turned out to
be false: that `scripts/` ships inside the wheel, and that something already runs
`bake_models.py` at boot. Neither is true — see Context.)

## Context

Today the PyPI wheel is the product. `.github/workflows/release.yml` publishes it via
trusted publishing; `Dockerfile` is deliberately one dev-only, bind-mount-oriented stage
("there is no shipped runtime image"); `docker-compose.yaml` says "there is no separate
deployment compose file". `tests/test_repo_hygiene.py` pins that absence — the `Dockerfile`
text may not contain `HEALTHCHECK` or `HF_HUB_OFFLINE`, may not name a build stage, and
exactly one `docker-compose*.y*ml` may exist at the repo root.

The client runs air-gapped. Nothing in that shape reaches them, and three guarantees the
delivery needs do not exist:

- `[tool.hatch.build.targets.wheel] packages = ["graphknows"]` is the only inclusion rule,
  so `scripts/` is in no wheel by any install path (`tests/test_readme_install.py` pins that
  fact and must not be relaxed).
- `scripts/docker-entrypoint.sh` runs only `preflight.py`, which early-returns when there is
  no bind-mounted source — i.e. always, in an image. The only caller of `bake_models.py` is
  the `Makefile`. A missing weight fails at first use, not at boot.
- No offline enforcement exists anywhere: `HF_HUB_OFFLINE` appears in a comment and in
  `bake_models.py`'s own `--check` path, nowhere else.
- No schema/namespace version stamp exists: `storage/namespace.py` maps a name to a database
  and nothing more; `_CORE_DDL` carries no version marker.

## Decision

We will deliver a **private versioned wheel plus a deployment compose the client operates** —
no public index, no HTTP surface. The wheel is built by CI and attached to a GitHub Release
on this (private) repo. A **self-contained release image** is built *from that wheel* by a
separate `Dockerfile.release`, which `COPY`s `scripts/bake_models.py` and
`scripts/docker-entrypoint.sh` from the build context (not from the wheel) and bakes every
model at build time. The dev `Dockerfile` and `docker-compose.yaml` stay untouched, so
`tests/test_repo_hygiene.py`'s Dockerfile assertions stay honest rather than relaxed.

We will make **the image's ENV the single source of truth for offline** (`HF_HUB_OFFLINE`,
`TRANSFORMERS_OFFLINE`, `HF_DATASETS_OFFLINE`, `NLTK_DATA`), and **extend
`docker-entrypoint.sh` to run `bake_models.py --present` once at boot** — files-on-disk, not
the minutes-long `--check`, and not on a HEALTHCHECK tick, because that duty cycle cost a
measured 5x on ingest.

We will **split an `ontology` extra (rdflib, networkx) out of `assisted` (dspy)** and repoint
the five ontology/RDF `MissingExtraError` call sites at it.

We will treat **upgrades as re-ingest**: `GraphStore.ensure_schema` — the one seam both
callers route through — stamps the namespace with a hash of the DDL and refuses a namespace
stamped differently (or not at all), with an error naming `--reset`. We will write no forward
migrations.

## Consequences

Good: one artifact runs with no network; a missing weight or a stale graph fails loudly at
boot instead of degrading silently; the two-tier `--check`/`--present` split already in
`bake_models.py` is reused rather than re-derived; a no-extras install that touches RDF now
names the extra it actually needs.

Bad: a second Dockerfile and a second compose file to keep in step; `release.yml` and
`tests/test_release_workflow.py` must be rewritten off trusted publishing; the image carries
every model, so it is ~10 GB and its transfer channel is still unsettled (GitHub release
assets cap below that); hashing the DDL means a cosmetic DDL edit strands existing graphs.

Neutral: `llm_assisted` and the remote embedder stay, documented as explicit opt-outs of the
air-gap guarantee. `tests/test_repo_hygiene.py`'s `test_exactly_one_compose_file` keeps its
root-scoped glob because the deployment compose lives under `deploy/`.

# 2. Air-gapped delivery: private wheel plus a client-built release image

Status: accepted. Supersedes the transfer mechanism in the untracked draft
`0001-air-gapped-release-image-delivery.md` (image archive attached to a Release); everything
else in that draft is restated here.

## Context

A client runs GraphKnows air-gapped. Today the product is a PyPI wheel published by
`.github/workflows/release.yml` through trusted publishing (`id-token: write`,
`pypa/gh-action-pypi-publish`). `Dockerfile` is one deliberately dev-only stage ("there is no
shipped runtime image") and `docker-compose.yaml` is documented as the only compose file;
`tests/test_repo_hygiene.py` pins both — no `HEALTHCHECK`/`HF_HUB_OFFLINE` and no build stage
in the root Dockerfile, exactly one root-level `docker-compose*.y*ml`.

Four guarantees the delivery needs do not exist. `scripts/` ships in no wheel (hatch includes
only `packages = ["graphknows"]`). `scripts/docker-entrypoint.sh` runs only `preflight.py`,
which early-returns without a bind mount — i.e. always in an image — so a missing weight
fails at first use, not at boot. `HF_HUB_OFFLINE` appears nowhere as image ENV or setting.
`GraphStore.ensure_schema` applies `_CORE_DDL` and stamps nothing, so a graph written by an
older schema is read back as if current. Separately, one `assisted` extra bundles dspy with
rdflib and networkx, and five of the six `MissingExtraError` call sites that name it are
RDF-only.

A ~10 GB image cannot be attached to a GitHub Release, and the repo is private, so neither a
public index nor an image archive is a transfer channel.

## Decision

We will ship a **private versioned wheel as a GitHub Release asset**, and a
**`deploy/compose.yaml` whose service uses a `build:` directive against a new
`deploy/Dockerfile`** — the client builds the release image locally from the wheel. The
air-gap is a **runtime** guarantee, not a build-time one: the build host needs network, and
the docs will say so. Because we no longer ship the tested bytes, `deploy/Dockerfile` pins its
base image by digest, the wheel by exact version, and model ids stay pinned in
`bake_models.py::_model_ids`. The dev `Dockerfile` and root compose are untouched, so
`tests/test_repo_hygiene.py` stays honest rather than relaxed.

**Image ENV is the single source of truth for offline** (`HF_HUB_OFFLINE`,
`TRANSFORMERS_OFFLINE`, `HF_DATASETS_OFFLINE`); settings may read it, never own it.
`scripts/docker-entrypoint.sh` gains one guarded step — `bake_models.py --present` when
`GRAPHKNOWS_REQUIRE_BAKED=1`, set only by the release image — because the dev image downloads
weights lazily into a named volume and would fail an unguarded boot check. `--present`, not
`--check`: the deep check on a duty cycle cost a measured 5x on ingest.

We will **split an `ontology` extra (rdflib, networkx) out of `assisted` (dspy)** and repoint
the five RDF call sites at it.

Upgrades are **re-ingest, never migration**. `ensure_schema` — the one seam every runtime path
crosses (`graphknows/memory.py:155`) — writes a `SCHEMA_STAMP` singleton carrying a hash of
`_CORE_DDL`, and refuses a namespace whose stamp differs, or which pre-existed unstamped, with
an error naming the mismatch and `--reset`. The singleton is backed by a UNIQUE index and a
duplicate-key rejection is read as "a concurrent connect stamped it first", per the throwaway
race prototype.

## Consequences

Good: one command brings up a stack that needs no network; a missing weight or a stale graph
fails loudly at boot instead of degrading silently; a no-extras install that touches RDF names
the extra it actually needs; the `--check`/`--present` split already in `bake_models.py` is
reused, and `database_exists` already distinguishes a fresh namespace from an old one, so the
stamp needs no data probe.

Bad: a second Dockerfile and compose file to keep in step; `release.yml` loses trusted
publishing and the client loses a public `pip install`; the client's build host must have
network and will produce bytes we never tested; hashing `_CORE_DDL` means a cosmetic DDL edit
strands existing graphs (the alternative, a hand-bumped constant, is a step someone forgets).

Neutral: `llm_assisted` and the remote embedder stay, documented as explicit opt-outs of the
air-gap guarantee. `test_exactly_one_compose_file` keeps its root-scoped glob because the
deployment compose lives under `deploy/`.

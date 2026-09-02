# Contract: Deployment Environment

The environment is the single source of truth for offline behaviour (FR-003). Library code does not
set these, and no `GraphKnowsSettings` field owns them. Settings may *read* them to produce a
helpful error — a read is not a toggle.

## Set by the release image (`deploy/Dockerfile`, `ENV`)

| Variable | Value | Effect |
|---|---|---|
| `HF_HUB_OFFLINE` | `1` | Hugging Face never reaches the hub, warm cache or cold |
| `TRANSFORMERS_OFFLINE` | `1` | transformers resolves locally only |
| `HF_DATASETS_OFFLINE` | `1` | datasets resolves locally only |
| `HF_HOME` | baked cache path | Where the build wrote model weights |
| `NLTK_DATA` | baked corpora path | Where the build wrote WordNet/FrameNet. `nltk.download()` ignores this variable, which is why the bake passes `download_dir` explicitly |
| `GRAPHKNOWS_REQUIRE_BAKED` | `1` | Turns on the boot presence check. **Release image only** |

`deploy/compose.yaml` must **not** override any of them. A test asserts this: an override in the
compose file would silently defeat the guarantee the image just made, and would be invisible in
review.

## Not set by the dev stack

The root `Dockerfile` and `docker-compose.yaml` set none of these and are not modified. The dev
container downloads weights lazily into a named volume, so `GRAPHKNOWS_REQUIRE_BAKED` must stay
unset there — an unguarded boot check would break a working dev cold start.

This is also why `tests/test_repo_hygiene.py` can keep asserting that `HF_HUB_OFFLINE` and
`HEALTHCHECK` are **absent** from the root `Dockerfile`: the release image is a different file, so
those assertions stay true as written rather than being relaxed.

## Boot sequence (`scripts/docker-entrypoint.sh`)

```sh
set -e                                    # already present
python preflight.py || exit 1             # already present; no-ops inside an image
# NEW:
[ "${GRAPHKNOWS_REQUIRE_BAKED:-}" = "1" ] && python bake_models.py --present
exec "$@"                                 # already present
```

`--present`, never `--check`: the former tests files on disk in seconds, the latter deep-loads every
model. `set -e` means a non-zero exit aborts before `exec`, with no extra plumbing.

## Image build

- Base image pinned **by digest**, not by tag.
- Library installed from the release wheel by **exact version** — never from the source tree, so
  `python -c "import graphknows; print(graphknows.__file__)"` resolves under `site-packages`.
- `scripts/bake_models.py`, `scripts/preflight.py`, `scripts/docker-entrypoint.sh` copied from the
  **build context**; they are not in the wheel (research.md R1).
- The bake step runs at build time and is the one step permitted to use the network.
- Install by exact filename or a quoted variable — `pip install /tmp/graphknows-*.whl[ontology]`
  silently matches nothing, because `[ontology]` is a glob character class (research.md R9).

## Compose stack

- `build:` directive, never `image:` — the client builds locally (research.md R8).
- Graph database service with a healthcheck; the app service waits on it via
  `depends_on: condition: service_healthy`, so a stopped database fails startup inside the
  healthcheck window with a message naming it, rather than hanging (SC-003).
- No workspace, eval, or GPU service. Those are dev-only and do not ship.
- No published application port. There is no HTTP surface.

## Acceptance

Neither exists today:

```bash
pytest tests/test_offline.py -q
sh deploy/acceptance.sh
```

The pytest file asserts the image declares the ENV block and copies both scripts, that the compose
file overrides none of the variables, and runs the egress guard: a monkeypatched
`socket.socket.connect` that raises on any address outside an allowlist of loopback **plus the host
resolved from `GRAPHKNOWS_ARCADEDB_URL`** (research.md R5), wrapped around an `llm_free` ingest and
recall.

# Deploying with no network

The release bundle — `deploy/`, `scripts/` and the wheel, laid out as they are
in the repository — builds into one image that runs with the host's outbound
traffic cut. The stack is the memory service and its ArcadeDB, nothing else.

```bash
docker compose -f deploy/compose.yaml build   # the one step that needs network
docker compose -f deploy/compose.yaml up -d   # this one needs none
```

## The build needs network; the guarantee is about running

Building downloads the base image, the wheel's dependencies, the spaCy model,
and then bakes every model GraphKnows loads into `/opt/models` —
`BAAI/bge-small-en-v1.5`, the relex and typing models, the relation verifier and
its encoder, and the NLTK corpora. Watching that step reach Hugging Face is the
build working, not the air-gap leaking: the offline guarantee covers **running**
the system, not building it.

After the bake — and only after it, or the bake itself would fail — the image
sets `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, `HF_DATASETS_OFFLINE=1` and
`GRAPHKNOWS_REQUIRE_BAKED=1`. The running container loads weights from disk and
never revalidates them against the hub. `GRAPHKNOWS_REQUIRE_BAKED=1` also makes
the entrypoint check the cache at boot, so an image built without a bake exits
immediately instead of failing on someone's first ingest.

Set none of those five in compose or in your environment. The image owns them,
and a value set outside it wins.

To confirm the running stack on a host where egress is actually blocked:

```bash
EGRESS_BLOCKED_BY=firewall sh deploy/acceptance.sh
```

The script ingests a document and searches for a phrase from it; an empty result
is a failure. It cannot detect whether egress was blocked — a networked laptop
prints the same output — so `EGRESS_BLOCKED_BY` records the method you used.

## What the guarantee does not cover

Three capabilities reach the network by design. All three are off in the shipped
`deploy/compose.yaml`; each is enabled only by a setting you write.

| Capability | Turned on by | What it dials |
| --- | --- | --- |
| LLM relation extraction and LLM topic summaries | `GRAPHKNOWS_MODE=llm_assisted`, or `GRAPHKNOWS_DSPY_RELATIONS=true` / `GRAPHKNOWS_TOPICS=llm` on their own | your LLM provider, on every ingest |
| Remote embeddings | `GRAPHKNOWS_EMBED_API_BASE` (with `GRAPHKNOWS_EMBED_API_KEY`) | the embedding API, on every ingest and every query |
| A model the image did not bake | `GRAPHKNOWS_EMBED_MODEL` or `GRAPHKNOWS_SPACY_MODEL` pointing elsewhere | nothing — offline mode turns the download into a failure |

The first two leave the guarantee. The third does not: the container fails
rather than fetching, which is why a model swap has to be baked into a rebuilt
image. Everything else — entity extraction, deterministic relations, frames,
ontology injection, retrieval — is local in both modes; see
[modes.md](modes.md).

## Upgrading

Rebuild the image from the new bundle and restart the stack:

```bash
docker compose -f deploy/compose.yaml down
docker compose -f deploy/compose.yaml build   # network again, for the bake
docker compose -f deploy/compose.yaml up -d
```

The data outlives the container, and that is where an upgrade can bite. Every
namespace records the schema version that created it. If a release changed the
graph schema — or you changed the embedding dimension, which the vector indexes
are typed by — opening that namespace raises `SchemaVersionMismatchError`,
naming the recorded version, the running one, and the remedy. It refuses before
the first read or write rather than serving a graph it cannot address.

There is no migration path, and none is planned: the graph is derived data, so
the upgrade is to rebuild it from the sources it was built from.

1. Keep the original documents. They are the only authoritative copy — nothing
   reconstructs them from the graph.
2. Drop the stale namespace: `Memory.drop_namespace()` (MCP tool
   `memory_drop_namespace`), or `Memory.purge_memory()` (MCP tool
   `memory_purge`) for the default namespace.
3. Re-ingest them against the new version.

Nothing is dropped automatically; the deletion is yours to run. Where a
rebuild is unacceptable, keep the old version running against that graph and
re-ingest into a second namespace on the new one
([multi-tenancy.md](multi-tenancy.md)).

## Backup and restore

The graph lives entirely in the `arcadedb_data` volume declared in
`deploy/compose.yaml`; nothing else in the stack holds state. Back it up by
archiving the volume through a throwaway container that mounts it, not by
reaching into the host filesystem where Docker stores it.

Stop `arcadedb` first — ArcadeDB writes its own files underneath the running
server, and archiving them while it writes is how you get a backup that
fails to restore.

```bash
docker compose -f deploy/compose.yaml stop arcadedb

docker run --rm \
  --volumes-from "$(docker compose -f deploy/compose.yaml ps -a -q arcadedb)" \
  -v "$(pwd)":/backup alpine \
  tar czf "/backup/arcadedb_data-$(date +%Y%m%d).tar.gz" -C /home/arcadedb/databases .

docker compose -f deploy/compose.yaml start arcadedb
```

Restore is the same trick in reverse: stop the server, wipe the volume's
contents, untar the archive back in, start the server.

```bash
docker compose -f deploy/compose.yaml stop arcadedb

docker run --rm \
  --volumes-from "$(docker compose -f deploy/compose.yaml ps -a -q arcadedb)" \
  -v "$(pwd)":/backup alpine \
  sh -c 'rm -rf /home/arcadedb/databases/* && tar xzf "/backup/$1" -C /home/arcadedb/databases' \
  -- "arcadedb_data-YYYYMMDD.tar.gz"

docker compose -f deploy/compose.yaml start arcadedb
```

A restored volume still carries whatever schema version it was backed up
with — the same `SchemaVersionMismatchError` check above applies when you
restore an older backup underneath a newer image.

#!/bin/sh
# The offline acceptance run (SC-001…SC-003): ingest a document into the running
# stack and search for a phrase from it. An empty result is a FAILURE here, never
# an answer — that is the whole point of running it.
#
#   EGRESS_BLOCKED_BY=firewall sh deploy/acceptance.sh
#
# HOW EGRESS IS ACTUALLY BLOCKED. This script cannot check it: a truly air-gapped
# host and a networked laptop produce byte-identical output below. So the method
# is named by the operator, echoed into the run log, and a run that names none is
# marked as what it is. Pick one before running:
#
#   internal: true   a `networks:` block in a compose override marking the stack's
#                    network internal — the containers get no route off the host,
#                    and only the app↔arcadedb link survives.
#   --network none   `docker run --network none` on a single container. Fine for
#                    the boot check, NOT for this run: the app must still reach
#                    arcadedb, which `none` also cuts.
#   firewall         host rules (iptables/nftables/pf) dropping outbound traffic
#                    from the container subnet, with the database left reachable.
#
# SC-002 and SC-003 are the preconditions of getting this far rather than steps of
# their own: the app only started because scripts/docker-entrypoint.sh found the
# baked models, and compose only started it because arcadedb passed its
# healthcheck. If either failed, the `exec` below fails loudly naming the service.
set -eu

EGRESS_BLOCKED_BY="${EGRESS_BLOCKED_BY:-UNSET — smoke test, not an acceptance run}"
echo "egress blocked by: $EGRESS_BLOCKED_BY"

docker compose -f "$(dirname "$0")/compose.yaml" exec -T app python - <<'PY'
import asyncio
import sys

from graphknows.memory import Memory
from graphknows.settings import GraphKnowsSettings

DOCUMENT = "Ada Lovelace wrote the first algorithm for the Analytical Engine in London."
PHRASE = "Analytical Engine"
SESSION = "acceptance"


async def main() -> None:
    """Ingest one document, search it back out of the graph, drop what we wrote."""
    async with Memory(GraphKnowsSettings(), namespace=SESSION) as memory:
        try:
            ingested = await memory.ingest_memory(text=DOCUMENT, session_id=SESSION)
            print(f"ingested: {ingested}")
            # scope="ltm", not the "both" default: "both" answers out of the
            # session's own TURN buffer as well, so this run would go green
            # against an empty graph — the exact false pass it exists to catch.
            recalled = await memory.recall_memory(PHRASE, session_id=SESSION, scope="ltm")
        finally:
            await memory.drop_namespace()

    if not recalled["hits"]:
        sys.exit(
            f"FAIL: searching the graph for {PHRASE!r} returned nothing. An empty result is a "
            "failed acceptance run, not an answer — something the pipeline needed was absent."
        )
    print(f"PASS: {len(recalled['hits'])} hit(s) for {PHRASE!r}")


asyncio.run(main())
PY

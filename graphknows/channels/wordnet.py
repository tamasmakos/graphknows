"""WordNetChannel — flush-time backfill for the WordNet schema plane.

MFS grounding happens per-chunk at ingest (``_STMIngestHandler._link_senses``),
which mints a SYNSET/SENSE pair but leaves the synset's hypernym chain and the
sense-to-sense lexical relations unwritten — walking those at ingest time would
be a per-entity corpus round trip. This channel does that work once per flush
instead, scoped to the session being flushed: it finds the SYNSETs THIS
session's chunks grounded that are still missing their BROADER chain and
mints the shortest hypernym path to a root, then mints SENSE_REL edges among
this session's grounded senses. Scoping by session (rather than scanning
every SYNSET/SENSE in the namespace) keeps flush cost proportional to the
session just ingested, not to everything ever flushed before it.

Only ``populate`` is implemented; ``collect`` stays the inherited empty
default. PPR/sense-based retrieval over this plane is deliberately stubbed
until issue #135's read-side proves the plane is worth querying — this is not
an oversight, it is the same "write before you build a second read path"
discipline the module docstring in ``graphknows/symbolic/wordnet.py`` follows.
"""

from __future__ import annotations

import logging
from typing import Any

from graphknows.channels.base import Channel
from graphknows.symbolic.wordnet import WN_VERSION, ancestors, sense_relations

log = logging.getLogger(__name__)


class WordNetChannel(Channel):
    """Flush-time ancestor + sense-relation backfill (write-only; issue #142)."""

    name = "wordnet"

    async def populate(self, store: Any, session_id: str) -> None:
        """Backfill hypernym chains and sense relations for THIS session.

        Scoped via ``CHUNK.session_id`` -> ``HAS_SENSE.chunk_id`` -> ``SENSE``
        -> ``SENSE.synset``, not a namespace-wide scan: cost is proportional
        to the senses THIS session's chunks grounded, not to every SYNSET/
        SENSE vertex ever written to the namespace.

        Idempotent (every write below is a MERGE) and soft-fails on an
        unprovisioned corpus: ``ancestors``/``sense_relations`` degrade to
        empty rather than raising, so a missing WordNet corpus backfills
        nothing instead of failing the flush.
        """
        chunk_rows = await store.query(
            "MATCH (c:CHUNK {session_id: $session_id}) RETURN c.chunk_id AS id",
            session_id=session_id,
        )
        chunk_ids = [str(r["id"]) for r in chunk_rows]
        if not chunk_ids:
            return

        sense_rows = await store.query(
            "MATCH ()-[h:HAS_SENSE]->(s:SENSE) WHERE h.chunk_id IN $chunk_ids "
            "RETURN DISTINCT s.id AS id, s.synset AS synset",
            chunk_ids=chunk_ids,
        )
        sense_ids = {str(r["id"]) for r in sense_rows}
        synset_ids = list({str(r["synset"]) for r in sense_rows if r.get("synset")})

        missing = await store.synsets_missing_parents(synset_ids)
        edges: list[dict[str, Any]] = []
        truncations = 0
        for row in missing:
            chain, truncated = ancestors(row["name"])
            edges.extend(chain)
            if truncated:
                truncations += 1
        if edges:
            await store.write_synset_hypernyms(edges)

        relations = sense_relations(sense_ids) if sense_ids else []
        if relations:
            await store.write_sense_relations(relations)

        await store.write_source(
            {
                "channel": "wordnet",
                "version": WN_VERSION,
                "source_hash": WN_VERSION,
                "imported": len(missing),
            }
        )
        log.info(
            "wordnet backfill session=%s: synsets=%d chain_truncations=%d sense_relations=%d",
            session_id,
            len(missing),
            truncations,
            len(relations),
        )


__all__ = ["WordNetChannel"]

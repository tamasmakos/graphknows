"""ArcadeDB single-database graph store (golden layer).

One ``GraphStore`` per namespace database (``mem`` / ``mem_<ns>``) absorbs the
roles of the legacy STM, LTM, and vector stores: SESSION/FILE/CHUNK spine,
ENTITY graph keyed on ``name_norm``, dynamic relation edges with chunk-level
``evidence`` provenance, TEMPORAL/TOPIC/FRAME layers, and vector search over
embeddings stored directly on CHUNK/ENTITY/TOPIC vertices.

Conventions (schema-wide):
- ``type`` not ``label``; ``confidence`` not ``score``;
  ``valid_at``/``invalid_at``/``is_current`` for bi-temporal relation edges.
- Lifecycle ``state`` in {``raw``, ``consolidated``} on SESSION/CHUNK/ENTITY
  and relation edges; writes always land ``raw``.

ArcadeDB constraints honoured here:
- Cypher has no DDL surface — schema uses SQL.
- Embedding values MUST be written as inline SQL literals (LIST params
  conflict with the LSM_VECTOR index); vector search uses ``vectorNeighbors``
  with no FROM clause (see ``_vec_search_sql``).
- openCypher has no ``IN $list`` / ``any(x IN $list)`` — every interpolated
  string goes through :mod:`graphknows.storage.arcadedb._sql`.
- Dynamic relation edge types must be declared before MERGE.
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import UTC, datetime
from typing import Any

from graphknows.exceptions import StoreError
from graphknows.storage.arcadedb._base import ArcadeStoreBase
from graphknows.storage.arcadedb._schema import _CORE_DDL
from graphknows.storage.arcadedb._sql import is_missing_type, quote, quoted_list, sanitize
from graphknows.storage.arcadedb.client import ArcadeDBClient

logger = logging.getLogger(__name__)

_DB = "mem"

# Bulk CHUNK writes are capped per statement to keep statement/body size bounded
# (embeddings are large; see plan §8 statement-size risk). Tunable.
_CHUNK_BATCH = 20

# Vector-index dimension token, replaced with the real dimension in
# ensure_schema(dims) so the index matches the configured embedder.

# Graph OLAP analytical view backing `CALL algo.*` (pageRank/node2vec/louvain).
# Ported from the retired ltm_store._ensure_analytical_view — see that method's
# docstring for why the view is required (CSR engine vs. slow OLTP traversal).
_OLAP_VIEW = "mem_olap"

# Known single-valued relation predicates — a new fact tombstones the old edge
# (sets invalid_at / is_current=false) instead of coexisting with it.
# FrameNet frame names (RESIDENCE, BEING_EMPLOYED) are deliberately absent: they
# are frames, not relex predicates, and the flattened frame->ENTITY projection is
# gone (write_frame_instances is reified PLAYS_ROLE only), so they match nothing
# today and would wrongly tombstone sibling frame edges if it ever returns.
# Phase 3 consolidation replaces this hardcoded set with frame-semantics-driven
# functional detection.
_SINGLE_VALUED_RELATIONS: frozenset[str] = frozenset(
    {
        "WORKS_AS",
        "WORKS_AT",
        "EMPLOYED_BY",
        "LIVES_IN",
        "LOCATED_IN",
        "BASED_IN",
        "MARRIED_TO",
        "AFFILIATED_WITH",
        "BELONGS_TO",
        "LED_BY",
    }
)

# Vertex labels wiped by clear_all() / walked by delete_session().
_VERTEX_LABELS = (
    "FRAME_INSTANCE",
    "FRAME",
    "CHUNK",
    "FILE",
    "TOPIC",
    "TEMPORAL",
    "ENTITY",
    "SESSION",
)


def _emb_sql(emb: list[float]) -> str:
    """Format an embedding as an ArcadeDB SQL array literal (inline, not a param)."""
    return "[" + ",".join(str(x) for x in emb) + "]"


def _vec_search_sql(index: str, emb: list[float], k: int) -> str:
    """Return SQL for a vectorNeighbors ANN search.

    No FROM clause — the index already knows its type. Adding FROM <type>
    makes ArcadeDB invoke vectorNeighbors once *per vertex*, returning N*k rows
    and making deduplication order-dependent.
    """
    return f'SELECT expand(vectorNeighbors("{index}", {_emb_sql(emb)}, {k}))'


def _name_norm(name: str) -> str:
    """Canonical ENTITY merge key: casefolded, whitespace-collapsed name."""
    return " ".join(name.split()).casefold()


def verifier_coverage_rows(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Fold ``{source, verifier_score}`` REL rows into per-tier coverage.

    Returns ``{source: {"edges", "scored", "mean"}}``. ``mean`` averages only
    the scored edges and is ``None`` when a tier has none — never 0.0, because
    ``write_relation`` defines a NULL ``verifier_score`` as "the verifier never
    judged this triple", and folding that into 0.0 would read as "the verifier
    hated it". A row with no ``source`` buckets under ``"unknown"`` so an
    unattributed producer is still counted rather than silently dropped.
    """
    coverage: dict[str, dict[str, Any]] = {}
    totals: dict[str, float] = {}
    for row in rows:
        source = str(row.get("source") or "").strip() or "unknown"
        tier = coverage.setdefault(source, {"edges": 0, "scored": 0, "mean": None})
        tier["edges"] += 1
        score = row.get("verifier_score")
        if score is not None:
            tier["scored"] += 1
            totals[source] = totals.get(source, 0.0) + float(score)
    for source, tier in coverage.items():
        if tier["scored"]:
            tier["mean"] = round(totals[source] / tier["scored"], 4)
    return coverage


# How much a chunk tagged with a SUBCLASS of the matched class is worth. Below
# 1.0 so an exact match always wins; well above 0 because inheritance is real
# evidence, not a tiebreak — a Dance Studio genuinely is a Facility.
_SUBCLASS_DISCOUNT = 0.6


class GraphStore(ArcadeStoreBase):
    """Single-database ArcadeDB store for one namespace (golden layer).

    Args:
        client: Shared ``ArcadeDBClient`` (connect() is idempotent).
        db: Database name, from :func:`graphknows.storage.namespace.db_name`.
    """

    def __init__(self, client: ArcadeDBClient, db: str = _DB) -> None:
        super().__init__(client, db)

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    async def ensure_schema(self, dims: int) -> None:
        """Create the database and the golden-layer schema idempotently.

        Args:
            dims: Embedding dimension for the LSM_VECTOR indexes (must match
                the configured embedder).
        """
        await self._apply_ddl(_CORE_DDL, dims=dims)
        logger.info("GraphStore schema ensured (db=%s, dims=%d)", self._db, dims)

    async def _set_embedding(
        self, vertex_type: str, key_prop: str, key: str, emb: list[float]
    ) -> None:
        """Set an embedding on a vertex matched by its key property.

        The match key is a Cypher parameter (handles quotes/newlines safely);
        the embedding stays an inline literal because LIST params conflict with
        the LSM_VECTOR index.
        """
        if not emb:
            return
        stmt = f"MATCH (n:{vertex_type} {{{key_prop}: $key}}) SET n.embedding = {_emb_sql(emb)}"  # nosec B608
        await self._c.command(self._db, stmt, params={"key": key})

    @staticmethod
    def _dedup_rows(rows: list[dict]) -> list[dict]:
        """Deduplicate vectorNeighbors results by @rid."""
        seen: set[str] = set()
        out: list[dict] = []
        for r in rows:
            rid = r.get("@rid", r.get("chunk_id", ""))
            if rid not in seen:
                seen.add(rid)
                out.append(r)
        return out

    # ------------------------------------------------------------------
    # Write API — spine (SESSION / FILE / CHUNK)
    # ------------------------------------------------------------------

    async def ensure_session(self, session_id: str, started_at: str = "") -> None:
        """Upsert a SESSION vertex; state starts ``raw``."""
        await self.command(
            "MERGE (s:SESSION {session_id: $sid}) "
            "SET s.started_at = coalesce(s.started_at, $sa), "
            "    s.state = coalesce(s.state, 'raw')",
            sid=session_id,
            sa=started_at,
        )

    async def ensure_file(self, file_id: str, session_id: str) -> None:
        """Upsert a FILE vertex linked to its SESSION via FILE-[:IN_SESSION]->SESSION."""
        await self.command(
            "MERGE (f:FILE {file_id: $fid}) "
            "SET f.session_id = $sid "
            "WITH f MATCH (s:SESSION {session_id: $sid}) "
            "MERGE (f)-[:IN_SESSION]->(s)",
            fid=file_id,
            sid=session_id,
        )

    async def write_chunk(self, chunk: dict[str, Any]) -> None:
        """Upsert one CHUNK vertex (state='raw') with its PART_OF edge and embedding.

        ``chunk`` keys: chunk_id (required), session_id, file_id, text, seq,
        kind, speaker, ts, memory_class, embedding.
        """
        await self.write_chunks_bulk([chunk])

    async def write_chunks_bulk(self, chunks: list[dict[str, Any]]) -> None:
        """Upsert CHUNK vertices in batches of ``_CHUNK_BATCH`` per statement.

        Per batch: one UNWIND MERGE for scalar properties, one UNWIND MERGE for
        PART_OF edges. Embeddings are written per chunk afterwards — they must
        be inline SQL literals, so they cannot ride in the batched params.
        """
        if not chunks:
            return
        now = datetime.now(UTC).isoformat()
        for start in range(0, len(chunks), _CHUNK_BATCH):
            batch = chunks[start : start + _CHUNK_BATCH]
            rows = [
                {
                    "cid": c["chunk_id"],
                    "sid": c.get("session_id", ""),
                    "fid": c.get("file_id", ""),
                    "text": c.get("text", ""),
                    "stext": c.get("search_text") or c.get("text", ""),
                    "seq": int(c.get("seq") or 0),
                    "kind": c.get("kind", "text"),
                    "speaker": c.get("speaker", "") or "",
                    "ts": c.get("ts", "") or "",
                    "mc": c.get("memory_class", "") or "",
                }
                for c in batch
            ]
            await self.command(
                "UNWIND $rows AS row "
                "MERGE (c:CHUNK {chunk_id: row.cid}) "
                "SET c.session_id = row.sid, c.text = row.text, c.seq = row.seq, "
                "    c.search_text = row.stext, "
                "    c.kind = row.kind, c.speaker = row.speaker, c.ts = row.ts, "
                "    c.memory_class = row.mc, "
                "    c.state = coalesce(c.state, 'raw'), "
                "    c.created_at = coalesce(c.created_at, $now)",
                rows=rows,
                now=now,
            )
            edge_rows = [r for r in rows if r["fid"]]
            if edge_rows:
                await self.command(
                    "UNWIND $rows AS row "
                    "MATCH (c:CHUNK {chunk_id: row.cid}), (f:FILE {file_id: row.fid}) "
                    "MERGE (c)-[:PART_OF]->(f)",
                    rows=edge_rows,
                )
            # CHUNK-[:IN_SESSION]->SESSION, from the session_id already on the
            # row. TURN has carried this edge since the buffer landed; without
            # it a chunk reached its session only via PART_OF->FILE->IN_SESSION,
            # which does not exist for a chunk written with no file_id. MATCH,
            # not MERGE, on the SESSION: the caller creates it (ensure_session),
            # and merging here would mint an empty-id vertex for any chunk that
            # legitimately has no session.
            session_rows = [r for r in rows if r["sid"]]
            if session_rows:
                await self.command(
                    "UNWIND $rows AS row "
                    "MATCH (c:CHUNK {chunk_id: row.cid}) "
                    "MATCH (s:SESSION {session_id: row.sid}) "
                    "MERGE (c)-[:IN_SESSION]->(s)",
                    rows=session_rows,
                )
            for c in batch:
                if c.get("embedding"):
                    await self._set_embedding("CHUNK", "chunk_id", c["chunk_id"], c["embedding"])

    # ------------------------------------------------------------------
    # Write API — TURN (short-term conversation buffer)
    # ------------------------------------------------------------------
    #
    # TURN is the cheap short-term-memory tier: one vertex per conversation
    # turn, written with no extraction and no embedding. flush() later ingests
    # the buffered raw turns through the real pipeline and flips them
    # raw -> consolidated via mark_turns_consolidated. The caller is
    # responsible for ensure_session() first — write_turn does not create it.

    async def write_turn(
        self,
        session_id: str,
        role: str,
        text: str,
        seq: int,
        ts: str = "",
        speaker: str = "",
        infer: bool = True,
    ) -> str:
        """Append one raw TURN to the session's short-term buffer; returns the turn_id.

        ``speaker`` and ``infer`` are carried so ``flush`` can reproduce the
        caller's original intent: who said it, and whether this turn was meant
        to be extracted from. Both are lost otherwise — the buffer would flatten
        every turn onto one shared decision made at flush time.
        """
        turn_id = f"{session_id}:{seq}"
        ts = ts or datetime.now(UTC).isoformat()
        await self.command(
            "MERGE (t:TURN {turn_id: $tid}) "
            "SET t.session_id = $sid, t.role = $role, t.text = $text, "
            "    t.seq = $seq, t.ts = $ts, t.speaker = $speaker, t.infer = $infer, "
            "    t.state = coalesce(t.state, 'raw') "
            "WITH t MATCH (s:SESSION {session_id: $sid}) "
            "MERGE (t)-[:IN_SESSION]->(s)",
            tid=turn_id,
            sid=session_id,
            role=role,
            text=text,
            seq=seq,
            ts=ts,
            speaker=speaker,
            infer=infer,
        )
        return turn_id

    async def next_turn_seq(self, session_id: str) -> int:
        """Next 1-based sequence number for this session's buffer (1 when empty)."""
        rows = await self.query(
            "MATCH (t:TURN {session_id: $sid}) RETURN max(t.seq) AS m",
            sid=session_id,
        )
        m = rows[0].get("m") if rows else None
        return int(m or 0) + 1

    async def list_turns(
        self, session_id: str, *, state: str = "raw", limit: int = 30
    ) -> list[dict[str, Any]]:
        """Return the session's buffered turns, newest first (ORDER BY seq DESC)."""
        rows = await self.query(
            "MATCH (t:TURN {session_id: $sid}) WHERE t.state = $state "
            "RETURN t.turn_id AS turn_id, t.session_id AS session_id, t.role AS role, "
            "       t.text AS text, t.seq AS seq, t.ts AS ts, t.state AS state, "
            "       t.speaker AS speaker, t.infer AS infer "
            "ORDER BY t.seq DESC LIMIT $limit",
            sid=session_id,
            state=state,
            limit=limit,
        )
        return rows

    async def mark_turns_consolidated(self, session_id: str) -> int:
        """Flip this session's raw TURNs to consolidated; returns how many flipped."""
        # Count first (query endpoint rejects a mutating SET...RETURN as
        # non-idempotent), then flip via the command endpoint. Same pattern as
        # consolidate_session.
        rows = await self.query(
            "MATCH (t:TURN {session_id: $sid}) WHERE t.state = 'raw' RETURN count(t) AS n",
            sid=session_id,
        )
        await self.command(
            "MATCH (t:TURN {session_id: $sid}) WHERE t.state = 'raw' SET t.state = 'consolidated'",
            sid=session_id,
        )
        flipped = int(rows[0].get("n") or 0) if rows else 0
        logger.info("mark_turns_consolidated %s: flipped %d turns", session_id, flipped)
        return flipped

    # ------------------------------------------------------------------
    # Write API — entities / relations
    # ------------------------------------------------------------------

    async def write_entities_bulk(
        self,
        session_id: str,
        chunk_id: str,
        entities: list[dict[str, Any]],
    ) -> None:
        """Upsert ENTITY vertices (MERGE on ``name_norm``) + MENTIONS edges.

        Callers pre-filter entities through the writer-layer filter
        (``graph_entities`` — drops temporal/numeric spans); this store does
        not duplicate that policy.

        Semantics per entity (dict with "name"; optional "type", "confidence"):
        - MERGE key is the casefolded name; ``name`` keeps the first-seen surface.
        - When a later surface differs from the canonical name it is appended
          to ``aliases`` (deduplicated).
        - ``confidence`` keeps the max seen; ``state`` starts ``raw``.
        - MENTIONS edge CHUNK->ENTITY carries ``confidence``.
        """
        if not entities:
            return
        # Collapse rows on the MERGE key before they reach the UNWIND. Two rows
        # sharing a name_norm are not a harmless repeat: the UNWIND MERGEs both
        # before the UNIQUE index settles, ArcadeDB raises DuplicatedKeyException,
        # and the caller's WHOLE DOCUMENT fails — silently deleting every chunk in
        # it from the graph. Extractors that emit overlapping spans (the same
        # surface under several labels) hit this routinely, so the guard belongs
        # here, at the write, rather than in each caller. Max confidence wins.
        by_key: dict[str, dict[str, Any]] = {}
        for e in entities:
            name = str(e.get("name") or "").strip()
            if not name:
                continue
            conf = e.get("confidence", e.get("score", 1.0))
            # Alias candidates are the surfaces overlap resolution discarded
            # ("offers"/"new" folded into "new offers"). They are the only record
            # the shorter surface was ever seen, since resolution stops it
            # reaching this writer as an entity of its own — drop them here and a
            # later chunk saying just "offers" can never match this node.
            aliases = [str(a).strip() for a in (e.get("aliases") or []) if str(a).strip()]
            row = {
                "nn": _name_norm(name),
                "name": name,
                "etype": e.get("type") or "ENTITY",
                "conf": float(conf) if isinstance(conf, int | float) else 1.0,
                "aliases": [a for a in dict.fromkeys(aliases) if a != name],
            }
            key = str(row["nn"])
            prev = by_key.get(key)
            if prev is None or row["conf"] > prev["conf"]:
                by_key[key] = row
        rows: list[dict[str, Any]] = list(by_key.values())
        if not rows:
            return

        # Pass 1: upsert ENTITY nodes on name_norm.
        await self.command(
            "UNWIND $rows AS row "
            "MERGE (e:ENTITY {name_norm: row.nn}) "
            "SET e.name = coalesce(e.name, row.name), "
            "    e.type = coalesce(e.type, row.etype), "
            "    e.state = coalesce(e.state, 'raw'), "
            "    e.confidence = CASE "
            "        WHEN row.conf > coalesce(e.confidence, 0) THEN row.conf "
            "        ELSE e.confidence "
            "    END, "
            # Two alias sources union in here: a different surface arriving under
            # the same name_norm (row.name), and the overlap-resolution candidates
            # the extractor folded into this row (row.aliases). The comprehension
            # is what makes the union idempotent — the same chunk re-ingested must
            # not grow the list, and ArcadeDB has no set type. Verified on 26.5.1:
            # three writes of overlapping candidate sets yield exactly the union.
            "    e.aliases = CASE "
            "        WHEN row.name <> e.name AND NOT row.name IN coalesce(e.aliases, []) "
            "        THEN coalesce(e.aliases, []) + [row.name] "
            "             + [a IN row.aliases WHERE NOT a IN coalesce(e.aliases, [])] "
            "        ELSE coalesce(e.aliases, []) "
            "             + [a IN row.aliases WHERE NOT a IN coalesce(e.aliases, [])] "
            "    END",
            rows=rows,
        )

        # Pass 2: MENTIONS edges with confidence.
        await self.command(
            "MATCH (c:CHUNK {chunk_id: $cid}) "
            "UNWIND $rows AS row "
            "MATCH (e:ENTITY {name_norm: row.nn}) "
            "MERGE (c)-[m:MENTIONS]->(e) "
            "SET m.confidence = CASE "
            "    WHEN row.conf > coalesce(m.confidence, 0) THEN row.conf "
            "    ELSE m.confidence "
            "END",
            cid=chunk_id,
            rows=rows,
        )

    async def _supersede_edge(self, predicate: str, src_norm: str, dst_norm: str, now: str) -> None:
        """Tombstone any current edge for (src, predicate) pointing elsewhere.

        Called only for single-valued predicates. Sets ``invalid_at``/
        ``is_current=false`` instead of deleting, preserving history.
        """
        try:
            await self.command(
                "MATCH (a:ENTITY {name_norm: $src})-[r:REL]->(b:ENTITY) "
                "WHERE r.predicate = $predicate AND r.is_current = true "
                "AND b.name_norm <> $dst "
                "SET r.invalid_at = $now, r.is_current = false",
                src=src_norm,
                dst=dst_norm,
                predicate=predicate,
                now=now,
            )
        except Exception as exc:
            logger.debug(
                "_supersede_edge %s (%s -> !=%s) skipped: %s", predicate, src_norm, dst_norm, exc
            )

    async def write_relation(
        self,
        src_name: str,
        predicate: str,
        dst_name: str,
        fact: str = "",
        chunk_id: str = "",
        confidence: float = 0.5,
        valid_at: str | None = None,
        source: str = "",
        src_type: str = "",
        dst_type: str = "",
        ontology_property: str = "",
        verifier_score: float | None = None,
        ontology_blocked: bool = False,
        speaker: str = "",
        is_current: bool = True,
        invalid_at: str | None = None,
    ) -> None:
        """MERGE a dynamic relation edge (type = taxonomy predicate).

        On a duplicate (src, predicate, dst) the ``chunk_id`` is appended to the
        edge's ``evidence`` list (deduplicated) and ``confidence`` keeps the max.
        Single-valued predicates tombstone the previous current edge first.

        ``src_type``/``dst_type`` type the MERGEd ENTITY endpoints (NER label, else
        WordNet supersense). They are ``coalesce``d, so an existing type is never
        overwritten — a later untyped mention cannot erase an earlier NER label.
        Without them the endpoints of a relation are created untyped, which is how
        the great majority of ENTITY vertices ended up with no type at all.

        ``ontology_property`` annotates the edge with the ontology property its
        predicate mapped onto, when an ontology is configured. It is an
        annotation only — an unmapped predicate is written unchanged. (The
        companion ``map_score`` was removed: it was written on every edge and
        read by nothing.)

        ``ontology_blocked`` records that a caller's range gate REFUSED to
        annotate this edge, and once true it is sticky: no later write can
        annotate it. The distinction matters because the gate runs per chunk and
        the same edge is written once per chunk asserting it. Measured on
        conv-30, (Jon, HAS, students) is written five times — four chunks type
        ``students`` PERSON and refuse, one types it POSSESSION and admits — and
        under plain coalesce the single admit won, publishing `Jon owns
        students`. A refusal is a verdict, not an absence of one.

        ``verifier_score`` is the relation verifier's judgment of this triple
        (see relations/verifier.py). It merges like ``confidence`` — the max
        wins — but is left NULL when the verifier never scored the relation
        (gate disabled, or a surface it could not locate and so passed through),
        so "unscored" stays distinguishable from "scored 0.0". It annotates
        only; ``confidence`` is unchanged by it.

        ``is_current`` says whether this assertion is live. False means the
        caller is replaying an already-tombstoned edge (see _repoint_relation),
        so the edge must keep its tombstone: ``is_current`` is never raised to
        true and the death date is carried over in ``invalid_at`` (which a live
        write ignores, clearing the tombstone instead). Without the flag every
        replay through this shared write path resurrects dead edges; without the
        date the record keeps the death but loses its when.
        """
        predicate = predicate.strip()
        if not predicate or not src_name or not dst_name:
            return
        src_norm = _name_norm(src_name)
        dst_norm = _name_norm(dst_name)
        now = datetime.now(UTC).isoformat()

        # A replayed dead edge must not tombstone the winner's live sibling.
        if is_current and predicate.upper() in _SINGLE_VALUED_RELATIONS:
            await self._supersede_edge(predicate, src_norm, dst_norm, now)

        # Evidence is appended via an interpolated quoted literal — openCypher
        # cannot express "$param IN list-property" reliably.
        cid_lit = quote(chunk_id) if chunk_id else ""
        evidence_set = (
            f"r.evidence = CASE "
            f"WHEN {cid_lit} IN coalesce(r.evidence, []) THEN r.evidence "
            f"ELSE coalesce(r.evidence, []) + [{cid_lit}] END, "
            if chunk_id
            else "r.evidence = coalesce(r.evidence, []), "
        )
        # Omitted entirely when unscored, so the property stays NULL rather than
        # collapsing "the verifier never saw this" into "the verifier hated it".
        # Floor is -1.0, not 0, so a legitimate 0.0 score still writes.
        verifier_set = (
            "r.verifier_score = CASE "
            "WHEN $vscore > coalesce(r.verifier_score, -1.0) THEN $vscore "
            "ELSE r.verifier_score END, "
            if verifier_score is not None
            else ""
        )
        # A live assertion clears any tombstone; a replay leaves invalid_at alone
        # and cannot raise is_current, so a dead edge stays dead.
        current_set = (
            "r.invalid_at = null, r.is_current = true, "
            if is_current
            else "r.is_current = coalesce(r.is_current, false), "
            "r.invalid_at = coalesce(r.invalid_at, $invalid_at), "
        )
        await self.command(
            f"MERGE (a:ENTITY {{name_norm: $src_nn}}) "
            f"SET a.name = coalesce(a.name, $src), a.state = coalesce(a.state, 'raw'), "
            f"    a.type = coalesce(a.type, $src_type) "
            f"MERGE (b:ENTITY {{name_norm: $dst_nn}}) "
            f"SET b.name = coalesce(b.name, $dst), b.state = coalesce(b.state, 'raw'), "
            f"    b.type = coalesce(b.type, $dst_type) "
            f"MERGE (a)-[r:REL {{predicate: $predicate}}]->(b) "
            f"SET r.predicate = $predicate, "
            f"    r.fact = coalesce(r.fact, $fact), "
            f"    {evidence_set}"
            f"    r.confidence = CASE "
            f"        WHEN $conf > coalesce(r.confidence, 0) THEN $conf "
            f"        ELSE r.confidence "
            f"    END, "
            f"    r.valid_at = coalesce(r.valid_at, $valid_at), "
            f"    {current_set}"
            f"    r.state = coalesce(r.state, 'raw'), "
            f"    r.source = coalesce(r.source, $source), "
            f"    r.ontology_blocked = coalesce(r.ontology_blocked, false) OR $ont_blocked, "
            f"    r.ontology_property = CASE "
            f"        WHEN coalesce(r.ontology_blocked, false) OR $ont_blocked THEN null "
            f"        ELSE coalesce(r.ontology_property, $ont_prop) END, "
            f"    {verifier_set}"
            f"    r.created_at = coalesce(r.created_at, $now)",
            src_nn=src_norm,
            dst_nn=dst_norm,
            src=src_name,
            dst=dst_name,
            predicate=predicate,
            fact=fact,
            conf=float(confidence),
            valid_at=valid_at or None,
            source=source,
            src_type=src_type or None,
            dst_type=dst_type or None,
            ont_prop=ontology_property or None,
            ont_blocked=bool(ontology_blocked),
            vscore=None if verifier_score is None else float(verifier_score),
            invalid_at=invalid_at or None,
            now=now,
        )

        # Provenance for the endpoints this call just MERGEd. Without it an
        # entity that only ever appears as a relation argument has no edge back
        # to any chunk: measured on conv-30, 4 of 49 relation endpoints
        # ('clothes store', 'running', 'stress buster', 'target audience') were
        # reachable from no CHUNK at all, so nothing could explain where they
        # came from and the entity->chunk retrieval path could not see them.
        # The chunk is already known here — it is the same one recorded in
        # r.evidence — so this adds provenance, never a new claim.
        # The chunk's SPEAKER is excluded. A relation head is usually the speaker
        # resolved from metadata ("my collection" -> Gina), and that relationship
        # is already carried by CHUNK.speaker — typed, and stronger than
        # MENTIONS. Writing it again here duplicated it in a weaker form and
        # attached the two conversation participants to chunks that never name
        # them: measured on conv-30, an unfiltered version of this added 62
        # MENTIONS edges to Jon and Gina, the two most saturated entities in the
        # graph and precisely the ones retrieval already has to de-weight.
        mention_norms = [n for n in (src_norm, dst_norm) if n != _name_norm(speaker or "")]
        if chunk_id and mention_norms:
            await self.command(
                "MATCH (c:CHUNK {chunk_id: $cid}) "
                "MATCH (e:ENTITY) WHERE e.name_norm IN $norms "
                "MERGE (c)-[m:MENTIONS]->(e) "
                "SET m.confidence = coalesce(m.confidence, $conf)",
                cid=chunk_id,
                norms=mention_norms,
                conf=float(confidence),
            )

    # ------------------------------------------------------------------
    # Write API — temporal / frames / topics
    # ------------------------------------------------------------------

    async def write_temporal(self, chunk_id: str, iso_dates: list[str]) -> None:
        """MERGE TEMPORAL nodes (key: ISO ``date`` string) + MENTIONS_DATE edges."""
        dates = [str(d).strip() for d in iso_dates if str(d).strip()]
        if not dates:
            return
        await self.command(
            "MATCH (c:CHUNK {chunk_id: $cid}) "
            "UNWIND $dates AS iso "
            "MERGE (d:TEMPORAL {date: iso}) "
            "MERGE (c)-[:MENTIONS_DATE]->(d)",
            cid=chunk_id,
            dates=dates,
        )

    async def write_frames(self, session_id: str, frames: list[dict[str, Any]]) -> int:
        """Bulk-write FRAME vertices + EVOKED_BY evidence edges to their chunks.

        Each ``frames`` item: ``{frame_type, chunk_id, lu, speaker,
        fes: [[role, filler], ...], fillers: [str, ...], confidence?, summary?,
        embedding?}``. ``id`` is deterministic per (session, chunk, index) so
        re-runs MERGE idempotently. Evidence is the FRAME-[:EVOKED_BY]->CHUNK
        edge — there is no ``chunk_id`` property on FRAME. Embeddings are written
        per row as inline literals (LIST params conflict with the LSM_VECTOR index).
        """
        if not frames:
            return 0
        rows = [
            {
                "id": f"{session_id}:{f['chunk_id']}:{i}",
                "sid": session_id,
                "ft": f["frame_type"],
                "cid": f["chunk_id"],
                "lu": f.get("lu", ""),
                "spk": f.get("speaker", "") or "",
                "fes": json.dumps(f.get("fes", [])),
                "fillers": [str(x).lower() for x in f.get("fillers", [])],
                "conf": float(f.get("confidence", 1.0)),
                "summary": f.get("summary", "") or "",
            }
            for i, f in enumerate(frames)
        ]
        await self.command(
            "UNWIND $rows AS r "
            "MERGE (n:FRAME {id: r.id}) "
            "SET n.session_id = r.sid, n.frame_type = r.ft, "
            "n.lu = r.lu, n.speaker = r.spk, n.fes = r.fes, n.fillers = r.fillers, "
            "n.summary = r.summary",
            rows=rows,
        )
        await self.command(
            "UNWIND $rows AS r "
            "MATCH (n:FRAME {id: r.id}), (c:CHUNK {chunk_id: r.cid}) "
            "MERGE (n)-[ev:EVOKED_BY]->(c) "
            "SET ev.confidence = r.conf",
            rows=rows,
        )
        for i, f in enumerate(frames):
            emb = f.get("embedding")
            if emb:
                await self._set_embedding("FRAME", "id", f"{session_id}:{f['chunk_id']}:{i}", emb)
        return len(rows)

    async def write_ontology(self, index: Any) -> int:
        """Persist an OntologyIndex as ONTOLOGY + CLASS/PROPERTY vertices (idempotent).

        Reference data, written once per namespace at schema-ensure time rather
        than on the ingest hot path. Term definition embeddings are copied from
        the already-built index matrix, never recomputed. Guarded by a count
        check so a restart does not re-MERGE hundreds of rows.

        Returns the number of terms present after the write.
        """
        terms = getattr(index, "terms", None)
        if not terms:
            return 0
        ont_id = getattr(index, "digest", "") or "default"
        source = getattr(index, "source", "")

        existing = await self.query(
            "MATCH (o:ONTOLOGY {id: $id}) RETURN o.term_count AS c", id=ont_id
        )
        if existing and int(existing[0].get("c") or 0) >= len(terms):
            return len(terms)

        await self.command(
            "MERGE (o:ONTOLOGY {id: $id}) SET o.source = $src, o.term_count = $n",
            id=ont_id,
            src=source,
            n=len(terms),
        )
        # Only classes are persisted: ONTOLOGY_CLASS is what search_ontology_classes
        # reads. The ONTOLOGY_PROPERTY vertices, the DECLARES edges and the
        # SUBCLASS_OF hierarchy were written here and read by nothing.
        class_terms = [t for t in terms if t.kind == "class"]
        # ``module`` is the sub-ontology the term is curated in ("AgentOntology")
        # and ``parents`` its declared superclasses, as labels. Both are
        # descriptive metadata on the node, not a second hierarchy: nothing
        # selects on them, they make a match traceable to what supplied it. CCO's
        # IRIs are opaque (``ont00001774``), so without the module a persisted
        # tag cannot be attributed to a source at all.
        rows = [
            {
                "uri": t.uri,
                "label": t.label,
                "definition": t.definition,
                "module": t.module,
                "parents": ", ".join(t.parents),
            }
            for t in class_terms
        ]
        if rows:
            await self.command(
                "UNWIND $rows AS r MERGE (n:ONTOLOGY_CLASS {uri: r.uri}) "
                "SET n.label = r.label, n.definition = r.definition, "
                "n.module = r.module, n.parents = r.parents",
                rows=rows,
            )
        # Embeddings are per-row inline literals: LIST params conflict with the
        # LSM_VECTOR index, the same constraint CHUNK/ENTITY/FRAME honour.
        for term in class_terms:
            emb = index.term_embedding(term.uri)
            if emb:
                await self._set_embedding("ONTOLOGY_CLASS", "uri", term.uri, emb)
        logger.info("Ontology persisted: %d terms from %s", len(terms), source)
        return len(terms)

    async def write_class_matches(self, chunk_id: str, matches: list[tuple[str, float]]) -> int:
        """MERGE CHUNK-[:MATCHES_CLASS {sim}]->ONTOLOGY_CLASS edges for one chunk."""
        if not matches:
            return 0
        rows = [{"uri": uri, "sim": float(sim)} for uri, sim in matches]
        await self.command(
            "UNWIND $rows AS r "
            "MATCH (c:CHUNK {chunk_id: $cid}), (o:ONTOLOGY_CLASS {uri: r.uri}) "
            "MERGE (c)-[m:MATCHES_CLASS]->(o) SET m.sim = r.sim",
            rows=rows,
            cid=chunk_id,
        )
        return len(rows)

    async def write_instance_of(self, mappings: list[dict[str, Any]]) -> int:
        """MERGE ENTITY-[:INSTANCE_OF {score}]->ONTOLOGY_CLASS edges.

        Each mapping: ``{name, uri, score}``, where ``score`` is *that entity's*
        confidence in being an instance of that class — not the class's own match
        score. An entity may accumulate several INSTANCE_OF edges — different
        chunks extract different types, which is legitimate multi-classing.
        Repeated writes keep the strongest mention rather than the latest, so the
        stored score does not depend on chunk order.
        """
        if not mappings:
            return 0
        rows = [
            {
                "nn": _name_norm(str(m["name"])),
                "uri": m["uri"],
                "score": float(m.get("score", 0.0)),
            }
            for m in mappings
            if m.get("name") and m.get("uri")
        ]
        if not rows:
            return 0
        await self.command(
            "UNWIND $rows AS r "
            "MATCH (e:ENTITY {name_norm: r.nn}), (o:ONTOLOGY_CLASS {uri: r.uri}) "
            "MERGE (e)-[i:INSTANCE_OF]->(o) "
            "SET i.score = CASE "
            "    WHEN r.score > coalesce(i.score, 0) THEN r.score "
            "    ELSE i.score "
            "END",
            rows=rows,
        )
        return len(rows)

    async def search_ontology_classes(
        self,
        embedding: list[float],
        top_k: int,
        session_id: str | None = None,
        state: str | None = None,
    ) -> list[dict[str, Any]]:
        """ANN over ONTOLOGY_CLASS.embedding, joined to the chunks that evoke it.

        Mirrors :meth:`query_frames_by_embedding`: vector-search the class
        DEFINITIONS persisted in this namespace, then traverse MATCHES_CLASS to
        the chunks tagged with them, scoring each chunk by its class's
        query-match. Returns ``[{chunk_id, label, uri, score}]``.
        """
        if not embedding:
            return []
        sql = _vec_search_sql("ONTOLOGY_CLASS[embedding]", embedding, top_k)
        try:
            rows = await self._c.query(self._db, sql, language="sql")
        except Exception as exc:
            logger.warning("search_ontology_classes failed: %s", exc)
            return []
        score: dict[str, float] = {}
        label: dict[str, str] = {}
        for r in rows:
            uri = r.get("uri")
            if not uri or uri in score:
                continue
            score[uri] = max(0.0, 1.0 - float(r.get("distance") or 1.0))
            label[uri] = r.get("label", "")
        if not score:
            return []
        score |= await self._subclass_scores(score, label)
        where = f"o.uri IN {quoted_list(list(score))}"
        if session_id:
            where += f" AND c.session_id = {quote(session_id)}"
        if state:
            where += f" AND c.state = {quote(state)}"

        out: list[dict[str, Any]] = []
        # The chunk itself evoked the class (direct, strongest). A second route
        # through ENTITY-[:INSTANCE_OF]->CLASS was measured byte-identical on
        # conv-26 evidence_recall once the pool is cosine-reranked, and removed.
        try:
            direct = await self.query(
                f"MATCH (c:CHUNK)-[m:MATCHES_CLASS]->(o:ONTOLOGY_CLASS) WHERE {where} "
                "RETURN c.chunk_id AS chunk_id, o.uri AS uri, m.sim AS sim"
            )
        except Exception as exc:
            logger.warning("search_ontology_classes chunk join failed: %s", exc)
            direct = []
        for r in direct:
            if r.get("chunk_id"):
                out.append(
                    {
                        "chunk_id": r["chunk_id"],
                        "uri": r["uri"],
                        "label": label.get(r["uri"], ""),
                        # Query-to-class match, weighted by how strongly the
                        # chunk evoked that class in the first place.
                        "score": score.get(r["uri"], 0.0) * float(r.get("sim") or 0.0),
                        "via": "chunk",
                    }
                )
        return out

    async def _subclass_scores(
        self, matched: dict[str, float], label: dict[str, str]
    ) -> dict[str, float]:
        """Descendant classes of the matched ones, at a discount.

        A query that matches ``Facility`` should reach a chunk tagged
        ``Dance Studio``: the chunk is about a facility, it just never used the
        word. That generalisation is the whole reason the ontology carries a
        hierarchy, and without this the channel can only ever return chunks
        whose tag the query matched head-on.

        One level, because ``parents`` is stored per node and each extra level
        costs another round trip for rapidly thinning evidence. Scored at
        ``_SUBCLASS_DISCOUNT`` of the parent's match, so a direct hit always
        outranks an inherited one.
        """
        names = {lbl for lbl in label.values() if lbl}
        if not names:
            return {}
        try:
            rows = await self.query(
                "MATCH (o:ONTOLOGY_CLASS) WHERE o.parents IS NOT NULL AND o.parents <> '' "
                "RETURN o.uri AS uri, o.label AS label, o.parents AS parents"
            )
        except Exception as exc:
            logger.warning("subclass expansion failed: %s", exc)
            return {}
        by_label = {v: k for k, v in label.items()}
        out: dict[str, float] = {}
        for r in rows:
            uri = str(r.get("uri") or "")
            if not uri or uri in matched:
                continue
            parents = {p.strip() for p in str(r.get("parents") or "").split(",")}
            if hit := (parents & names):
                parent_uri = by_label.get(next(iter(hit)), "")
                out[uri] = matched.get(parent_uri, 0.0) * _SUBCLASS_DISCOUNT
                label[uri] = str(r.get("label") or "")
        if out:
            logger.debug("ontology query expanded to %d subclasses", len(out))
        return out

    async def write_frame_instances(
        self, session_id: str, chunk_id: str, instances: list[dict[str, Any]]
    ) -> int:
        """MERGE role-filled FRAME_INSTANCE vertices + PLAYS_ROLE edges.

        Each ``instances`` item: ``{frame, match_sim, role_confidence,
        roles: {role_name: [filler_name, ...]}}``.

        A FRAME_INSTANCE is a FACT, not a per-chunk artifact: its id is a content
        hash of ``frame`` + the sorted role/filler signature, so the same event
        stated in several chunks MERGEs onto ONE node accumulating a
        FRAME_EVOKED_IN edge per evidence chunk (the same model RELATION.evidence
        uses) — instead of a duplicate node per chunk. Any ``id`` on the input is
        ignored. Writes are batched: one UNWIND for the instances+evidence edges,
        one for all PLAYS_ROLE edges (ENTITY fillers MERGE on ``name_norm``).

        The reified (FRAME_INSTANCE)-[:PLAYS_ROLE {role}]->(ENTITY) form is the
        ONLY representation. A flattened ENTITY->ENTITY projection (predicate
        ``f"{FRAME}_{ROLE}"``) was written here too and has been removed: it was a
        lossy re-encoding of these same PLAYS_ROLE edges (it only linked one
        agent-like filler to the others, so it lost fills), it minted one edge
        TYPE per frame/role pair, and ``entity_fact_context`` reads the reified
        form directly — so the flat copy only re-stated the same facts in a second
        format that the fact-sheet dedup could not match, consuming the cap twice.

        Idempotent (all MERGE). Returns the number of instances processed.
        """
        if not instances:
            return 0
        inst_rows: list[dict[str, Any]] = []
        role_rows: list[dict[str, Any]] = []
        for inst in instances:
            frame = str(inst.get("frame") or "").strip()
            roles: dict[str, list[str]] = inst.get("roles") or {}
            if not frame:
                continue
            # Content-hash id: same (frame + role fills) -> same node across chunks.
            sig = sorted(
                f"{r}={_name_norm(str(f))}"
                for r, fs in roles.items()
                for f in (fs or [])
                if str(f).strip()
            )
            if not sig:
                continue
            # Content hash for a stable frame-instance id — not a security digest.
            digest = hashlib.sha1("|".join(sig).encode(), usedforsecurity=False).hexdigest()
            fid = f"{frame}:{digest[:16]}"
            role_conf = float(inst.get("role_confidence") or 0.0)
            inst_rows.append(
                {
                    "fid": fid,
                    "frame": frame,
                    "cid": chunk_id,
                    "sid": session_id,
                    "ms": float(inst.get("match_sim") or 0.0),
                    "rc": role_conf,
                }
            )
            for role, fillers in roles.items():
                for filler in fillers or []:
                    name = str(filler).strip()
                    if name:
                        role_rows.append(
                            {"fid": fid, "nn": _name_norm(name), "name": name, "role": role}
                        )
        if not inst_rows:
            return 0
        # Batch 1: FRAME_INSTANCE + FRAME_EVOKED_IN evidence edge (per chunk).
        await self.command(
            "UNWIND $rows AS row "
            "MATCH (c:CHUNK {chunk_id: row.cid}) "
            "MERGE (fi:FRAME_INSTANCE {id: row.fid}) "
            "SET fi.frame = row.frame, fi.session_id = row.sid, "
            "    fi.chunk_id = coalesce(fi.chunk_id, row.cid), "
            "    fi.match_sim = CASE WHEN row.ms > coalesce(fi.match_sim, 0) "
            "                        THEN row.ms ELSE fi.match_sim END, "
            "    fi.role_confidence = CASE WHEN row.rc > coalesce(fi.role_confidence, 0) "
            "                              THEN row.rc ELSE fi.role_confidence END "
            "MERGE (fi)-[:FRAME_EVOKED_IN]->(c)",
            rows=inst_rows,
        )
        # Batch 2a: MERGE the distinct filler ENTITIES first, on their own.
        # role_rows is keyed (fid, nn, role), so ONE entity filling TWO roles of a
        # frame repeats its name_norm — and MERGEing the entity inside the same
        # UNWIND as the edge creates it twice before the ENTITY[name_norm] UNIQUE
        # index settles, raising DuplicatedKeyException and failing the caller's
        # whole document. ("Being_named" with Entity=X and Name=X is exactly this.)
        # Collapsing on the merge key first makes the edge pass a pure MATCH.
        if role_rows:
            # A role edge attaches to an entity the EXTRACTION path already
            # minted — it never creates one. This used to MERGE, and a MERGE here
            # invents a node with no type, no confidence and no MENTIONS edge to
            # any chunk: 624 of 2105 nodes on LoCoMo conv-30 (30%) existed only
            # because a frame pointed at a span. They are unreachable by any
            # entity query, they cannot be resolved or merged with anything, and
            # they reached the generator's fact-sheet as role fillers.
            #
            # A span worth a graph node is one the extractor recognised on its
            # own evidence. If it did not, the frame layer citing it is not extra
            # knowledge — it is an assertion with no provenance, so the role is
            # dropped. MATCH, never MERGE.
            await self.command(
                "UNWIND $rows AS row "
                "MATCH (fi:FRAME_INSTANCE {id: row.fid}) "
                "MATCH (e:ENTITY {name_norm: row.nn}) "
                "MERGE (fi)-[:PLAYS_ROLE {role: row.role}]->(e)",
                rows=role_rows,
            )
        return len(inst_rows)

    async def write_topics(self, topics: list[dict[str, Any]]) -> int:
        """Upsert TOPIC vertices + IN_TOPIC membership edges.

        Each ``topics`` item: ``{topic_id, summary, modularity, title?,
        keywords?: [str, ...], coherence?, entities?: [(name, pagerank), ...]}``.

        ``entities`` -> ``ENTITY-[:IN_TOPIC]->TOPIC`` (membership, the actual
        output of the clustering) — matched on ``name_norm`` since that is
        ENTITY's unique key, not ``name``.

        No ``TOPIC.embedding`` and no ``CHUNK-[:HAS_TOPIC]->TOPIC``: both were
        write-only. The topic retrieval channel that traversed HAS_TOPIC, and
        the vector search that would have used the embedding, were removed with
        the rest of the topic read path (see ``channels/registry``); the only
        reader left is ``Memory.ltm_topics``, which walks IN_TOPIC. The cluster
        centroid is still computed — it is what ``coherence`` is measured
        against — it is just not persisted.
        """
        written = 0
        for t in topics:
            tid = str(t.get("topic_id") or "").strip()
            if not tid:
                continue
            await self.command(
                "MERGE (tp:TOPIC {topic_id: $tid}) "
                "SET tp.summary = $summary, tp.modularity = $mod, "
                "    tp.title = $title, tp.keywords = $keywords, tp.coherence = $coherence",
                tid=tid,
                summary=t.get("summary", ""),
                mod=t.get("modularity"),
                title=t.get("title", ""),
                keywords=[str(k) for k in (t.get("keywords") or [])],
                coherence=float(t.get("coherence") or 0.0),
            )
            entity_rows = [
                {"nn": _name_norm(str(name)), "pr": float(pr or 0.0)}
                for name, pr in (t.get("entities") or [])
                if str(name or "").strip()
            ]
            if entity_rows:
                # Entities that don't resolve to an ENTITY node are skipped
                # (MATCH, not MERGE) rather than minting a stray vertex.
                await self.command(
                    "MATCH (tp:TOPIC {topic_id: $tid}) "
                    "UNWIND $rows AS row "
                    "MATCH (e:ENTITY {name_norm: row.nn}) "
                    "MERGE (e)-[r:IN_TOPIC]->(tp) "
                    "SET r.pagerank = row.pr",
                    tid=tid,
                    rows=entity_rows,
                )
            written += 1
        return written

    # ------------------------------------------------------------------
    # Read API — chunks
    # ------------------------------------------------------------------

    async def search_chunks_ann(
        self,
        embedding: list[float],
        top_k: int,
        session_id: str | None = None,
        state: str | None = None,
    ) -> list[dict[str, Any]]:
        """ANN search over CHUNK.embedding via vectorNeighbors.

        session/state filters are applied client-side over an over-fetched
        candidate set (vectorNeighbors has no WHERE support).
        """
        if not embedding:
            return []
        k_pre = max(top_k * 20, 200) if (session_id or state) else top_k
        sql = _vec_search_sql("CHUNK[embedding]", embedding, k_pre)
        try:
            rows = await self._c.query(self._db, sql, language="sql")
        except Exception as exc:
            logger.warning("search_chunks_ann failed: %s", exc)
            return []
        rows = self._dedup_rows(rows)
        results: list[dict[str, Any]] = []
        for r in rows:
            if session_id and r.get("session_id") != session_id:
                continue
            if state and r.get("state") != state:
                continue
            dist = float(r.get("distance") or 1.0)
            results.append(
                {
                    "chunk_id": r.get("chunk_id", ""),
                    "session_id": r.get("session_id") or "",
                    "text": r.get("text", ""),
                    "seq": r.get("seq"),
                    "kind": r.get("kind") or "",
                    "speaker": r.get("speaker") or "",
                    "ts": r.get("ts") or "",
                    "state": r.get("state") or "",
                    "memory_class": r.get("memory_class") or "",
                    "cosine_score": max(0.0, 1.0 - dist),
                }
            )
            if len(results) >= top_k:
                break
        return results

    @staticmethod
    def _chunk_filters(session_id: str | None, state: str | None) -> str:
        """Optional AND-filter tail for CHUNK queries (interpolated via _sql)."""
        parts = []
        if session_id:
            parts.append(f"c.session_id = {quote(session_id)}")
        if state:
            parts.append(f"c.state = {quote(state)}")
        return (" AND " + " AND ".join(parts)) if parts else ""

    async def query_entity_chunks(
        self,
        entity_names: list[str],
        session_id: str | None = None,
        top_k: int = 5,
        state: str | None = None,
    ) -> list[dict[str, Any]]:
        """CHUNKs linked (via MENTIONS) to any of the given entity names.

        Short tokens (< 5 chars) use exact match to avoid substring false
        positives ("war" matching "edward"); longer tokens use CONTAINS.
        """
        if not entity_names:
            return []
        safe = [sanitize(n) for n in entity_names if n]
        if not safe:
            return []
        where_entity = " OR ".join(
            f"e.name_norm = '{n.casefold()}'"
            if len(n) < 5
            else f"e.name_norm CONTAINS '{n.casefold()}'"
            for n in safe
        )
        filters = self._chunk_filters(session_id, state)
        rows = await self.query(
            f"MATCH (c:CHUNK)-[:MENTIONS]->(e:ENTITY) "
            f"WHERE ({where_entity}){filters} "
            f"RETURN DISTINCT c.chunk_id AS chunk_id, c.text AS text LIMIT $k",
            k=top_k,
        )
        return [{"chunk_id": r["chunk_id"], "text": r.get("text", "")} for r in rows]

    async def full_text_search_chunks(
        self,
        terms: list[str],
        session_id: str | None = None,
        top_k: int = 5,
        require_all: bool = False,
        state: str | None = None,
    ) -> list[dict[str, Any]]:
        """Return CHUNKs whose text contains the given terms.

        A CONTAINS scan for now; a FULL_TEXT index replaces this in Phase 4.
        ``require_all=True`` uses AND (all terms must appear); False uses OR.

        Scans ``search_text`` — ``text`` prefixed with the speaker — so a query
        naming a person reaches utterances that never name them ("I'm opening a
        dance studio" is about Gina and says so nowhere). ``text`` stays in the
        predicate as a fallback: chunks written before ``search_text`` existed
        have it unset, and a bare CONTAINS on a null would silently drop them.
        """
        if not terms:
            return []
        safe = [sanitize(t) for t in terms if t]
        if not safe:
            return []
        op = " AND " if require_all else " OR "
        text_where = op.join(
            f"(toLower(c.search_text) CONTAINS '{t.lower()}' "
            f"OR toLower(c.text) CONTAINS '{t.lower()}')"
            for t in safe
        )
        filters = self._chunk_filters(session_id, state)
        rows = await self.query(
            f"MATCH (c:CHUNK) WHERE ({text_where}){filters} "
            f"RETURN c.chunk_id AS chunk_id, c.text AS text LIMIT $k",
            k=top_k,
        )
        return [{"chunk_id": r["chunk_id"], "text": r.get("text", "")} for r in rows]

    async def query_temporal_chunks(
        self,
        iso_dates: list[str],
        session_id: str | None = None,
        top_k: int = 10,
        state: str | None = None,
    ) -> list[dict[str, Any]]:
        """CHUNKs linked to TEMPORAL nodes matching the given ISO dates.

        Exact match per date; 4-digit years also prefix-match ``YYYY-``.
        """
        dates = [sanitize(str(d).strip()) for d in iso_dates if str(d).strip()]
        if not dates:
            return []
        conds: list[str] = []
        for d in dates:
            conds.append(f"d.date = '{d}'")
            if len(d) == 4 and d.isdigit():
                conds.append(f"d.date STARTS WITH '{d}-'")
        where_iso = " OR ".join(conds)
        filters = self._chunk_filters(session_id, state)
        rows = await self.query(
            f"MATCH (c:CHUNK)-[:MENTIONS_DATE]->(d:TEMPORAL) "
            f"WHERE ({where_iso}){filters} "
            f"RETURN DISTINCT c.chunk_id AS chunk_id, c.text AS text, d.date AS date "
            f"LIMIT $k",
            k=top_k,
        )
        return [
            {"chunk_id": r["chunk_id"], "text": r.get("text", ""), "date": r.get("date", "")}
            for r in rows
        ]

    async def query_dated_chunks_by_entities(
        self,
        entity_names: list[str],
        session_id: str | None = None,
        top_k: int = 10,
        state: str | None = None,
    ) -> list[dict[str, Any]]:
        """Date-bearing CHUNKs that also mention one of ``entity_names``.

        Serves the "when did X happen" shape, which
        :func:`~graphknows.channels.temporal.query_date_candidates` cannot: such
        a question asks FOR a date and so contains none, leaving the date-keyed
        lookup with nothing to match. Here the entity is the key and the date is
        what we are looking for — chunks that mention the subject AND carry a
        TEMPORAL edge are exactly the dated evidence about it.

        Ordered by how many of the query's entities a chunk mentions, so a chunk
        naming both subjects of a question outranks one naming either.
        """
        safe = [sanitize(n) for n in entity_names if n]
        if not safe:
            return []
        # Same short-token rule as query_entity_chunks: exact match under 5
        # chars so "war" does not match "edward", CONTAINS above it.
        where_entity = " OR ".join(
            f"e.name_norm = '{n.casefold()}'"
            if len(n) < 5
            else f"e.name_norm CONTAINS '{n.casefold()}'"
            for n in safe
        )
        filters = self._chunk_filters(session_id, state)
        rows = await self.query(
            f"MATCH (c:CHUNK)-[:MENTIONS]->(e:ENTITY) "
            f"WHERE ({where_entity}){filters} "
            f"WITH c "
            f"MATCH (c)-[:MENTIONS_DATE]->(d:TEMPORAL) "
            f"RETURN DISTINCT c.chunk_id AS chunk_id, c.text AS text, d.date AS date "
            f"LIMIT $k",
            k=top_k,
        )
        return [
            {
                "chunk_id": r["chunk_id"],
                "text": r.get("text", ""),
                "date": r.get("date", ""),
            }
            for r in rows
        ]

    async def neighbor_chunks(
        self, chunk_ids: list[str], radius: int = 1
    ) -> dict[str, list[dict[str, Any]]]:
        """``{seed_chunk_id: [neighbor {chunk_id, text}, ...]}`` within ±radius seq.

        A neighbor shares the seed's source FILE and sits within ``radius`` of its
        ``seq``. Ingested long messages split into consecutive same-file chunks,
        so an answer often lives in a sibling of the chunk retrieval actually
        matched; pulling the siblings recovers it. One round trip: fetch each
        seed's (file, seq), then all same-file chunks in the seq windows.
        """
        ids = [c for c in chunk_ids if c]
        if not ids or radius < 1:
            return {}
        seeds = await self.query(
            f"MATCH (c:CHUNK)-[:PART_OF]->(f:FILE) WHERE c.chunk_id IN {quoted_list(ids)} "
            f"RETURN c.chunk_id AS chunk_id, f.file_id AS fid, c.seq AS seq"
        )
        want: dict[str, tuple[int, int]] = {}  # file -> (min_seq, max_seq) window
        seed_key: dict[str, tuple[str, int]] = {}
        for r in seeds:
            fid, seq = r.get("fid") or "", r.get("seq")
            if not fid or seq is None:
                continue
            seed_key[r["chunk_id"]] = (fid, int(seq))
            lo, hi = int(seq) - radius, int(seq) + radius
            if fid in want:
                want[fid] = (min(want[fid][0], lo), max(want[fid][1], hi))
            else:
                want[fid] = (lo, hi)
        if not want:
            return {}
        # Fetch every candidate neighbour once, keyed by (file, seq).
        clauses = " OR ".join(
            f"(f.file_id = {quote(fid)} AND c.seq >= {lo} AND c.seq <= {hi})"
            for fid, (lo, hi) in want.items()
        )
        rows = await self.query(
            f"MATCH (c:CHUNK)-[:PART_OF]->(f:FILE) WHERE {clauses} "
            f"RETURN c.chunk_id AS chunk_id, f.file_id AS fid, c.seq AS seq, c.text AS text"
        )
        by_pos: dict[tuple[str, int], dict[str, Any]] = {}
        for r in rows:
            if r.get("seq") is not None and r.get("chunk_id"):
                by_pos[(r.get("fid") or "", int(r["seq"]))] = {
                    "chunk_id": r["chunk_id"],
                    "text": r.get("text") or "",
                    "seq": int(r["seq"]),
                }
        out: dict[str, list[dict[str, Any]]] = {}
        for cid, (fid, seq) in seed_key.items():
            sibs = [
                by_pos[(fid, s)]
                for s in range(seq - radius, seq + radius + 1)
                if s != seq and (fid, s) in by_pos
            ]
            if sibs:
                out[cid] = sibs
        return out

    async def get_chunks_by_ids(self, chunk_ids: list[str]) -> dict[str, dict[str, Any]]:
        """Fetch chunks by id. Returns {chunk_id: {text, session_id, ...}}."""
        ids = [c for c in chunk_ids if c]
        if not ids:
            return {}
        rows = await self.query(
            f"MATCH (c:CHUNK) WHERE c.chunk_id IN {quoted_list(ids)} "
            f"RETURN c.chunk_id AS chunk_id, c.session_id AS session_id, c.text AS text, "
            f"c.seq AS seq, c.kind AS kind, c.speaker AS speaker, c.ts AS ts, "
            f"c.state AS state, c.memory_class AS memory_class"
        )
        result: dict[str, dict[str, Any]] = {}
        for r in rows:
            cid = r.get("chunk_id", "")
            if cid and cid not in result:
                result[cid] = {
                    "text": r.get("text", ""),
                    "session_id": r.get("session_id") or "",
                    "seq": r.get("seq"),
                    "kind": r.get("kind") or "",
                    "speaker": r.get("speaker") or "",
                    "ts": r.get("ts") or "",
                    "state": r.get("state") or "",
                    "memory_class": r.get("memory_class") or "",
                }
        return result

    # ------------------------------------------------------------------
    # Read API — topics / frames / entity graph
    # ------------------------------------------------------------------

    async def query_frames_by_embedding(
        self,
        embedding: list[float],
        top_k: int,
        session_id: str | None = None,
        state: str | None = None,
    ) -> list[dict[str, Any]]:
        """ANN over FRAME.embedding, then join FRAME-[:EVOKED_BY]->CHUNK.

        Vector-searches the frame SUMMARIES actually evoked in this namespace, then
        returns the evoked chunks tagged with the frame match score (1 - cosine
        distance). Namespace-local — replaces the global numpy frame lookup on the
        query side. session/state filters apply over an over-fetched candidate set
        (vectorNeighbors has no WHERE support).
        """
        if not embedding:
            return []
        k_pre = max(top_k * 20, 200) if (session_id or state) else top_k
        sql = _vec_search_sql("FRAME[embedding]", embedding, k_pre)
        try:
            rows = await self._c.query(self._db, sql, language="sql")
        except Exception as exc:
            logger.warning("query_frames_by_embedding failed: %s", exc)
            return []
        # Dedup by frame id + apply session filter; keep the best score per frame.
        score: dict[str, float] = {}
        for r in rows:
            fid = r.get("id")
            if not fid or fid in score:
                continue
            if session_id and r.get("session_id") != session_id:
                continue
            dist = float(r.get("distance") or 1.0)
            score[fid] = max(0.0, 1.0 - dist)
            if len(score) >= top_k:
                break
        if not score:
            return []
        where = f"n.id IN {quoted_list(list(score))}"
        if state:
            where += f" AND c.state = {quote(state)}"
        joined = await self.query(
            f"MATCH (n:FRAME)-[:EVOKED_BY]->(c:CHUNK) WHERE {where} "
            f"RETURN n.id AS id, n.frame_type AS frame_type, c.chunk_id AS chunk_id"
        )
        # A row with no id cannot be scored: `score.get(None, 0.0)` would hand
        # back a silent 0.0 and rank a real frame chunk last. Drop it instead.
        return [
            {
                "chunk_id": r["chunk_id"],
                "frame_type": r.get("frame_type"),
                "score": score.get(r["id"], 0.0),
            }
            for r in joined
            if r.get("chunk_id") and r.get("id")
        ]

    async def verifier_coverage(self) -> dict[str, dict[str, Any]]:
        """Per-tier audit: which relation tiers the verifier is able to score.

        A gate nobody counts is indistinguishable from a gate that never fires
        — the reason ``_STMIngestHandler._report`` records ontology conformance
        rather than trusting it. This is the same instrument for verification:
        a producer that bypasses the gate shows up here as a tier with edges
        and zero scored, instead of going unnoticed.

        Aggregated in Python, not in the query: ArcadeDB's Cypher subset is
        exercised here only for grouped ``count()`` (see ``_entity_degree_map``),
        and this needs a conditional count plus a mean over non-nulls. The REL
        count is thousands, so the fold is cheap and provably correct.
        """
        rows = await self.query(
            "MATCH (a:ENTITY)-[r:REL]->(b:ENTITY) "
            "RETURN r.source AS source, r.verifier_score AS verifier_score"
        )
        return verifier_coverage_rows(rows)

    # ------------------------------------------------------------------
    # Read API — structured fact context (fact-sheet for the generator)
    # ------------------------------------------------------------------

    async def resolve_query_entities(self, words: list[str], cap: int = 6) -> list[str]:
        """Resolve query tokens to ENTITY surface names for the fact-sheet.

        A token matches ``name_norm`` only on a WORD boundary — exact, or as a
        whole space-delimited word of a multi-word entity ("studio" -> "dance
        studio"). Intra-word substrings do NOT match (so "there" never resolves
        "theresa"). Soft-fails to ``[]`` — best-effort enrichment, never a hard
        dependency of retrieval.
        """
        toks = sorted({w.strip().casefold() for w in words if w and len(w.strip()) >= 3})
        if not toks:
            return []

        def _word_boundary(t: str) -> str:
            return (
                f"(e.name_norm = {quote(t)} "
                f"OR e.name_norm STARTS WITH {quote(t + ' ')} "
                f"OR e.name_norm ENDS WITH {quote(' ' + t)} "
                f"OR e.name_norm CONTAINS {quote(' ' + t + ' ')})"
            )

        conds = " OR ".join(_word_boundary(t) for t in toks)
        try:
            rows = await self.query(
                f"MATCH (e:ENTITY) WHERE {conds} RETURN DISTINCT e.name AS name LIMIT {int(cap)}"
            )
        except Exception as exc:
            logger.debug("resolve_query_entities failed (soft): %s", exc)
            return []
        return [r["name"] for r in rows if r.get("name")]

    async def entity_fact_context(self, entity_names: list[str], max_facts: int = 40) -> list[str]:
        """Compact structured fact-sheet for the given entities (for the generator).

        Two graph reads keyed on ``name_norm`` (interpolated list literal — the
        store's IN-$list workaround):
        - semantic ENTITY->ENTITY relations (the ``REL`` layer; structural edges are
          other types and never match), formatted ``"{subject} {predicate} {object}"``
          with the predicate made readable (underscores -> spaces, lower-cased);
        - FULL FRAME_INSTANCE role fills touching any seed entity (every co-role,
          not just the seed's own), formatted ``"{frame}: role1=filler1, role2=…"``.

        The two reads are complementary, not overlapping: frame facts come ONLY
        from the reified PLAYS_ROLE layer, since the flattened ``{FRAME}_{ROLE}``
        projection that used to restate them as REL edges is no longer written.

        Prioritisation: frame instances with < 2 role fills dropped; frames and
        evidenced relations ranked before bare relations; identical strings
        deduped; capped at ``max_facts``. Soft-fails to ``[]`` (never raises into
        retrieval).

        ``max_facts`` is the graph's ENTIRE bandwidth into the answer, and it
        binds hardest on exactly the questions the graph should win: a
        list/aggregation question ("what books has X read", "what events has X
        attended") needs every matching tail, and a cap of 15 — shared with the
        frame facts — silently truncates the list the question is asking for.
        Each fact is one short line, so the cost of a larger cap is tens of
        tokens against a generator context measured in thousands.
        """
        norms = sorted({_name_norm(n) for n in entity_names if n and n.strip()})
        if not norms:
            return []
        name_lit = quoted_list(norms)
        try:
            rel_rows = await self.query(
                # Both directions so object-side facts surface ("Caroline HAS Luna"
                # when the query resolves Luna). Only current (non-tombstoned) edges.
                # REL is the single semantic ENTITY->ENTITY type; the predicate is
                # its `predicate` property (the raw human-readable label).
                f"MATCH (a:ENTITY)-[r:REL]->(b:ENTITY) "
                f"WHERE (a.name_norm IN {name_lit} OR b.name_norm IN {name_lit}) "
                f"AND (r.is_current = true OR r.is_current IS NULL) "
                f"RETURN a.name AS s, r.predicate AS p, "
                f"b.name AS o, r.evidence AS ev, r.ontology_property AS op"
            )
            frame_rows = await self.query(
                f"MATCH (fi:FRAME_INSTANCE)-[:PLAYS_ROLE]->(seed:ENTITY) "
                f"WHERE seed.name_norm IN {name_lit} "
                f"WITH DISTINCT fi "
                f"MATCH (fi)-[p:PLAYS_ROLE]->(o:ENTITY) "
                # Group per instance (fi.id), not per frame TYPE — otherwise every
                # instance of a type collapses into one giant concatenated fact.
                f"RETURN fi.id AS id, fi.frame AS frame, "
                f"collect(p.role + '=' + o.name) AS roles"
            )
        except Exception as exc:
            logger.debug("entity_fact_context failed (soft): %s", exc)
            return []

        # Frame instances first (highest structural signal); drop < 2 role fills.
        frame_facts: list[str] = []
        for r in frame_rows:
            frame = str(r.get("frame") or "").strip()
            roles = [str(x).strip() for x in (r.get("roles") or []) if x and "=" in str(x)]
            if not frame or len(roles) < 2:
                continue
            frame_facts.append(f"{frame}: {', '.join(roles)}")

        # Relations: evidenced ones outrank bare ones.
        evidenced: list[str] = []
        plain: list[str] = []
        for r in rel_rows:
            s = str(r.get("s") or "").strip()
            p = str(r.get("p") or "").strip()
            o = str(r.get("o") or "").strip()
            if not s or not p or not o:
                continue
            fact = f"{s} {p.replace('_', ' ').lower()} {o}"
            # An ontology-mapped predicate names the relation in the domain's own
            # vocabulary ("went camping with" -> friend_of). Appending it rather
            # than substituting keeps the surface form the text actually used,
            # while telling the generator what KIND of relation it is.
            op = str(r.get("op") or "").strip()
            if op and op.replace("_", " ").lower() != p.replace("_", " ").lower():
                fact = f"{fact} [{op.replace('_', ' ').lower()}]"
            (evidenced if r.get("ev") else plain).append(fact)

        # Interleave: give frames at most half the cap, THEN evidenced relations,
        # then the rest. A frame-dense conversation (dozens of instances) would
        # otherwise fill the whole cap with frames and starve the relation facts —
        # including the object-side and reserved-word predicates surfaced above.
        half = max(1, max_facts // 2)
        ranked = frame_facts[:half] + evidenced + frame_facts[half:] + plain
        out: list[str] = []
        seen: set[str] = set()
        for fact in ranked:
            if fact in seen:
                continue
            seen.add(fact)
            out.append(fact)
            if len(out) >= max_facts:
                break
        return out

    # ------------------------------------------------------------------
    # Graph analytics
    # ------------------------------------------------------------------

    async def _ensure_analytical_view(self) -> None:
        """Create + rebuild the Graph OLAP view so ``CALL algo.*`` uses the CSR engine.

        Ported from the retired ``ltm_store._ensure_analytical_view``. Without a
        covering Graph Analytical View, ArcadeDB runs graph algorithms through
        the OLTP traversal path — orders of magnitude slower, enough to exceed
        the client read timeout and look hung. Requires ArcadeDB >=26.3.2. A
        full-graph view (no VERTEX/EDGE TYPES clause) covers every type the
        native algos touch. ``UPDATE MODE OFF`` marks the view stale on each
        commit, so it is rebuilt against current data on every call — cheap
        relative to a wrong/stale result, but not free, so callers running
        several algos in one pass (see ``memory.flush_memory``) should call
        this once and pass ``skip_view_ensure=True`` to the rest. Raises on
        failure so callers can decide how to react (pagerank falls back to
        numpy; node2vec/louvain soft-fail to 0).
        """
        try:
            await self._c.command(
                self._db,
                f"CREATE GRAPH ANALYTICAL VIEW {_OLAP_VIEW} UPDATE MODE OFF",
                language="sql",
            )
        except Exception as exc:
            if "already exists" not in str(exc).lower():
                raise
        await self._c.command(
            self._db, f"REBUILD GRAPH ANALYTICAL VIEW {_OLAP_VIEW}", language="sql"
        )

    async def _entity_degree_map(self) -> dict[str, int]:
        """Total (in+out) semantic-relation degree per ENTITY, keyed by ``name_norm``.

        Only ``REL`` is matched, so structural edge types are excluded by
        construction. Two directed queries avoid relying on ArcadeDB's
        undirected Cypher pattern support.
        """
        out_rows = await self.query(
            "MATCH (a:ENTITY)-[r:REL]->(b:ENTITY) RETURN a.name_norm AS nn, count(r) AS deg"
        )
        in_rows = await self.query(
            "MATCH (a:ENTITY)<-[r:REL]-(b:ENTITY) RETURN a.name_norm AS nn, count(r) AS deg"
        )
        degree: dict[str, int] = {}
        for r in out_rows + in_rows:
            nn = r.get("nn")
            if nn:
                degree[nn] = degree.get(nn, 0) + int(r.get("deg") or 0)
        return degree

    async def _bulk_set_entity_scalar(self, prop: str, rows: list[dict[str, Any]]) -> int:
        """Persist a scalar property onto many ENTITY nodes in one UNWIND command.

        ``rows``: ``[{"nn": name_norm, "v": value}, ...]``. ``prop`` is a code-owned
        column name (pagerank / community_id), interpolated safely. Replaces the
        per-entity SET loops that issued one round-trip per entity.
        """
        rows = [r for r in rows if r.get("nn") is not None]
        if not rows:
            return 0
        await self.command(
            f"UNWIND $rows AS r MATCH (e:ENTITY {{name_norm: r.nn}}) SET e.{prop} = r.v",
            rows=rows,
        )
        return len(rows)

    async def compute_and_persist_pagerank(self, skip_view_ensure: bool = False) -> int:
        """Compute PageRank over the entity-relation graph and persist ``pagerank``.

        Tries ArcadeDB's native ``CALL algo.pageRank()`` (scoped to ENTITY rows
        via ``node.name_norm IS NOT NULL`` — ``name_norm`` is ENTITY-only in
        this schema, and is the scoping filter that actually works: ArcadeDB's
        Cypher does not honour a ``node:ENTITY`` label predicate inside a
        ``CALL ... YIELD ... WHERE`` clause — it silently matches zero rows).
        Soft-fails to 0 on any error or an empty result, the same as its
        ``compute_communities``/``compute_graph_embeddings`` siblings.

        Args:
            skip_view_ensure: Skip rebuilding the analytical view (the caller
                already did so in this pass — see ``_ensure_analytical_view``).

        Returns:
            Number of ENTITY nodes updated with a pagerank score.
        """
        try:
            if not skip_view_ensure:
                await self._ensure_analytical_view()
            # ArcadeDB's algo.pageRank yields the score column as `score`, not
            # `rank` — YIELDing `rank` binds an unknown field that comes back
            # null for every row, which the null-filter below then discards,
            # silently forcing the numpy fallback. Alias `score AS rank` so the
            # rest of this method keeps reading `rank`.
            rows = await self.query(
                "CALL algo.pageRank() YIELD node, score "
                "WHERE node.name_norm IS NOT NULL "
                "RETURN node.name_norm AS nn, score AS rank"
            )
            # A row with a null rank is not a usable result — treat "every rank
            # is null" the same as "no rows" so we don't persist meaningless
            # zeros over a graph that never actually got ranked.
            rows = [r for r in rows if r.get("nn") and r.get("rank") is not None]
        except Exception as exc:
            logger.info(
                "compute_and_persist_pagerank: native algo.pageRank unavailable (%s); skipping", exc
            )
            return 0
        if not rows:
            logger.info(
                "compute_and_persist_pagerank: no usable (non-null) ranks returned; skipping"
            )
            return 0

        updated = await self._bulk_set_entity_scalar(
            "pagerank", [{"nn": r["nn"], "v": float(r.get("rank") or 0.0)} for r in rows]
        )
        logger.info(
            "compute_and_persist_pagerank: native algo.pageRank updated %d ENTITY nodes", updated
        )
        return updated

    async def compute_graph_embeddings(
        self,
        dimensions: int = 32,
        skip_view_ensure: bool = False,
        walk_length: int = 20,
        walks: int = 20,
    ) -> int:
        """Compute node2vec embeddings over the entity-relation graph -> ``ENTITY.graph_embedding``.

        New capability (no numpy equivalent — this is not a fallback-guarded
        replacement of an existing feature). Soft-fails to 0 on any error or
        an empty result; entities with fewer than 3 semantic-relation edges are
        skipped (too sparse for a meaningful walk-based embedding).

        Args:
            dimensions: node2vec embedding dimension.
            skip_view_ensure: Skip rebuilding the analytical view (see
                ``_ensure_analytical_view``).
            walk_length: Steps per random walk.
            walks: Walks started per node.

        Returns:
            Number of ENTITY nodes updated with a ``graph_embedding``.

        Note:
            These parameters shape the embedding space, which drives PPR seed
            expansion, entity similarity, and — when the topic stage clusters on
            ``graph_embedding`` — topic granularity. They do NOT affect the
            Louvain community layer, which never sees node2vec output.
        """
        try:
            if not skip_view_ensure:
                await self._ensure_analytical_view()
            # Scoped via node.name_norm IS NOT NULL, not node:ENTITY — see
            # compute_and_persist_pagerank's docstring for why the label
            # predicate doesn't work in ArcadeDB's Cypher.
            rows = await self.query(
                f"CALL algo.node2vec({{walkLength:{int(walk_length)},"
                f"walks:{int(walks)},dimensions:{int(dimensions)}}}) "
                "YIELD node, embedding "
                "WHERE node.name_norm IS NOT NULL "
                "RETURN node.name_norm AS nn, embedding"
            )
        except Exception as exc:
            logger.info("compute_graph_embeddings: native node2vec unavailable (%s); skipping", exc)
            return 0

        rows = [r for r in rows if r.get("nn") and r.get("embedding")]
        if not rows:
            logger.info("compute_graph_embeddings: no embeddings returned; skipping")
            return 0

        # A genuinely empty degree map (no entity has any semantic relation)
        # still means every entity is sparse — only a *failed* lookup disables
        # the filter, tracked separately so it isn't mistaken for "empty".
        degree_available = True
        try:
            degree = await self._entity_degree_map()
        except Exception as exc:
            logger.warning(
                "compute_graph_embeddings: degree lookup failed (%s); skipping sparsity filter",
                exc,
            )
            degree = {}
            degree_available = False

        updated = 0
        for r in rows:
            nn = r["nn"]
            if degree_available and degree.get(nn, 0) < 3:
                continue
            try:
                # Inline literal, not a $param — verified live that ArcadeDB's
                # Cypher SET double-wraps a LIST-typed param (n.prop becomes
                # [[...]] instead of [...]), same reason _set_embedding uses
                # an inline literal for CHUNK/ENTITY/TOPIC.embedding.
                emb_literal = _emb_sql([float(x) for x in r["embedding"]])
                await self.command(
                    f"MATCH (e:ENTITY {{name_norm: $nn}}) SET e.graph_embedding = {emb_literal}",
                    nn=nn,
                )
                updated += 1
            except Exception as exc:
                logger.warning("compute_graph_embeddings: entity %r failed: %s", nn, exc)
        logger.info("compute_graph_embeddings: wrote %d ENTITY graph_embedding(s)", updated)
        return updated

    async def compute_communities(self, skip_view_ensure: bool = False) -> int:
        """Compute community_id via ArcadeDB's native Louvain -> ``ENTITY.community_id``.

        Replaces the retired igraph/leidenalg community step; runs in both
        llm_free and llm_assisted modes (plan §1: ``graph_embedding``/community
        are both-modes signals now). New capability — soft-fails to 0 on any
        error or an empty result; no fallback required.

        Args:
            skip_view_ensure: Skip rebuilding the analytical view (see
                ``_ensure_analytical_view``).

        Returns:
            Number of ENTITY nodes updated with a ``community_id``.
        """
        try:
            if not skip_view_ensure:
                await self._ensure_analytical_view()
            rows = await self.query(
                "CALL algo.louvain() YIELD node, communityId "
                "WHERE node.name_norm IS NOT NULL "
                "RETURN node.name_norm AS nn, communityId AS cid"
            )
        except Exception as exc:
            logger.info("compute_communities: native louvain unavailable (%s); skipping", exc)
            return 0

        rows = [r for r in rows if r.get("nn") and r.get("cid") is not None]
        if not rows:
            logger.info("compute_communities: no rows returned; skipping")
            return 0

        updated = await self._bulk_set_entity_scalar(
            "community_id", [{"nn": r["nn"], "v": int(r["cid"])} for r in rows]
        )
        logger.info("compute_communities: wrote %d ENTITY community_id(s)", updated)
        return updated

    # ------------------------------------------------------------------
    # Entity resolution primitives (consolidation / flush path)
    # ------------------------------------------------------------------

    async def list_entities(self) -> list[dict[str, Any]]:
        """List every ENTITY with its incoming-MENTIONS count (namespace-wide).

        Returns rows of ``{name_norm, name, type, mention_count}``. Entities are
        namespace-global, so this is not session-scoped — it feeds the layered
        entity resolver, which decides drops/merges over the whole graph.
        """
        rows = await self.query(
            "MATCH (e:ENTITY) WHERE e.name_norm IS NOT NULL "
            "OPTIONAL MATCH (c:CHUNK)-[:MENTIONS]->(e) "
            "RETURN e.name_norm AS name_norm, e.name AS name, e.type AS type, "
            "count(c) AS mention_count"
        )
        return [
            {
                "name_norm": r["name_norm"],
                "name": r.get("name") or r["name_norm"],
                "type": r.get("type") or "",
                "mention_count": int(r.get("mention_count") or 0),
            }
            for r in rows
            if r.get("name_norm")
        ]

    async def distinct_speakers(self) -> list[str]:
        """Casefolded distinct CHUNK.speaker values (the known session speakers)."""
        rows = await self.query(
            "MATCH (c:CHUNK) WHERE c.speaker IS NOT NULL AND c.speaker <> '' "
            "RETURN DISTINCT c.speaker AS speaker"
        )
        return [str(r["speaker"]).casefold() for r in rows if r.get("speaker")]

    async def rename_entity(self, old_norm: str, new_name: str) -> bool:
        """Rename an ENTITY in place, keeping its edges. Returns True on success.

        Refuses when an entity already holds the target ``name_norm`` — the
        caller should ``merge_entities`` into it instead, since two nodes cannot
        share the UNIQUE key. The old surface is kept as an alias.
        """
        new_norm = _name_norm(new_name)
        if not old_norm or not new_norm or old_norm == new_norm:
            return False
        clash = await self.query(
            "MATCH (e:ENTITY {name_norm: $nn}) RETURN e.name AS n LIMIT 1", nn=new_norm
        )
        if clash:
            return False
        rows = await self.query(
            "MATCH (e:ENTITY {name_norm: $nn}) RETURN e.name AS name, e.aliases AS aliases LIMIT 1",
            nn=old_norm,
        )
        if not rows:
            return False
        old_name = rows[0].get("name") or old_norm
        aliases = list(rows[0].get("aliases") or [])
        if old_name and old_name != new_name and old_name not in aliases:
            aliases.append(old_name)
        await self.command(
            "MATCH (e:ENTITY {name_norm: $old}) "
            "SET e.name = $name, e.name_norm = $new, e.aliases = $aliases",
            old=old_norm,
            name=new_name,
            new=new_norm,
            aliases=aliases,
        )
        logger.info("rename_entity: %s -> %s", old_norm, new_norm)
        return True

    async def drop_entities(self, name_norms: list[str]) -> int:
        """DETACH DELETE the given ENTITY nodes (removes MENTIONS/RELATION edges).

        Used by the junk/hub filter. Returns the number of nodes deleted.
        """
        norms = [n for n in name_norms if n]
        if not norms:
            return 0
        rows = await self.query(
            f"MATCH (e:ENTITY) WHERE e.name_norm IN {quoted_list(norms)} RETURN count(e) AS n"
        )
        await self.command(
            f"MATCH (e:ENTITY) WHERE e.name_norm IN {quoted_list(norms)} DETACH DELETE e"
        )
        return int(rows[0].get("n") or 0) if rows else 0

    async def _repoint_relation(self, src_name: str, dst_name: str, row: dict[str, Any]) -> None:
        """Re-create one loser relation edge on the winner via write_relation.

        Reuses the audited write_relation (evidence-append dedup, max confidence,
        single-valued supersede). Each evidence chunk is replayed so the winner
        edge accumulates the loser's full provenance.

        The currency flag is carried, not defaulted, for the same reason
        ``verifier_score`` is: write_relation asserts "live as of now" by
        default, so replaying a tombstoned edge would resurrect it as current.
        """
        predicate = row.get("predicate") or ""
        if not predicate or not src_name or not dst_name:
            return
        conf = float(row.get("confidence") or 0.5)
        fact = row.get("fact") or ""
        valid_at = row.get("valid_at") or None
        source = row.get("source") or ""
        # Carried, not defaulted: the loser edge is DETACH DELETEd right after, so
        # dropping this would wipe the verifier's judgment to NULL — which means
        # "never scored" — on every edge the entity repair touches.
        vscore = row.get("verifier_score")
        # Same reason: a NULL/absent flag means current (that is how the read
        # path's `is_current = true OR is_current IS NULL` filter reads it), but
        # an explicit false is a tombstone the replay must not undo.
        current = row.get("is_current") is not False
        evidence = [e for e in (row.get("evidence") or []) if e]
        if not evidence:
            await self.write_relation(
                src_name,
                predicate,
                dst_name,
                fact=fact,
                confidence=conf,
                valid_at=valid_at,
                source=source,
                verifier_score=vscore,
                is_current=current,
                invalid_at=row.get("invalid_at"),
            )
            return
        for cid in evidence:
            await self.write_relation(
                src_name,
                predicate,
                dst_name,
                fact=fact,
                chunk_id=cid,
                confidence=conf,
                valid_at=valid_at,
                source=source,
                verifier_score=vscore,
                is_current=current,
                invalid_at=row.get("invalid_at"),
            )

    async def merge_entities(self, winner_norm: str, loser_norm: str) -> bool:
        """Merge the loser ENTITY into the winner. Idempotent and self-safe.

        - Re-points loser MENTIONS (CHUNK->loser) onto the winner (MERGE dedups;
          max confidence kept).
        - Re-points loser semantic RELATION edges (both directions) onto the
          winner, skipping self-loops, via write_relation (merges evidence /
          keeps max confidence).
        - Appends the loser's name + aliases to the winner's aliases (deduped).
        - DETACH DELETE the loser.

        Returns True when a merge happened, False when it was a no-op (loser or
        winner missing, or same node).
        """
        if not winner_norm or not loser_norm or winner_norm == loser_norm:
            return False
        loser = await self.query(
            "MATCH (e:ENTITY {name_norm: $nn}) RETURN e.name AS name, e.aliases AS aliases LIMIT 1",
            nn=loser_norm,
        )
        if not loser:
            return False  # already gone — idempotent
        winner = await self.query(
            "MATCH (e:ENTITY {name_norm: $nn}) RETURN e.name AS name, e.aliases AS aliases LIMIT 1",
            nn=winner_norm,
        )
        if not winner:
            return False  # winner must exist
        loser_name = loser[0].get("name") or loser_norm
        loser_aliases = list(loser[0].get("aliases") or [])
        winner_name = winner[0].get("name") or winner_norm
        aliases = list(winner[0].get("aliases") or [])

        # 1. Re-point MENTIONS onto the winner (old loser edges vanish on delete).
        await self.command(
            "MATCH (c:CHUNK)-[m:MENTIONS]->(:ENTITY {name_norm: $loser}) "
            "MATCH (w:ENTITY {name_norm: $winner}) "
            "MERGE (c)-[nm:MENTIONS]->(w) "
            "SET nm.confidence = CASE "
            "    WHEN coalesce(m.confidence, 0) > coalesce(nm.confidence, 0) "
            "    THEN m.confidence ELSE nm.confidence END",
            loser=loser_norm,
            winner=winner_norm,
        )

        # 2. Re-point semantic RELATION edges (both directions), skip self-loops.
        out_rows = await self.query(
            "MATCH (:ENTITY {name_norm: $loser})-[r:REL]->(b:ENTITY) "
            "RETURN b.name AS dst, b.name_norm AS dst_nn, "
            "r.predicate AS predicate, r.fact AS fact, r.evidence AS evidence, "
            "r.confidence AS confidence, r.valid_at AS valid_at, r.source AS source, "
            "r.verifier_score AS verifier_score, r.is_current AS is_current, "
            "r.invalid_at AS invalid_at",
            loser=loser_norm,
        )
        in_rows = await self.query(
            "MATCH (a:ENTITY)-[r:REL]->(:ENTITY {name_norm: $loser}) "
            "RETURN a.name AS src, a.name_norm AS src_nn, "
            "r.predicate AS predicate, r.fact AS fact, r.evidence AS evidence, "
            "r.confidence AS confidence, r.valid_at AS valid_at, r.source AS source, "
            "r.verifier_score AS verifier_score, r.is_current AS is_current, "
            "r.invalid_at AS invalid_at",
            loser=loser_norm,
        )
        # An endpoint with neither a name nor a name_norm cannot be repointed:
        # passing None through would recreate the edge against a null-named
        # node, i.e. manufacture the junk entity the merge exists to remove.
        for row in out_rows:
            dst = row.get("dst") or row.get("dst_nn")
            if not dst or row.get("dst_nn") == winner_norm:
                continue
            await self._repoint_relation(winner_name, dst, row)
        for row in in_rows:
            src = row.get("src") or row.get("src_nn")
            if not src or row.get("src_nn") == winner_norm:
                continue
            await self._repoint_relation(src, winner_name, row)

        # 3. Fold the loser's surface + aliases into the winner's aliases.
        for a in [loser_name, *loser_aliases]:
            if a and a != winner_name and a not in aliases:
                aliases.append(a)
        await self.command(
            "MATCH (w:ENTITY {name_norm: $winner}) SET w.aliases = $aliases",
            winner=winner_norm,
            aliases=aliases,
        )

        # 4. Remove the loser (and its now-orphaned original edges).
        await self.command(
            "MATCH (lo:ENTITY {name_norm: $loser}) DETACH DELETE lo", loser=loser_norm
        )
        logger.info("merge_entities: %s -> %s", loser_norm, winner_norm)
        return True

    async def consolidate_session(self, session_id: str) -> int:
        """Flip lifecycle ``state`` raw -> consolidated in place (Phase-1 shim).

        This is the thin structural consolidation for Phase 1 — NOT the full
        Phase-3 pipeline (entity merge / relation canonicalize / topics). It
        flips the session's SESSION + CHUNK nodes, and every ``raw`` ENTITY and
        relation edge (entities/relations are namespace-global, so any still-raw
        artifact is promoted). Nothing is deleted; data stays queryable.

        Returns the number of CHUNK nodes flipped for this session.
        """
        now = datetime.now(UTC).isoformat()
        # Count first (query endpoint rejects a mutating SET...RETURN as
        # non-idempotent), then flip via the command endpoint.
        chunk_rows = await self.query(
            "MATCH (c:CHUNK {session_id: $sid}) WHERE c.state = 'raw' RETURN count(c) AS n",
            sid=session_id,
        )
        await self.command(
            "MATCH (c:CHUNK {session_id: $sid}) WHERE c.state = 'raw' SET c.state = 'consolidated'",
            sid=session_id,
        )
        await self.command(
            "MATCH (s:SESSION {session_id: $sid}) "
            "SET s.state = 'consolidated', s.consolidated_at = $now",
            sid=session_id,
            now=now,
        )
        await self.command("MATCH (e:ENTITY) WHERE e.state = 'raw' SET e.state = 'consolidated'")
        await self.command(
            "MATCH (:ENTITY)-[r]->(:ENTITY) WHERE r.state = 'raw' SET r.state = 'consolidated'"
        )
        flipped = int(chunk_rows[0].get("n") or 0) if chunk_rows else 0
        logger.info("consolidate_session %s: flipped %d chunks", session_id, flipped)
        return flipped

    # ------------------------------------------------------------------
    # Deletion
    # ------------------------------------------------------------------

    async def delete_session(self, session_id: str) -> None:
        """Delete a session's FRAME/CHUNK/FILE vertices and its SESSION node.

        Namespace-global vertices (ENTITY, TOPIC, TEMPORAL) are untouched —
        they may be evidenced by other sessions.

        Raises:
            StoreError: If any label failed to delete for a reason other than the
                type not existing yet. ``purge_memory`` reports this to the
                caller instead of claiming a clean purge over orphaned nodes.
        """
        failures: list[str] = []
        for label in ("FRAME_INSTANCE", "FRAME", "CHUNK", "FILE", "SESSION"):
            try:
                await self.command(
                    f"MATCH (n:{label} {{session_id: $sid}}) DETACH DELETE n",
                    sid=session_id,
                )
            except Exception as exc:
                if is_missing_type(exc):
                    logger.debug("delete_session: %s type absent, nothing to delete", label)
                    continue
                logger.warning("delete_session: %s delete failed: %s", label, exc)
                failures.append(f"{label}: {exc}")
        if failures:
            raise StoreError(f"delete_session({session_id!r}) failed for: {'; '.join(failures)}")

    async def clear_all(self) -> None:
        """Delete every vertex (and, via DETACH, every edge) in this database."""
        for label in _VERTEX_LABELS:
            try:
                await self.command(f"MATCH (n:{label}) DETACH DELETE n")
            except Exception:
                logger.debug("clear_all: %s delete ignored", label, exc_info=True)


__all__ = ["GraphStore"]

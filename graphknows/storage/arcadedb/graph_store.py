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
import logging
from collections import defaultdict
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any

from graphknows.exceptions import SchemaVersionMismatchError, StoreError
from graphknows.storage.arcadedb._base import ArcadeStoreBase
from graphknows.storage.arcadedb._schema import _CORE_DDL, schema_version
from graphknows.storage.arcadedb._sql import is_missing_type, quote, quoted_list, sanitize
from graphknows.storage.arcadedb.client import ArcadeDBClient

logger = logging.getLogger(__name__)

_DB = "mem"

# Key of the SCHEMA_STAMP singleton; the UNIQUE index on it is the singleton.
_STAMP_ID = "singleton"

# Co-role REL edges written per frame instance. A 7-role instance is 21 ordered
# pairs, and the tail of that list is co-role trivia rather than fact.
_MAX_FRAME_REL_PAIRS = 6

# Mirrors frame_srl.MIN_CORE_ROLES. Deliberately duplicated, not imported —
# storage must not depend on ingestion. The extraction-side gate counts filled
# role SLOTS; PLAYS_ROLE only ever draws an edge for a filler that MATCHes an
# existing ENTITY, so a slot can be "filled" and still contribute zero edges.
# This is the write path's own re-check against what actually survived.
_MIN_SURVIVING_ROLE_EDGES = 2

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
#
# This is now the FALLBACK, not the whole story: _is_functional() checks the
# persisted ONTOLOGY_PROPERTY.functional flag first (owl:FunctionalProperty,
# threaded through the ontology loader) and only falls back to this hand-coded
# set for a predicate no configured ontology declares.
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

# Vertex labels wiped by clear_all() / walked by delete_session(). FE and
# SEMTYPE are deliberately absent, same as ONTOLOGY/ONTOLOGY_CLASS: they are
# reference data written once per namespace by write_frame_elements, not
# per-namespace content tied to a session's lifecycle.
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


def _evidence_set(chunk_id: str) -> str:
    """Return the ``SET`` fragment that appends *chunk_id* to ``r.evidence``.

    THE provenance contract for REL edges: every writer of a REL edge appends
    the chunk that asserted it here, deduplicated, so ``r.evidence`` is the
    complete list of chunks a reader can cite for the edge — whichever writer
    minted it. Both ``write_relation`` and ``write_frame_instances``' co-role
    projection use this; do not hand-roll a second form.

    The chunk id is interpolated as a quoted literal rather than bound, because
    openCypher cannot express ``$param IN <list-property>`` reliably. Safe
    inside an ``UNWIND`` body: the id is constant for the whole statement.

    An empty *chunk_id* still initialises the list, so the property is an empty
    list rather than NULL — but that edge is then UNSOURCED, and every caller
    that can produce one must count it (see the residue logging in both
    writers) instead of letting it pass silently.
    """
    if not chunk_id:
        return "r.evidence = coalesce(r.evidence, []), "
    cid = quote(chunk_id)
    return (
        f"r.evidence = CASE "
        f"WHEN {cid} IN coalesce(r.evidence, []) THEN r.evidence "
        f"ELSE coalesce(r.evidence, []) + [{cid}] END, "
    )


# Structural REL columns a merge repair must carry over by hand: write_relation
# coalesces them to its own tier-3/4 defaults, which is right for a fresh relex
# edge and wrong for a replay of an edge that already had a tier-1 identity.
_CARRIED_REL_COLUMNS = (
    "canonical_predicate",
    "anchor",
    "reified",
    "frame",
    "frame_instance_id",
    "head_role",
    "tail_role",
    "polarity",
    "modality",
    "epistemic",
    "hedged",
    "ontology_property",
)

# The co-role frame projection's MERGE key beyond `predicate`: REL column -> the
# UNWIND row field that fills it (write_frame_instances, batch 2b). `polarity`
# and `modality` are in it deliberately — without them a fact and its denial are
# ONE edge and the later write silently wins.
#
# It lives here, next to the repair path, because there are two REL writers with
# two different MERGE keys and the repair must rebuild an edge under the key its
# own writer used. Replaying a frame edge through write_relation's
# `{predicate}`-only key merged every edge between the pair that shared a
# predicate — collapsing role and polarity variants into one — and the survivor,
# lacking those properties, then matched no future projection MERGE, so the next
# ingest wrote a duplicate beside it.
_FRAME_REL_KEY: dict[str, str] = {
    "head_role": "row.hr",
    "tail_role": "row.tr",
    "polarity": "row.pol",
    "modality": "row.mod",
}


def _frame_merge_key(row: dict[str, Any]) -> dict[str, str] | None:
    """Rebuild the projection's MERGE key from a REL row, or None if it is not one.

    Only a reified edge that kept both role names can be re-created under the
    projection's key; anything else (a relex edge, or a frame edge written before
    roles existed) is replayed under ``{predicate}`` alone as before.
    ``polarity``/``modality`` fall back to the projection's OWN defaults, so the
    rebuilt edge is the one the next ingest MERGEs onto rather than a second edge
    beside it.
    """
    if not row.get("reified"):
        return None
    head, tail = str(row.get("head_role") or ""), str(row.get("tail_role") or "")
    if not head or not tail:
        return None
    return {
        "head_role": head,
        "tail_role": tail,
        "polarity": str(row.get("polarity") or "pos"),
        "modality": str(row.get("modality") or "none"),
    }


# The single REL projection merge_entities reads a loser edge through — BOTH
# directions. Outgoing and incoming used to carry different column lists, and
# the shorter one (incoming) dropped every structural column: measured on
# conv-30, 109 REL edges came back from an entity merge as unreified `lemma:`
# edges with no roles, no frame_instance_id and no evidence — unsourced by both
# provenance routes. One list, so the two directions cannot drift again.
_REL_ROW_SELECT = ", ".join(
    f"r.{c} AS {c}"
    for c in (
        "predicate",
        "fact",
        "evidence",
        "confidence",
        "valid_at",
        "source",
        "verifier_score",
        "ontology_blocked",
        # The tombstone pair. write_relation defaults to "live as of now", so a
        # repair that cannot READ these resurrects every dead edge it replays --
        # the failure #171 fixed on the write side, undone on the read side.
        "is_current",
        "invalid_at",
        *_CARRIED_REL_COLUMNS,
    )
)


# How much a chunk tagged with a SUBCLASS of the matched class is worth. Below
# 1.0 so an exact match always wins; well above 0 because inheritance is real
# evidence, not a tiebreak — a Dance Studio genuinely is a Facility.
_SUBCLASS_DISCOUNT = 0.6


def _derive_frame_instance(
    inst: dict[str, Any], known: set[str]
) -> tuple[str, dict[str, list[str]], str, set[tuple[str, str, str]]] | None:
    """Pull ``(frame, roles, assertion, persistable)`` out of one raw instance.

    Returns ``None`` when the instance carries no frame name — nothing to
    write. ``persistable`` is the set of ``(role, name_norm, name)`` fills
    whose filler resolves to an ENTITY in ``known`` — the only fills a
    PLAYS_ROLE edge can ever attach to, since that edge MATCHes its entity and
    never MERGEs one (see ``write_frame_instances``). ``assertion`` folds in
    polarity/modality/epistemic because how a fact is asserted is part of the
    fact's identity: "Melanie moved to Berlin" and "Melanie did NOT move to
    Berlin" must not hash to the same id.
    """
    frame = str(inst.get("frame") or "").strip()
    if not frame:
        return None
    roles: dict[str, list[str]] = inst.get("roles") or {}
    persistable = {
        (r, _name_norm(str(f)), str(f).strip())
        for r, fs in roles.items()
        for f in (fs or [])
        if str(f).strip() and _name_norm(str(f)) in known
    }
    assertion = (
        f"{inst.get('polarity') or 'pos'}/"
        f"{inst.get('modality') or 'none'}/"
        f"{inst.get('epistemic') or 'asserted'}"
    )
    return frame, roles, assertion, persistable


def _drop_subsumed_siblings(
    records: list[tuple[int, str, str, frozenset[tuple[str, str, str]]]],
) -> dict[int, int]:
    """Map each subsumed sibling's index to the index that subsumes it.

    Two instances of the same ``(frame, assertion)`` written in one call are
    one fact extracted twice, at different completeness, when one's surviving
    fillers are a PROPER SUBSET of the other's — keep only the maximal one.
    Fillers that are merely overlapping or disjoint stay separate: only a
    strict subset/superset pair collapses. Equal filler sets already hash to
    the same id and MERGE on their own, so they need no entry here.

    Deterministic tie-break when several supersets qualify (a diamond, or a
    chain): the largest surviving set, then the lowest index.
    """
    by_key: dict[tuple[str, str], list[tuple[int, frozenset[tuple[str, str, str]]]]] = defaultdict(
        list
    )
    for index, frame, assertion, persistable in records:
        by_key[(frame, assertion)].append((index, persistable))

    absorbed_by: dict[int, int] = {}
    for group in by_key.values():
        for index, fills in group:
            supersets = [
                (other_index, other_fills)
                for other_index, other_fills in group
                if other_index != index and fills and fills < other_fills
            ]
            if supersets:
                survivor, _ = min(supersets, key=lambda pair: (-len(pair[1]), pair[0]))
                absorbed_by[index] = survivor
    return absorbed_by


class GraphStore(ArcadeStoreBase):
    """Single-database ArcadeDB store for one namespace (golden layer).

    Args:
        client: Shared ``ArcadeDBClient`` (connect() is idempotent).
        db: Database name, from :func:`graphknows.storage.namespace.db_name`.
    """

    def __init__(self, client: ArcadeDBClient, db: str = _DB) -> None:
        super().__init__(client, db)
        # Frames whose per-frame data (definition embedding, BROADER parents)
        # this process has already written. One store is one database, and
        # neither changes, so re-sending a ~1KB inline vector and the taxonomy
        # on every chunk that evokes the frame is pure cost. A cold start just
        # writes them once more — both are idempotent MERGE/SET.
        self._frames_materialised: set[str] = set()
        # Cache of {label, uri} strings declared owl:FunctionalProperty in the
        # persisted ontology (see _functional_predicates). One store is one
        # database and the ontology is reference data written once, so a cold
        # cache just re-queries once more.
        self._functional_predicates_cache: set[str] | None = None

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    async def ensure_schema(self, dims: int) -> None:
        """Create the database and the golden-layer schema idempotently.

        Args:
            dims: Embedding dimension for the LSM_VECTOR indexes (must match
                the configured embedder).

        Raises:
            SchemaVersionMismatchError: The database was built by a different
                schema version, so it is refused before the caller's first
                query (contracts/schema-stamp.md).
        """
        # Asked BEFORE _apply_ddl, which creates the database: afterwards
        # "did this database already exist?" is unanswerable.
        pre_existing = await self._c.database_exists(self._db)
        await self._apply_ddl(_CORE_DDL, dims=dims)
        await self._check_schema_stamp(dims, pre_existing=pre_existing)
        # This database may have been dropped and recreated under us, so nothing
        # written by a previous life of this object can be assumed present.
        self._frames_materialised.clear()
        self._functional_predicates_cache = None
        logger.info("GraphStore schema ensured (db=%s, dims=%d)", self._db, dims)

    async def _check_schema_stamp(self, dims: int, *, pre_existing: bool) -> None:
        """Refuse this database unless the schema version it records is ours.

        A fresh database carries no stamp and gets one. A pre-existing database
        that carries none predates stamping and is not assumed compatible
        (FR-014) — the same refusal as a differing stamp, hence one comparison.
        """
        running = schema_version(dims)
        recorded = await self._read_stamp()
        if recorded is None and not pre_existing:
            recorded = await self._write_stamp(running)
        if recorded != running:
            namespace = "" if self._db == _DB else self._db.removeprefix(f"{_DB}_")
            raise SchemaVersionMismatchError(namespace, self._db, recorded, running)

    async def _read_stamp(self) -> str | None:
        """Version on the SCHEMA_STAMP singleton, or None if it is unstamped."""
        rows = await self.query(
            "MATCH (s:SCHEMA_STAMP {stamp_id: $id}) RETURN s.version AS version",
            id=_STAMP_ID,
        )
        return rows[0].get("version") if rows else None

    async def _write_stamp(self, running: str) -> str:
        """Stamp a fresh database, returning the version that ended up on it.

        A concurrent first connect may have stamped it between the read and
        this insert; the UNIQUE index on stamp_id rejects the second writer,
        who re-reads rather than failing (FR-016).
        """
        try:
            await self.command(
                "CREATE (s:SCHEMA_STAMP {stamp_id: $id, version: $v})",
                id=_STAMP_ID,
                v=running,
            )
        except Exception as exc:
            exc_s = str(exc).lower()
            if "already exists" not in exc_s and "duplicate" not in exc_s:
                raise
            return await self._read_stamp() or running
        return running

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
        kind, speaker, ts, memory_class, decode_state, embedding.
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
                    "ds": c.get("decode_state", "") or "",
                }
                for c in batch
            ]
            await self.command(
                "UNWIND $rows AS row "
                "MERGE (c:CHUNK {chunk_id: row.cid}) "
                "SET c.session_id = row.sid, c.text = row.text, c.seq = row.seq, "
                "    c.search_text = row.stext, "
                "    c.kind = row.kind, c.speaker = row.speaker, c.ts = row.ts, "
                "    c.memory_class = row.mc, c.decode_state = row.ds, "
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

    async def max_chunk_seq(self, file_id: str) -> int | None:
        """Highest ``CHUNK.seq`` written for this file; ``None`` when it has none.

        ``seq`` is a per-file position, not a per-batch counter: ``neighbor_chunks``
        expands a hit to its siblings by walking ``(file_id, seq ± radius)``. A
        second ingest into the same file therefore has to continue the sequence —
        restarting at 0 makes the new chunks read as neighbours of the old ones.
        The file is reached through PART_OF; CHUNK carries no ``file_id`` property.
        """
        rows = await self.query(
            "MATCH (c:CHUNK)-[:PART_OF]->(f:FILE {file_id: $fid}) RETURN max(c.seq) AS m",
            fid=file_id,
        )
        m = rows[0].get("m") if rows else None
        return None if m is None else int(m)

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

    async def _functional_predicates(self) -> set[str]:
        """Labels of ontology properties declared ``owl:FunctionalProperty``.

        Cached on the instance like ``_frames_materialised`` — the ontology is
        reference data written once per namespace. Soft-fails to an empty set
        (the type may not exist in an older namespace that predates it, or a
        lightweight test double may never have run ``GraphStore.__init__``)
        rather than raising: an ontology-detection failure must not block
        relation consolidation, only weaken it to the ``_SINGLE_VALUED_RELATIONS``
        fallback.
        """
        cached = getattr(self, "_functional_predicates_cache", None)
        if cached is not None:
            return cached
        predicates: set[str] = set()
        try:
            rows = await self.query(
                "MATCH (p:ONTOLOGY_PROPERTY) WHERE p.functional = true RETURN p.label AS label"
            )
            for row in rows:
                label = row.get("label")
                if label:
                    predicates.add(str(label))
        except Exception as exc:
            logger.debug("_functional_predicates: soft-failed, falling back: %s", exc)
        self._functional_predicates_cache = predicates
        return predicates

    async def _is_functional(self, canonical_predicate: str, predicate: str) -> bool:
        """Whether a predicate is single-valued: ontology-declared first, else the hand list."""
        if canonical_predicate in await self._functional_predicates():
            return True
        return predicate.upper() in _SINGLE_VALUED_RELATIONS

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
        merge_key: dict[str, str] | None = None,
    ) -> None:
        """MERGE a dynamic relation edge (type = taxonomy predicate).

        PROVENANCE CONTRACT (both REL writers, see also
        ``write_frame_instances``): every REL edge carries the chunks that
        asserted it in ``r.evidence``, appended through :func:`_evidence_set` and
        deduplicated. A reader may treat an empty ``evidence`` as "no chunk
        asserted this edge" — it is not a shorthand for "provenance lives
        somewhere else". Frame-projected edges additionally carry
        ``frame_instance_id``, whose FRAME_INSTANCE reaches the same chunks via
        ``FRAME_EVOKED_IN``; that is a second route to the SAME provenance, not
        a substitute for the first. An edge written with no ``chunk_id`` is
        unsourced by both routes and is logged as such rather than passing
        silently.

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
        ``merge_key`` widens the edge's MERGE key beyond ``predicate``, as
        ``{column: value}``. The OTHER REL writer — ``write_frame_instances``'
        co-role projection — keys on :data:`_FRAME_REL_KEY` as well as the
        predicate, and a replay of one of ITS edges (the entity-merge repair)
        must use the same key or two edges differing only in role or polarity
        collapse into one, losing exactly the fact/denial distinction the wider
        key exists to protect. Only the repair path passes it; everything else
        keeps the one-edge-per-predicate semantics described above.
        """
        predicate = predicate.strip()
        if not predicate or not src_name or not dst_name:
            return
        # Stability ladder, tiers 3-4: an ontology property IRI when relex
        # selected one, else the bare lemma marked as unanchored. Tiers 1-2
        # (frame, verb synset) are minted by write_frame_instances, and the
        # coalesce below means they win when both paths reach the same edge.
        canon = ontology_property or f"lemma:{predicate.casefold()}"
        anchor = "ontology" if ontology_property else "none"
        src_norm = _name_norm(src_name)
        dst_norm = _name_norm(dst_name)
        now = datetime.now(UTC).isoformat()

        # A replayed dead edge must not tombstone the winner's live sibling.
        if is_current and await self._is_functional(canon, predicate):
            await self._supersede_edge(predicate, src_norm, dst_norm, now)

        evidence_set = _evidence_set(chunk_id)
        if not chunk_id:
            # No chunk asserted this edge, so neither provenance route exists for
            # it: `evidence` stays empty and there is no FRAME_INSTANCE to follow.
            # The only in-tree caller that reaches here is _repoint_relation
            # replaying an edge that was already unsourced.
            logger.info(
                "write_relation: unsourced REL edge (%s -%s-> %s) — no chunk_id, "
                "so r.evidence stays empty",
                src_name,
                predicate,
                dst_name,
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
        # Inline property maps are used for MERGE deliberately (they are only
        # unreliable for MATCHING); the values stay bound parameters.
        key_sql = "".join(f", {column}: $key_{column}" for column in (merge_key or {}))
        key_params = {f"key_{column}": value for column, value in (merge_key or {}).items()}
        await self.command(
            f"MERGE (a:ENTITY {{name_norm: $src_nn}}) "
            f"SET a.name = coalesce(a.name, $src), a.state = coalesce(a.state, 'raw'), "
            f"    a.type = coalesce(a.type, $src_type) "
            f"MERGE (b:ENTITY {{name_norm: $dst_nn}}) "
            f"SET b.name = coalesce(b.name, $dst), b.state = coalesce(b.state, 'raw'), "
            f"    b.type = coalesce(b.type, $dst_type) "
            f"MERGE (a)-[r:REL {{predicate: $predicate{key_sql}}}]->(b) "
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
            f"    r.support = coalesce(r.support, 1), "
            # The STABLE identifier, which is what dedup and contradiction
            # detection key on — as against `predicate`, which stays the most
            # SPECIFIC label for display. No frame instance covers this edge
            # (those are written by write_frame_instances with an `fn:` id), so
            # the ladder falls to the ontology property, then to the lemma.
            # `coalesce` so a frame projection that already claimed this edge
            # keeps its `fn:` identifier: tier 1 outranks tier 3.
            f"    r.canonical_predicate = coalesce(r.canonical_predicate, $canon), "
            f"    r.anchor = coalesce(r.anchor, $anchor), "
            f"    r.reified = coalesce(r.reified, false), "
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
            canon=canon,
            anchor=anchor,
            now=now,
            **key_params,
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

    @staticmethod
    def _rel_where(row: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        """WHERE fragment + params pinning the exact REL edge a fetched row came from.

        Matching on (head, tail, predicate) alone hits every role variant two
        entities share under the same predicate — co-role frame projections —
        the exact failure ``_carry_rel_columns`` documents. Pinning polarity/
        modality/head_role/tail_role too reproduces each writer's own MERGE
        key, so this can only ever match the one row it was read from.
        """
        params = {
            "h": row["h"],
            "t": row["t"],
            "p": row["p"],
            "pol": row.get("pol"),
            "mod": row.get("mod"),
            "hr": row.get("hr"),
            "tr": row.get("tr"),
        }
        where = (
            "a.name_norm = $h AND b.name_norm = $t AND r.predicate = $p "
            "AND coalesce(r.polarity,'') = coalesce($pol,'') "
            "AND coalesce(r.modality,'') = coalesce($mod,'') "
            "AND coalesce(r.head_role,'') = coalesce($hr,'') "
            "AND coalesce(r.tail_role,'') = coalesce($tr,'')"
        )
        return where, params

    async def _merge_contradicts(
        self,
        *,
        head: str,
        tail: str,
        canonical_predicate: str,
        reason: str,
        current_tail: str,
        now: str,
    ) -> None:
        """MERGE a CONTRADICTS edge between the head and tail ENTITY vertices.

        Links entities, not the two REL rows: ArcadeDB edges connect vertices
        only, so the contradicting PAIR is identified by (head,
        canonical_predicate) plus ``current_tail`` — the surviving edge's tail
        name_norm — which together are exactly enough to find both REL rows
        again.
        """
        await self.command(
            "MATCH (a:ENTITY {name_norm: $head}), (b:ENTITY {name_norm: $tail}) "
            "MERGE (a)-[c:CONTRADICTS {canonical_predicate: $cp, reason: $reason}]->(b) "
            "SET c.current_tail = $current_tail, "
            "    c.created_at = coalesce(c.created_at, $now)",
            head=head,
            tail=tail,
            cp=canonical_predicate,
            reason=reason,
            current_tail=current_tail,
            now=now,
        )

    async def consolidate_relations(self) -> dict[str, int]:
        """Triple-level consolidation over REL edges (issue #140, B7 + B10).

        The write-time paths (``write_relation`` / ``write_frame_instances``)
        MERGE on the specific relex predicate label, so the same fact arriving
        under two verbs (``hasSpouse`` vs. ``isMarriedTo``) never collapses,
        and a fact that contradicts a functional-property edge never
        invalidates it. This is the flush-time pass that reconciles the whole
        namespace afterwards:

        - Exact duplicates (same dedup key, several predicate surfaces) merge
          onto the highest-confidence edge, whose ``support`` becomes the
          summed group support and whose ``evidence`` absorbs the losers'.
        - A polarity contradiction (same fact, one asserted one denied) keeps
          both edges and links them with a CONTRADICTS edge — nothing deleted.
        - A functional-property collision (a second tail for a single-valued
          predicate, ontology-declared or from the ``_SINGLE_VALUED_RELATIONS``
          fallback) tombstones every edge but the newest and links each to the
          survivor with CONTRADICTS — tombstoned, never deleted.

        ponytail: this reads every REL edge in the namespace once per flush —
        fine at conversation scale, but the ceiling to watch if edge counts
        ever make a full scan expensive; scope it to the session's chunks then.
        """
        rows = await self.query(
            "MATCH (a:ENTITY)-[r:REL]->(b:ENTITY) WHERE r.is_current = true "
            "RETURN a.name_norm AS h, a.name AS h_name, b.name_norm AS t, "
            "b.name AS t_name, r.canonical_predicate AS cp, r.predicate AS p, "
            "r.polarity AS pol, r.modality AS mod, r.valid_at AS va, "
            "r.is_current AS cur, r.confidence AS conf, r.evidence AS ev, "
            "r.support AS support, r.created_at AS created_at, "
            "r.head_role AS hr, r.tail_role AS tr"
        )
        if not rows:
            return {"deduped": 0, "contradictions": 0, "invalidated": 0}

        now = datetime.now(UTC).isoformat()

        def bucket(row: dict[str, Any]) -> str:
            # No in-tree caller sets valid_at on a relex edge today, so this is
            # uniformly "" and costs nothing — it is in the key because the
            # key is the contract for when one does.
            return str(row.get("va") or "")[:10]

        groups: dict[tuple[str, str, str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            key = (
                str(row["h"]),
                str(row["cp"]),
                str(row["t"]),
                str(row.get("pol") or "pos"),
                str(row.get("mod") or "none"),
                bucket(row),
            )
            groups[key].append(row)

        deduped = 0
        survivors: list[dict[str, Any]] = []
        for group in groups.values():
            if len(group) == 1:
                survivors.append(group[0])
                continue
            # Highest confidence wins; ties broken by created_at then predicate
            # so the pass is deterministic and idempotent.
            ranked = sorted(
                group,
                key=lambda r: (
                    -float(r.get("conf") or 0),
                    str(r.get("created_at") or ""),
                    str(r["p"]),
                ),
            )
            winner, losers = ranked[0], ranked[1:]
            total_support = sum(int(r.get("support") or 1) for r in group)
            evidence = list(winner.get("ev") or [])
            for loser in losers:
                for cid in loser.get("ev") or []:
                    if cid not in evidence:
                        evidence.append(cid)
            # The confidence-ranked winner is not necessarily the most recently
            # RESTATED surface of this fact — a low-confidence restatement can
            # still be the latest word on it. The functional-collision pass
            # below picks a survivor by created_at alone, so a merged group
            # that kept the winner's own (possibly stale) created_at would let
            # an older fact silently outrank a newer, unrelated collision on
            # the same predicate. ISO 8601 strings compare correctly here, same
            # as the `ranked` sort key above.
            latest_created_at = max((str(r.get("created_at") or "") for r in group), default="")
            where, params = self._rel_where(winner)
            await self.command(
                f"MATCH (a:ENTITY)-[r:REL]->(b:ENTITY) WHERE {where} "
                "SET r.support = $support, r.evidence = $evidence, "
                "    r.created_at = $created_at",
                support=total_support,
                evidence=evidence,
                created_at=latest_created_at,
                **params,
            )
            for loser in losers:
                where, params = self._rel_where(loser)
                await self.command(
                    f"MATCH (a:ENTITY)-[r:REL]->(b:ENTITY) WHERE {where} DELETE r",
                    **params,
                )
                deduped += 1
            merged = dict(winner)
            merged["support"] = total_support
            merged["ev"] = evidence
            merged["created_at"] = latest_created_at
            survivors.append(merged)

        # Opposite polarity: surviving rows sharing everything but polarity.
        # Both edges are kept — nothing deleted, nothing tombstoned.
        contradictions = 0
        by_fact: dict[tuple[str, str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
        for row in survivors:
            fact_key = (
                str(row["h"]),
                str(row["cp"]),
                str(row["t"]),
                str(row.get("mod") or "none"),
                bucket(row),
            )
            by_fact[fact_key].append(row)
        for (head, cp, tail, _mod, _bucket), fact_rows in by_fact.items():
            pols = {str(r.get("pol") or "pos") for r in fact_rows}
            if {"pos", "neg"} <= pols:
                await self._merge_contradicts(
                    head=head,
                    tail=tail,
                    canonical_predicate=cp,
                    reason="polarity",
                    current_tail=tail,
                    now=now,
                )
                contradictions += 1

        # Functional collision: surviving rows sharing (head, canonical_predicate)
        # but pointing at DIFFERENT tails, where the predicate is single-valued.
        # The newest stays current; every older row is tombstoned, never deleted.
        invalidated = 0
        by_head_pred: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for row in survivors:
            by_head_pred[(str(row["h"]), str(row["cp"]))].append(row)
        for (head, cp), pred_rows in by_head_pred.items():
            tails = {str(r["t"]) for r in pred_rows}
            if len(tails) < 2:
                continue
            is_functional = False
            for row in pred_rows:
                if await self._is_functional(cp, str(row["p"])):
                    is_functional = True
                    break
            if not is_functional:
                continue
            ranked = sorted(pred_rows, key=lambda r: str(r.get("created_at") or ""))
            newest, older_rows = ranked[-1], ranked[:-1]
            for older in older_rows:
                where, params = self._rel_where(older)
                await self.command(
                    f"MATCH (a:ENTITY)-[r:REL]->(b:ENTITY) WHERE {where} "
                    "SET r.invalid_at = $now, r.is_current = false",
                    now=now,
                    **params,
                )
                await self._merge_contradicts(
                    head=head,
                    tail=str(older["t"]),
                    canonical_predicate=cp,
                    reason="functional",
                    current_tail=str(newest["t"]),
                    now=now,
                )
                invalidated += 1
                contradictions += 1

        return {
            "deduped": deduped,
            "contradictions": contradictions,
            "invalidated": invalidated,
        }

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

    async def write_frames(self, frames: list[dict[str, Any]]) -> int:
        """Bulk-write canonical FRAME vertices + EVOKED_BY evidence edges to their chunks.

        Each ``frames`` item: ``{frame_type, chunk_id, confidence?, summary?,
        embedding?, parents?}``, where ``parents`` is a list of
        ``{id, summary?, embedding?}`` for the frames it generalises to.

        FRAME is CANONICAL: one vertex per frame type per namespace, MERGEd on
        the frame name. It used to be one vertex per (session, chunk, rank),
        which duplicated a frame once per chunk that evoked it — that made the
        frame taxonomy untraversable (no stable node to hang FRAME->FRAME edges
        off) and forced saturation coverage to be counted off the node.

        Everything per-chunk therefore lives on the FRAME-[:EVOKED_BY
        {confidence}]->CHUNK edge; there is no ``chunk_id`` property on FRAME,
        and no session on it either — a frame spans every session that evokes
        it, so the session filter belongs on the CHUNK side of that edge.

        Embeddings are inline literals (LIST params conflict with the
        LSM_VECTOR index) and are written once per frame per process: the
        definition embedding is a property of the frame, not of the evocation.
        """
        if not frames:
            return 0
        rows = [
            {
                "id": f["frame_type"],
                "cid": f["chunk_id"],
                "conf": float(f.get("confidence", 1.0)),
                "summary": f.get("summary", "") or "",
            }
            for f in frames
        ]
        await self.command(
            "UNWIND $rows AS r "
            "MERGE (n:FRAME {id: r.id}) "
            "SET n.frame_type = r.id, n.summary = r.summary",
            rows=rows,
        )
        await self.command(
            "UNWIND $rows AS r "
            "MATCH (n:FRAME {id: r.id}), (c:CHUNK {chunk_id: r.cid}) "
            "MERGE (n)-[ev:EVOKED_BY]->(c) "
            "SET ev.confidence = r.conf",
            rows=rows,
        )
        # Everything below is a property of the FRAME, not of this evocation, so
        # it is written once per frame per process rather than once per chunk.
        fresh = [f for f in frames if str(f["frame_type"]) not in self._frames_materialised]
        for f in fresh:
            if emb := f.get("embedding"):
                await self._set_embedding("FRAME", "id", str(f["frame_type"]), emb)

        # FrameNet's own hierarchy, as SKOS-shaped BROADER edges. A parent is
        # never itself evoked (no EVOKED_BY), but it still needs its own
        # definition/embedding so match_frames can enter the taxonomy at an
        # interior node, not just walk down to it from a leaf.
        pairs = [
            {
                "id": str(f["frame_type"]),
                "parent": str(p["id"]),
                "summary": p.get("summary", "") or "",
            }
            for f in fresh
            for p in f.get("parents") or ()
            if p and str(p["id"]) != str(f["frame_type"])
        ]
        if pairs:
            await self.command(
                "UNWIND $rows AS r MERGE (p:FRAME {id: r.parent}) "
                "SET p.frame_type = r.parent, p.summary = r.summary",
                rows=pairs,
            )
            await self.command(
                "UNWIND $rows AS r "
                "MATCH (n:FRAME {id: r.id}), (p:FRAME {id: r.parent}) "
                "MERGE (n)-[:BROADER]->(p)",
                rows=pairs,
            )
            fresh_parents = {
                str(p["id"]): p
                for f in fresh
                for p in f.get("parents") or ()
                if str(p["id"]) not in self._frames_materialised
            }
            for pid, p in fresh_parents.items():
                if emb := p.get("embedding"):
                    await self._set_embedding("FRAME", "id", pid, emb)
            self._frames_materialised.update(fresh_parents)
        self._frames_materialised.update(str(f["frame_type"]) for f in fresh)
        return len(rows)

    async def write_ontology(self, index: Any) -> int:
        """Persist an OntologyIndex as ONTOLOGY + CLASS/PROPERTY vertices (idempotent).

        Reference data, written once per namespace at schema-ensure time rather
        than on the ingest hot path. Term definition embeddings are copied from
        the already-built index matrix, never recomputed. Guarded by a count
        check so a restart does not re-MERGE hundreds of rows.

        This writes every class in the loaded ontology (1437 for the bundled
        CCO digest), whether or not any chunk in the namespace ever gets tagged
        with it — deliberately, not because materialisation is lazy elsewhere.
        A class no chunk ever matched directly is still the entry point that
        makes its grounded descendants reachable from a general query:
        `_narrower_scores` walks one hop DOWN the `BROADER` edges written here,
        at a discount, so a query that matches `Facility` reaches a chunk
        tagged only `Dance Studio` even though `Facility` itself tags nothing.
        Two honest limits on that story: the reader sits behind
        `GRAPHKNOWS_ENABLE_ONTOLOGY_CHANNEL`, default-off pending its A/B, and
        a class with no grounded descendant in a given namespace genuinely
        contributes nothing there — it is only dead weight for namespaces that
        never populate its subtree, not in general.

        Returns the number of terms present after the write.
        """
        terms = getattr(index, "terms", None)
        if not terms:
            return 0
        ont_id = getattr(index, "digest", "") or "default"
        source = getattr(index, "source", "")

        class_terms = [t for t in terms if t.kind == "class"]
        property_terms = [t for t in terms if t.kind != "class"]
        existing = await self.query(
            "MATCH (o:ONTOLOGY {id: $id}) RETURN o.term_count AS c", id=ont_id
        )
        if existing and int(existing[0].get("c") or 0) >= len(terms):
            # The terms are here, but BROADER may predate this feature — and the
            # subclass expansion reads ONLY those edges. Backfilling them is the
            # difference between an older namespace generalising and silently
            # not; re-MERGEing the classes themselves is not needed. Same story
            # for ONTOLOGY_PROPERTY: a namespace ingested before it existed has
            # the term count already but none of these rows — without the
            # backfill here, _functional_predicates finds nothing forever.
            await self._write_class_broader(class_terms)
            await self._write_properties(property_terms)
            return len(terms)

        await self.command(
            "MERGE (o:ONTOLOGY {id: $id}) SET o.source = $src, o.term_count = $n",
            id=ont_id,
            src=source,
            n=len(terms),
        )
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
        await self._write_class_broader(class_terms)
        await self._write_properties(property_terms)
        # Embeddings are per-row inline literals: LIST params conflict with the
        # LSM_VECTOR index, the same constraint CHUNK/ENTITY/FRAME honour. Only
        # ONTOLOGY_CLASS is vector-searched, so properties get none.
        for term in class_terms:
            emb = index.term_embedding(term.uri)
            if emb:
                await self._set_embedding("ONTOLOGY_CLASS", "uri", term.uri, emb)
        logger.info("Ontology persisted: %d terms from %s", len(terms), source)
        return len(terms)

    async def _write_properties(self, property_terms: list[Any]) -> int:
        """MERGE ONTOLOGY_PROPERTY vertices, carrying the ``functional`` flag.

        Reinstated (issue #140): these vertices are how ``functional``
        (owl:FunctionalProperty) reaches the store at all — storage sits below
        symbolic in the import-linter layer contract and cannot import the
        ontology loader to consult an in-memory ``OntologyTerm``. The reader is
        ``_functional_predicates``. ``getattr`` rather than ``t.functional``:
        ``index`` and its terms are duck-typed, and a term shape that predates
        the flag must default to not-functional rather than raise.
        """
        rows = [
            {
                "uri": t.uri,
                "label": t.label,
                "definition": t.definition,
                "functional": bool(getattr(t, "functional", False)),
            }
            for t in property_terms
        ]
        if not rows:
            return 0
        await self.command(
            "UNWIND $rows AS r MERGE (n:ONTOLOGY_PROPERTY {uri: r.uri}) "
            "SET n.label = r.label, n.definition = r.definition, "
            "n.functional = r.functional",
            rows=rows,
        )
        return len(rows)

    async def _write_class_broader(self, class_terms: list[Any]) -> int:
        """MERGE ONTOLOGY_CLASS-[:BROADER]->ONTOLOGY_CLASS from declared parents.

        The same hierarchy the ``parents`` string carries, reified so subsumption
        is one indexed hop rather than a full scan of every class plus a string
        split. :meth:`_narrower_scores` is the reader.

        Parents are DECLARED as labels but written as URIs: ``ONTOLOGY_CLASS``
        indexes ``label`` NOTUNIQUE, so a label-side MATCH fans the edge out
        across every class sharing that label. Resolving here — against the
        terms already in hand — keys the MERGE on the unique property. A parent
        the ontology does not itself define is dropped rather than invented.

        Idempotent, and count-guarded so the common case (already written) costs
        one count query rather than a 1400-row UNWIND on every connect.
        """
        by_label = {t.label: t.uri for t in class_terms if t.label}
        pairs = [
            {"uri": t.uri, "puri": by_label[p]}
            for t in class_terms
            for p in t.parents
            if p in by_label and by_label[p] != t.uri
        ]
        if not pairs:
            return 0
        try:
            have = await self.query(
                "MATCH (:ONTOLOGY_CLASS)-[b:BROADER]->(:ONTOLOGY_CLASS) RETURN count(b) AS c"
            )
        except Exception:  # type absent on a namespace that predates the edge
            have = []
        if have and int(have[0].get("c") or 0) >= len(pairs):
            return len(pairs)
        await self.command(
            "UNWIND $rows AS r "
            "MATCH (c:ONTOLOGY_CLASS {uri: r.uri}), (p:ONTOLOGY_CLASS {uri: r.puri}) "
            "MERGE (c)-[:BROADER]->(p)",
            rows=pairs,
        )
        logger.info("Ontology hierarchy persisted: %d BROADER edges", len(pairs))
        return len(pairs)

    async def write_frame_elements(
        self,
        fe_rows: Sequence[Any],
        semtype_rows: Sequence[Any] = (),
        *,
        version: str = "",
    ) -> int:
        """Persist FrameNet's frame-element (FE) plane (issue #143).

        Coreness, Requires/Excludes, CoreSets and semantic types. This plane
        exists to make the constraints INSPECTABLE and TRAVERSABLE,
        not to gate anything: the validity stage in ingest
        (``_STMIngestHandler._frame_instances``) reads coreness/CoreSet/
        Requires/Excludes straight from ``symbolic.framenet.fe``, never
        through the graph. Shipped-corpus counts: 1221 frames, 11428 FEs,
        109 semantic types, 105 requires, 455 excludes.

        Reference data, written once per namespace like ``write_ontology``:
        rows are duck-typed (dict or attribute access) since storage sits
        below symbolic in the import-linter layer contract and cannot import
        ``symbolic.framenet.fe`` to build them itself. Batched UNWINDs
        throughout, no per-row statements. Returns 0 and writes nothing on
        empty input.

        Guarded by IDENTITY, not presence, when ``version`` is given: the
        write is skipped only if :meth:`sources` already lists that exact
        (channel="framenet", version) pair, so a corpus upgrade — same FE
        count, different content — re-imports instead of a stale namespace
        keeping FrameNet 1.7's plane forever against a later corpus (the
        defect issue #145 names). The import counts (source hash included) are
        recorded on the SOURCE row via ``write_source``, which this method
        does not write (it knows only the FE rows, not the import
        counts). When ``version`` is empty, callers get today's presence
        guard unchanged: skip once at least as many FE vertices exist as rows
        given, so a reconnect costs one query, not a 25k-row rewrite.
        """
        if not fe_rows:
            return 0
        if version:
            have_versions = {s.get("version") for s in await self.sources("framenet")}
            if version in have_versions:
                return len(fe_rows)
        else:
            existing = await self.query("MATCH (n:FE) RETURN count(n) AS c")
            if existing and int(existing[0].get("c") or 0) >= len(fe_rows):
                return len(fe_rows)

        def _get(row: Any, *names: str, default: Any = "") -> Any:
            for name in names:
                value = row.get(name) if isinstance(row, dict) else getattr(row, name, None)
                if value is not None and value != "":
                    return value
            return default

        fes = []
        for r in fe_rows:
            frame = _get(r, "frame")
            name = _get(r, "name", "fe")
            fes.append(
                {
                    "id": _get(r, "id", default=f"{frame}:{name}"),
                    "frame": frame,
                    "name": name,
                    "coreness": _get(r, "coreness"),
                    "definition": _get(r, "definition"),
                    "semtype": _get(r, "sem_type", "semtype"),
                    "requires": _get(r, "requires"),
                    "excludes": _get(r, "excludes"),
                    "core_set": _get(r, "core_set", default=-1),
                }
            )

        # SEMTYPE vertices: everything semtype_rows names, plus every semtype
        # a FE references but semtype_rows didn't carry (metadata-less).
        semtypes = {
            _get(s, "name"): {
                "name": _get(s, "name"),
                "abbrev": _get(s, "abbrev"),
                "definition": _get(s, "definition"),
            }
            for s in semtype_rows
        }
        for row in fes:
            if row["semtype"] and row["semtype"] not in semtypes:
                semtypes[row["semtype"]] = {"name": row["semtype"], "abbrev": "", "definition": ""}
        if semtypes:
            await self.command(
                "UNWIND $rows AS r MERGE (n:SEMTYPE {name: r.name}) "
                "SET n.abbrev = r.abbrev, n.definition = r.definition",
                rows=list(semtypes.values()),
            )

        # FRAME vertices the FEs name -- MERGE only, matching write_frames'
        # own MERGE-on-parents, so an FE imported before its frame is ever
        # evoked does not clobber an existing frame's summary/embedding.
        frame_rows = [{"frame": fr} for fr in {row["frame"] for row in fes}]
        await self.command(
            "UNWIND $rows AS r MERGE (f:FRAME {id: r.frame}) SET f.frame_type = r.frame",
            rows=frame_rows,
        )

        await self.command(
            "UNWIND $rows AS r MERGE (n:FE {id: r.id}) "
            "SET n.frame = r.frame, n.name = r.name, n.coreness = r.coreness, "
            "n.definition = r.definition, n.semtype = r.semtype, n.core_set = r.core_set",
            rows=fes,
        )
        await self.command(
            "UNWIND $rows AS r MATCH (f:FRAME {id: r.frame}), (e:FE {id: r.id}) "
            "MERGE (f)-[:HAS_FE]->(e)",
            rows=fes,
        )
        semtype_edges = [row for row in fes if row["semtype"]]
        if semtype_edges:
            await self.command(
                "UNWIND $rows AS r MATCH (e:FE {id: r.id}), (s:SEMTYPE {name: r.semtype}) "
                "MERGE (e)-[:HAS_SEMTYPE]->(s)",
                rows=semtype_edges,
            )
        requires_edges = [
            {"id": row["id"], "target": f"{row['frame']}:{row['requires']}"}
            for row in fes
            if row["requires"]
        ]
        if requires_edges:
            await self.command(
                "UNWIND $rows AS r MATCH (a:FE {id: r.id}), (b:FE {id: r.target}) "
                "MERGE (a)-[:REQUIRES_FE]->(b)",
                rows=requires_edges,
            )
        excludes_edges = [
            {"id": row["id"], "target": f"{row['frame']}:{row['excludes']}"}
            for row in fes
            if row["excludes"]
        ]
        if excludes_edges:
            await self.command(
                "UNWIND $rows AS r MATCH (a:FE {id: r.id}), (b:FE {id: r.target}) "
                "MERGE (a)-[:EXCLUDES_FE]->(b)",
                rows=excludes_edges,
            )
        logger.info(
            "FE plane persisted: %d frame elements, %d semantic types", len(fes), len(semtypes)
        )
        return len(fes)

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

    # Full provenance block a grounding row must carry — asserted complete at
    # the single write choke point rather than trusted at each call site.
    _GROUNDING_FIELDS = (
        "name",
        "uri",
        "channel",
        "version",
        "matcher",
        "score",
        "rank",
        "chunk_id",
        "span",
    )

    async def write_groundings(self, rows: list[dict[str, Any]]) -> int:
        """MERGE ENTITY-[:INSTANCE_OF {provenance}]->ONTOLOGY_CLASS edges.

        The single place a grounding edge is written. Each row must carry the
        full provenance block (``name, uri, channel, version, matcher, score,
        rank, chunk_id, span``); a row missing any field raises immediately —
        a provenance gap that only shows up as a missing property in the graph
        months later is the silent failure this checks against.

        ``score`` is *that entity's* confidence in being an instance of that
        class — not the class's own match score. An entity may accumulate
        several INSTANCE_OF edges — different chunks extract different types,
        which is legitimate multi-classing. Repeated writes keep the strongest
        mention rather than the latest, so the stored score does not depend on
        chunk order.

        ``version`` is part of the MERGE pattern, not just a SET property: a
        resource upgrade (issue #145) writes new groundings under a new
        version string, and without version in the key the second write
        overwrites the first on the same (entity, class) pair — the old and
        new versions could never coexist. Keying on it means two versions of
        one grounding are two edges; :meth:`supersede_groundings` is how the
        older one gets flagged, never overwritten.

        Callers pass only the accepted candidate (``rank`` 0): nothing reads
        INSTANCE_OF at retrieval today (the ENTITY route through it was
        measured byte-identical to the direct CHUNK->MATCHES_CLASS route and
        removed — see the comment in ``search_ontology_classes`` below), so
        writing k-best sibling edges would triple a write-only layer for no
        reader.
        """
        if not rows:
            return 0
        for row in rows:
            missing = [f for f in self._GROUNDING_FIELDS if row.get(f) is None]
            if missing:
                raise ValueError(f"grounding row missing provenance field(s) {missing}: {row}")
        merged = [
            {
                "nn": _name_norm(str(r["name"])),
                "uri": r["uri"],
                "channel": r["channel"],
                "version": r["version"],
                "matcher": r["matcher"],
                "score": float(r["score"]),
                "rank": int(r["rank"]),
                "chunk_id": r["chunk_id"],
                "span": r["span"],
            }
            for r in rows
        ]
        await self.command(
            "UNWIND $rows AS r "
            "MATCH (e:ENTITY {name_norm: r.nn}), (o:ONTOLOGY_CLASS {uri: r.uri}) "
            "MERGE (e)-[i:INSTANCE_OF {version: r.version}]->(o) "
            "SET i.score = CASE "
            "    WHEN r.score > coalesce(i.score, 0) THEN r.score "
            "    ELSE i.score "
            "END, "
            "i.channel = r.channel, i.matcher = r.matcher, "
            "i.rank = r.rank, i.chunk_id = r.chunk_id, i.span = r.span",
            rows=merged,
        )
        return len(merged)

    async def supersede_groundings(self, channel: str, keep_version: str) -> int:
        """Flag every ``channel`` grounding NOT on ``keep_version`` as superseded.

        Never DELETEs: the point of keying INSTANCE_OF on version (see
        :meth:`write_groundings`) is that a rebind is a dual-write, not data
        loss — an evaluated graph keeps its old groundings, flagged, right
        alongside the new ones. Returns how many edges were flagged.
        """
        rows = await self.command(
            "MATCH (:ENTITY)-[i:INSTANCE_OF]->(:ONTOLOGY_CLASS) "
            "WHERE i.channel = $channel AND i.version <> $keep_version "
            "SET i.superseded = true "
            "RETURN count(i) AS c",
            channel=channel,
            keep_version=keep_version,
        )
        return int(rows[0].get("c") or 0) if rows else 0

    # Full provenance block a WordNet grounding row must carry -- asserted
    # complete at the single write choke point, same contract as
    # ``_GROUNDING_FIELDS``.
    _SENSE_FIELDS = (
        "name",
        "synset_id",
        "synset_name",
        "pos",
        "definition",
        "sense_id",
        "sense_key",
        "lemma",
        "chunk_id",
        "span",
        "channel",
        "version",
        "matcher",
        "score",
        "rank",
    )

    async def write_senses(self, rows: list[dict[str, Any]]) -> int:
        """MERGE one WordNet grounding: SYNSET, SENSE and the HAS_SENSE edge.

        The single choke point for a WordNet sense grounding. This MINTS the
        SYNSET and SENSE lazily -- install never walks the ~117k-synset
        corpus; a synset/sense exists in the graph only once something
        actually grounds to it (issue #142).

        Each row must carry the full provenance block (``name, synset_id,
        synset_name, pos, definition, sense_id, sense_key, lemma, chunk_id,
        span, channel, version, matcher, score, rank``); a row missing any
        field raises immediately -- a provenance gap that only surfaces
        months later as a missing property is exactly the silent failure
        this guards against, same contract as :meth:`write_groundings`.

        ``chunk_id``, ``span`` and ``synset`` sit INSIDE the ``HAS_SENSE``
        pattern braces, not only in the SET clause: that is what makes the
        same entity grounded to two different senses in two different chunks
        two edges rather than one row where the second write clobbers the
        first. Returns the number of rows written.
        """
        if not rows:
            return 0
        for row in rows:
            missing = [f for f in self._SENSE_FIELDS if row.get(f) is None]
            if missing:
                raise ValueError(f"sense row missing provenance field(s) {missing}: {row}")
        merged = [
            {
                "nn": _name_norm(str(r["name"])),
                "synset_id": r["synset_id"],
                "synset_name": r["synset_name"],
                "pos": r["pos"],
                "definition": r["definition"],
                "sense_id": r["sense_id"],
                "sense_key": r["sense_key"],
                "lemma": r["lemma"],
                "chunk_id": r["chunk_id"],
                "span": str(r["span"]),
                "channel": r["channel"],
                "version": r["version"],
                "matcher": r["matcher"],
                "score": float(r["score"]),
                "rank": int(r["rank"]),
            }
            for r in rows
        ]
        await self.command(
            "UNWIND $rows AS r MERGE (y:SYNSET {id: r.synset_id}) "
            "SET y.name = r.synset_name, y.pos = r.pos, y.definition = r.definition",
            rows=merged,
        )
        await self.command(
            "UNWIND $rows AS r MERGE (s:SENSE {id: r.sense_id}) "
            "SET s.sense_key = r.sense_key, s.lemma = r.lemma, s.synset = r.synset_id",
            rows=merged,
        )
        await self.command(
            "UNWIND $rows AS r MATCH (s:SENSE {id: r.sense_id}), (y:SYNSET {id: r.synset_id}) "
            "MERGE (s)-[:IN_SYNSET]->(y)",
            rows=merged,
        )
        await self.command(
            "UNWIND $rows AS r "
            "MATCH (e:ENTITY {name_norm: r.nn}), (s:SENSE {id: r.sense_id}) "
            "MERGE (e)-[h:HAS_SENSE {chunk_id: r.chunk_id, span: r.span, synset: r.synset_id}]->(s) "
            "SET h.channel = r.channel, h.version = r.version, h.matcher = r.matcher, "
            "h.score = r.score, h.rank = r.rank",
            rows=merged,
        )
        return len(merged)

    async def write_synset_hypernyms(self, rows: list[dict[str, Any]]) -> int:
        """MERGE a synset's shortest hypernym chain as BROADER edges.

        Rows: ``{"child", "parent", "parent_name", "parent_definition", "rel",
        "transitive"}``, as produced by :func:`graphknows.symbolic.wordnet.ancestors`.
        The parent SYNSET is MERGEd here too -- ``ancestors()`` mints only the
        one shortest path from a grounded synset to its root, so a parent may
        not have been written by :meth:`write_senses` yet.
        """
        if not rows:
            return 0
        await self.command(
            "UNWIND $rows AS r MERGE (p:SYNSET {id: r.parent}) "
            "SET p.name = r.parent_name, p.definition = r.parent_definition",
            rows=rows,
        )
        await self.command(
            "UNWIND $rows AS r "
            "MATCH (c:SYNSET {id: r.child}), (p:SYNSET {id: r.parent}) "
            "MERGE (c)-[b:BROADER {rel: r.rel}]->(p) "
            "SET b.transitive = r.transitive",
            rows=rows,
        )
        return len(rows)

    async def write_sense_relations(self, rows: list[dict[str, Any]]) -> int:
        """MERGE SENSE-[:SENSE_REL {rel}]->SENSE edges.

        Rows: ``{"src", "dst", "rel"}``. Matching ``:SENSE`` on BOTH endpoints
        is the enforcement, not a convention: a sense-level relation
        (antonym, derivationally_related_form, pertainym, participle) simply
        cannot land on a SYNSET vertex through this writer.
        """
        if not rows:
            return 0
        await self.command(
            "UNWIND $rows AS r MATCH (a:SENSE {id: r.src}), (b:SENSE {id: r.dst}) "
            "MERGE (a)-[x:SENSE_REL {rel: r.rel}]->(b)",
            rows=rows,
        )
        return len(rows)

    async def synsets_missing_parents(
        self, synset_ids: list[str] | None = None
    ) -> list[dict[str, str]]:
        """``{"id", "name"}`` of SYNSET vertices with no outgoing BROADER edge.

        What the flush-time ancestor backfill (``WordNetChannel.populate()``)
        consumes: a synset minted by :meth:`write_senses` has no hypernym
        chain yet until :meth:`write_synset_hypernyms` runs for it.

        ``synset_ids`` scopes the scan to one flushed session's synsets
        instead of the whole namespace: passing ``None`` checks every SYNSET
        vertex, which a per-session caller should not do once the namespace
        has accumulated many sessions. Passing ``[]`` short-circuits to no
        rows rather than issuing a query that would (via ``IN []``) also
        match none, since an empty scope means the session grounded nothing.
        """
        if synset_ids is not None:
            if not synset_ids:
                return []
            rows = await self.query(
                "MATCH (y:SYNSET) WHERE y.id IN $ids "
                "OPTIONAL MATCH (y)-[b:BROADER]->() "
                "WITH y, b WHERE b IS NULL RETURN y.id AS id, y.name AS name",
                ids=synset_ids,
            )
        else:
            rows = await self.query(
                "MATCH (y:SYNSET) OPTIONAL MATCH (y)-[b:BROADER]->() "
                "WITH y, b WHERE b IS NULL RETURN y.id AS id, y.name AS name"
            )
        return [{"id": str(r["id"]), "name": str(r["name"])} for r in rows]

    async def write_source(self, row: dict[str, Any]) -> str:
        """MERGE the SOURCE vertex for one (channel, version) import (issue #145).

        Keyed on ``"<channel>@<version>"`` so a channel upgrade MERGEs a new
        vertex rather than overwriting the record of the version it replaces
        — the same identity gap ``write_groundings`` closes for INSTANCE_OF.
        ``imported_at`` defaults to now when the caller does not supply one;
        missing counts default to 0 rather than raising, since a channel that
        drops nothing has no reason to compute a zero itself. Returns the id.
        """
        channel = str(row["channel"])
        version = str(row["version"])
        source_id = f"{channel}@{version}"
        await self.command(
            "MERGE (s:SOURCE {id: $id}) "
            "SET s.channel = $channel, s.version = $version, s.source_hash = $source_hash, "
            "s.imported_at = $imported_at, s.imported = $imported, "
            "s.dropped_expressions = $dropped_expressions, "
            "s.dropped_fe_mappings = $dropped_fe_mappings",
            id=source_id,
            channel=channel,
            version=version,
            source_hash=row.get("source_hash", ""),
            imported_at=row.get("imported_at") or datetime.now(UTC).isoformat(),
            imported=int(row.get("imported") or 0),
            dropped_expressions=int(row.get("dropped_expressions") or 0),
            dropped_fe_mappings=int(row.get("dropped_fe_mappings") or 0),
        )
        return source_id

    async def sources(self, channel: str = "") -> list[dict[str, Any]]:
        """Enumerate SOURCE rows, newest ``imported_at`` first.

        The reader that makes SOURCE more than a write-only ledger: it is how
        :meth:`write_frame_elements` asks "is this version already imported?"
        and how a facade reports what a namespace currently holds.
        """
        match = "MATCH (s:SOURCE)"
        if channel:
            match += f" WHERE s.channel = {quote(channel)}"
        return await self.query(f"{match} RETURN s ORDER BY s.imported_at DESC")

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
        for uri, (sub_score, sub_label) in (
            await self._narrower_scores("ONTOLOGY_CLASS", "uri", score)
        ).items():
            score[uri] = sub_score
            label[uri] = sub_label
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

    async def write_frame_instances(
        self,
        session_id: str,
        chunk_id: str,
        instances: list[dict[str, Any]],
        nesting: list[tuple[int, str, int]] | None = None,
    ) -> int:
        """MERGE role-filled FRAME_INSTANCE vertices + PLAYS_ROLE edges.

        Each ``instances`` item: ``{frame, match_sim, role_confidence,
        roles: {role_name: [filler_name, ...]}}``.

        A FRAME_INSTANCE is a FACT, not a per-chunk artifact: its id is a content
        hash of ``frame`` + the sorted role/filler signature, so the same event
        stated in several chunks MERGEs onto ONE node accumulating a
        FRAME_EVOKED_IN edge per evidence chunk (the same model RELATION.evidence
        uses) — instead of a duplicate node per chunk. Any ``id`` on the input is
        ignored.

        Identity is computed over the fillers that CAN PERSIST, which is why the
        entity lookup below happens before the hash rather than after it. A role
        edge MATCHes its entity and never mints one (see batch 2), so a filler
        the extractor never minted is dropped — and hashing it anyway made
        identity depend on something absent from the graph. Measured on conv-30:
        20 nodes carried 7 distinct signatures, 13 of them duplicates that were
        indistinguishable once written, each holding its own single evidence
        edge instead of pooling them onto one fact. Writes are batched: one UNWIND for the instances+evidence edges,
        one for all PLAYS_ROLE edges (ENTITY fillers MERGE on ``name_norm``).

        Two more checks happen here, both because this method is the only
        place that knows which fillers actually survived the entity lookup
        (issue #161):

        - *Surviving-edge gate.* The extraction-side gate
          (``frame_srl.MIN_CORE_ROLES``, re-checked in
          ``ingest._frame_instances``) counts filled role SLOTS. A slot can be
          "filled" and still draw zero PLAYS_ROLE edges once unminted fillers
          are dropped, so an instance is re-gated here on the number of
          SURVIVING edges (``_MIN_SURVIVING_ROLE_EDGES``). Measured on conv-30: 166 of 1240 instances
          (13%) were written with fewer than two surviving edges under the old
          "non-empty" check — the "13% under-connected" half of the issue.
        - *Sibling collapse.* Two instances of the same frame written in one
          call, with the same assertion, whose surviving ``(role, name_norm)``
          fills are a subset/superset pair are one fact stated at different
          completeness — the lesser is dropped and its evidence/nesting
          resolve to the surviving, more complete instance
          (:func:`_drop_subsumed_siblings`) — the "duplicates inside one
          chunk" half of the issue.

        Each instance is also linked to its canonical FRAME vertex by
        ``EVOKES``, which is what makes instance -> frame -> BROADER* ->
        ancestor -> sibling-instance traversal possible. The cost is one extra
        UNWIND round trip per chunk, over rows that already exist — not one
        call per instance.

        The reified (FRAME_INSTANCE)-[:PLAYS_ROLE {role}]->(ENTITY) form remains
        the primary representation. Alongside it, co-role fillers are also linked
        ENTITY->ENTITY as ``REL {predicate: "fn:{frame}", source: 'frame'}`` — an
        entity-adjacency projection, so a frame fact is walkable between the
        things it relates and not only through its instance node. The ``fn:``
        prefix marks the predicate as frame-vocabulary so it self-identifies
        next to relation-vocabulary predicates (``owns``, ``plans to``) that
        share the same column; the bare frame name is unchanged in
        ``REL.frame``, and ``canonical_predicate`` is unchanged.

        PROVENANCE CONTRACT (identical to ``write_relation``, deliberately): the
        projected REL edges carry ``chunk_id`` in ``r.evidence`` via the shared
        :func:`_evidence_set`, so ``REL.evidence`` means the same thing whichever
        writer minted the edge. They ALSO carry ``frame_instance_id``, reaching
        the same chunks through ``FRAME_INSTANCE-[:FRAME_EVOKED_IN]->CHUNK`` —
        a second route to the same provenance, kept because the instance pools
        evidence across chunks that the per-write append cannot see. The
        projection used to write no evidence at all: measured on conv-30 that
        left 2431 of 2696 current REL edges with an empty ``evidence`` list, and
        a consumer reading that field reasonably concluded the graph was
        unsourced. Cost of closing it, measured against ArcadeDB at the realistic
        per-call size (18 rows): no extra round trip, no extra rows, delta inside
        noise (-0.4ms median over 30 paired calls).

        An earlier flat projection was written here and REMOVED; this one is not
        that one, and each of its three recorded defects is addressed:

        - *lossy* — it linked one agent-like filler to the others and dropped the
          rest. This writes every ordered pair, with ``head_role``/``tail_role``
          on the edge, so no fill is lost and the roles survive the flattening.
          Edge DIRECTION follows role name and means nothing — co-role fillers
          are siblings of one fact; read these undirected.
        - *one edge TYPE per frame/role pair* — REL is now a single type whose
          predicate is a property, so no DDL is minted per frame.
        - *consumed the fact-sheet cap twice* — these carry ``source='frame'``
          and ``entity_fact_context`` excludes them from its REL read, because
          its frame facts already state the same content in a better format.

        Idempotent (all MERGE). Returns the number of instances processed.
        """
        if not instances:
            return 0
        # Which fillers actually exist as entities? Identity is computed over
        # these alone, so the lookup precedes the hash. One batched query.
        wanted = {
            _name_norm(str(f))
            for inst in instances
            for fs in (inst.get("roles") or {}).values()
            for f in (fs or [])
            if str(f).strip()
        }
        known: set[str] = set()
        if wanted:
            try:
                rows = await self.query(
                    f"MATCH (e:ENTITY) WHERE e.name_norm IN {quoted_list(sorted(wanted))} "
                    f"RETURN e.name_norm AS nn"
                )
                known = {str(r["nn"]) for r in rows if r.get("nn")}
            except Exception as exc:
                logger.warning("frame-instance filler lookup failed: %s", exc)
                return 0

        inst_rows: list[dict[str, Any]] = []
        role_rows: list[dict[str, Any]] = []
        rel_rows: list[dict[str, Any]] = []
        # Same-role filler pairs refused by the projection (see below).
        co_role_dropped = 0
        # Instances dropped for having fewer than _MIN_SURVIVING_ROLE_EDGES
        # surviving PLAYS_ROLE edges (issue #161, "13% under-connected").
        gated_dropped = 0
        # Position in `instances` -> the id it was written under, so a nesting
        # pair given by index survives the drops below (no fills, no frame).
        fid_at: dict[int, str] = {}

        derived = [
            (index, *parts)
            for index, inst in enumerate(instances)
            if (parts := _derive_frame_instance(inst, known)) is not None
        ]
        # Same (frame, assertion) siblings whose surviving fills are a strict
        # subset of another's are the same fact stated less completely — one
        # extraction pass finding two candidate spans for the same event
        # (issue #161, "duplicates inside one chunk").
        absorbed_by = _drop_subsumed_siblings(
            [
                (index, frame, assertion, frozenset(persistable))
                for index, frame, _, assertion, persistable in derived
            ]
        )
        collapsed = len(absorbed_by)

        for index, frame, _roles, assertion, persistable in derived:
            if index in absorbed_by:
                continue  # written under the surviving sibling's fid, below
            inst = instances[index]
            sig = [*sorted(f"{r}={nn}" for r, nn, _ in persistable), f"@{assertion}"]
            # Nothing that can be written is nothing to identify the fact BY, so
            # the instance is dropped rather than written as a bare frame label.
            # Tested on `persistable`, not `sig`: sig now always carries the
            # assertion suffix and is therefore never empty.
            #
            # Gate on EDGE COUNT, not distinct role names — two surviving
            # fills sharing one role still clear the floor.
            if len(persistable) < _MIN_SURVIVING_ROLE_EDGES:
                gated_dropped += 1
                continue
            # Content hash for a stable frame-instance id — not a security digest.
            digest = hashlib.sha1("|".join(sig).encode(), usedforsecurity=False).hexdigest()
            fid = f"{frame}:{digest[:16]}"
            fid_at[index] = fid
            role_conf = float(inst.get("role_confidence") or 0.0)
            inst_rows.append(
                {
                    "fid": fid,
                    "frame": frame,
                    "cid": chunk_id,
                    "sid": session_id,
                    "ms": float(inst.get("match_sim") or 0.0),
                    "rc": role_conf,
                    "pol": str(inst.get("polarity") or "pos"),
                    "mod": str(inst.get("modality") or "none"),
                    "epi": str(inst.get("epistemic") or "asserted"),
                    "hed": bool(inst.get("hedged") or False),
                    "voi": str(inst.get("voice") or "active"),
                    "ten": str(inst.get("tense") or ""),
                    "att": str(inst.get("attributed_to") or ""),
                    "mru": str(inst.get("modality_rules") or ""),
                    # Provenance about HOW the trigger was found (a light-verb
                    # construction's noun object vs. the frame's own LU) — not
                    # part of the fact's identity, so it plays no role in `sig`
                    # / the content hash above, and never gates or projects.
                    "lvc": bool(inst.get("lvc") or False),
                }
            )
            ordered = sorted(persistable)
            for role, nn, name in ordered:
                role_rows.append({"fid": fid, "nn": nn, "name": name, "role": role})
            # Entity adjacency for this fact: every ordered pair of fillers of
            # DIFFERENT roles. Capped — a 7-role instance would otherwise be 21
            # edges, and the tail of that list is co-role trivia, not fact.
            #
            # ``a[0] != b[0]`` is the load-bearing half. Two fillers of the SAME
            # role are co-participants, not a relation: emitting them produced
            # `(Gina) -Residence[Co_resident->Co_resident]-> (Jon)` and
            # `(banker) -Avoiding[Agent->Agent]-> (Gina)`, which read as facts and
            # assert nothing true. Measured on conv-30 before this filter: **829
            # of 2590 reified edges — 32% of the fact layer — related a role to
            # itself.** Co-participation is already expressed correctly, by both
            # fillers hanging off the same FRAME_INSTANCE via PLAYS_ROLE; the
            # projection only ever needed to add the cross-role adjacency that
            # walking entity-to-entity cannot otherwise reach.
            same_role = sum(
                1
                for i, a in enumerate(ordered)
                for b in ordered[i + 1 :]
                if a[1] != b[1] and a[0] == b[0]
            )
            if same_role:
                co_role_dropped += same_role
            pairs = [
                (a, b)
                for i, a in enumerate(ordered)
                for b in ordered[i + 1 :]
                if a[1] != b[1] and a[0] != b[0]
            ][:_MAX_FRAME_REL_PAIRS]
            for (hr, hnn, _), (tr, tnn, _) in pairs:
                rel_rows.append(
                    {
                        "h": hnn,
                        "t": tnn,
                        "p": f"fn:{frame}",
                        "f": frame,
                        "hr": hr,
                        "tr": tr,
                        # Tier 1 of the stability ladder: a frame + its two role
                        # names is the same identifier however the surface
                        # varies, which is what makes it safe to dedup on.
                        "cp": f"fn:{frame}:{hr}->{tr}",
                        "fid": fid,
                        "pol": str(inst.get("polarity") or "pos"),
                        "mod": str(inst.get("modality") or "none"),
                        "epi": str(inst.get("epistemic") or "asserted"),
                        "hed": bool(inst.get("hedged") or False),
                    }
                )
        # A subsumed sibling maps to the id it was ABSORBED into (so a nesting
        # pair naming it still resolves), never to nothing — that only happens
        # to an instance the gate above dropped outright.
        for dropped_index, survivor_index in absorbed_by.items():
            if survivor_index in fid_at:
                fid_at[dropped_index] = fid_at[survivor_index]
        # B19 / R13: PLAYS_ROLE MATCHes ENTITY by name_norm and never MERGEs —
        # correct policy (a filler the extractor never minted must not be
        # invented here), but it means a filler with no ENTITY produces no edge
        # and nothing recorded it. That is the "4% of objects reach a chunk"
        # family of silent hole. Do not change the MATCH; make the miss visible.
        unbound = sum(
            1
            for inst in instances
            for fs in (inst.get("roles") or {}).values()
            for f in (fs or [])
            if str(f).strip() and _name_norm(str(f)) not in known
        )
        if unbound or gated_dropped:
            logger.info(
                "frame role fillers unbound: %d (no ENTITY was minted for them, so they "
                "get no PLAYS_ROLE edge); %d instance(s) consequently dropped for surviving "
                "fewer than %d role edge(s), in chunk %s",
                unbound,
                gated_dropped,
                _MIN_SURVIVING_ROLE_EDGES,
                chunk_id,
            )
        if collapsed:
            logger.info(
                "frame instances collapsed: %d in chunk %s (a less completely filled "
                "sibling of the same fact — same frame and assertion, its surviving "
                "fills a subset of another instance's)",
                collapsed,
                chunk_id,
            )
        if co_role_dropped:
            logger.debug(
                "frame projection: %d same-role filler pair(s) not written as REL in chunk %s "
                "(co-participation, not a relation — they remain joined by their instance)",
                co_role_dropped,
                chunk_id,
            )
        if not inst_rows:
            return 0
        # Batch 1: FRAME_INSTANCE + FRAME_EVOKED_IN evidence edge (per chunk).
        # RETURNs its count because the leading MATCH is SILENT when the CHUNK is
        # absent: this batch then writes nothing at all while batch 2b still
        # writes REL edges carrying a frame_instance_id that was never created.
        written = await self.command(
            "UNWIND $rows AS row "
            "MATCH (c:CHUNK {chunk_id: row.cid}) "
            "MERGE (fi:FRAME_INSTANCE {id: row.fid}) "
            "SET fi.frame = row.frame, fi.session_id = row.sid, "
            "    fi.chunk_id = coalesce(fi.chunk_id, row.cid), "
            "    fi.match_sim = CASE WHEN row.ms > coalesce(fi.match_sim, 0) "
            "                        THEN row.ms ELSE fi.match_sim END, "
            "    fi.role_confidence = CASE WHEN row.rc > coalesce(fi.role_confidence, 0) "
            "                              THEN row.rc ELSE fi.role_confidence END, "
            "    fi.polarity = row.pol, fi.modality = row.mod, fi.epistemic = row.epi, "
            "    fi.hedged = row.hed, fi.voice = row.voi, fi.tense = row.ten, "
            "    fi.attributed_to = row.att, fi.modality_rules = row.mru, "
            "    fi.lvc = row.lvc "
            "MERGE (fi)-[:FRAME_EVOKED_IN]->(c) "
            "RETURN count(fi) AS n",
            rows=inst_rows,
        )
        # Only a count we actually read can say "nothing was written" — no result
        # row at all is an unknown, not a zero, and must not raise a false alarm.
        # (ArcadeDB returns [{'n': 0}] for the miss; verified against a live db.)
        if chunk_id and written and not int(written[0].get("n") or 0):
            logger.warning(
                "frame instances dropped: chunk %s has no CHUNK vertex, so %d "
                "FRAME_INSTANCE(s) were not written — the co-role REL edges below "
                "still carry their frame_instance_id and it resolves to nothing",
                chunk_id,
                len(inst_rows),
            )
        # Batch 1b: link each instance to its canonical FRAME vertex. Own
        # command, not folded into batch 1 — batch 1's leading MATCH is already
        # silent when the CHUNK is missing, and adding a second MATCH there
        # would drop the whole instance write whenever the FRAME vertex is
        # absent. Never MERGE the FRAME here: a FRAME minted without its
        # definition, embedding and BROADER parents is a stub, and an edge into
        # a stub buys nothing (that is the ingest-path FRAME writer's job).
        linked = await self.command(
            "UNWIND $rows AS row "
            "MATCH (fi:FRAME_INSTANCE {id: row.fid}) "
            "MATCH (f:FRAME {id: row.frame}) "
            "MERGE (fi)-[:EVOKES]->(f) "
            "RETURN count(*) AS n",
            rows=inst_rows,
        )
        if linked and int(linked[0].get("n") or 0) < len(inst_rows):
            logger.info(
                "frame instances under-linked: chunk %s has %d FRAME_INSTANCE(s) "
                "but only %d EVOKES edge(s) to a FRAME vertex — the missing FRAME "
                "vertices mean those instances cannot generalise through the "
                "BROADER taxonomy",
                chunk_id,
                len(inst_rows),
                int(linked[0].get("n") or 0),
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
        # Batch 2b: co-role entity adjacency (see the docstring). MATCH, never
        # MERGE, on the endpoints — same rule as the role edges: a filler the
        # extractor never minted does not get invented here.
        if rel_rows:
            # Same evidence append as write_relation (see the contract in both
            # docstrings). chunk_id is constant across rel_rows, so one
            # interpolated literal covers the whole UNWIND.
            # ponytail: forward-only, and the residue is counted at write time,
            # not surveyed. Namespaces ingested before this keep their empty
            # `evidence` until re-ingest (backfill is one Cypher pass:
            # MATCH (fi:FRAME_INSTANCE)-[:FRAME_EVOKED_IN]->(c), (a)-[r:REL]->(b)
            # WHERE r.frame_instance_id = fi.id SET r.evidence = collect(c.chunk_id)),
            # and a graph-wide "unsourced by either route" census belongs in the
            # flush return dict, which lives outside this package.
            frame_key_sql = ", ".join(
                f"{column}: {field}" for column, field in _FRAME_REL_KEY.items()
            )
            await self.command(
                "UNWIND $rows AS row "
                "MATCH (a:ENTITY {name_norm: row.h}) "
                "MATCH (b:ENTITY {name_norm: row.t}) "
                # polarity/modality are part of the MERGE key for the same reason
                # they are part of the instance's content hash: without them a
                # fact and its denial are one edge, and the later write silently
                # wins. This is the §6.6 dedup key minus the temporal bucket.
                # Built from _FRAME_REL_KEY so the entity-merge repair replays a
                # projected edge under the very key written here.
                f"MERGE (a)-[r:REL {{predicate: row.p, {frame_key_sql}}}]->(b) "
                "SET r.source = 'frame', r.is_current = true, "
                "    r.canonical_predicate = row.cp, r.frame = row.f, "
                f"    {_evidence_set(chunk_id)}"
                "    r.frame_instance_id = row.fid, r.reified = true, r.anchor = 'frame', "
                "    r.epistemic = row.epi, r.hedged = row.hed",
                rows=rel_rows,
            )
            if not chunk_id:
                # Residue: no chunk to cite, and batch 1's `MATCH (c:CHUNK ...)`
                # matched nothing either, so the FRAME_INSTANCE these edges point
                # at was never written. Unsourced by BOTH routes — count it.
                logger.warning(
                    "frame co-role REL edges with no recoverable provenance: %d "
                    "(no chunk_id, so neither r.evidence nor FRAME_EVOKED_IN)",
                    len(rel_rows),
                )
        # Batch 3: frame-to-frame nesting, the structure a flat role list cannot
        # hold ("...because my laptop died" as Explanation -> Causation -> Death).
        # Both endpoints must have been written; a pair naming a dropped instance
        # is dropped with it rather than pointing at nothing.
        nest_rows = [
            {"p": fid_at[p], "role": role, "c": fid_at[c]}
            for p, role, c in (nesting or [])
            if p in fid_at and c in fid_at and fid_at[p] != fid_at[c]
        ]
        if nest_rows:
            await self.command(
                "UNWIND $rows AS row "
                "MATCH (p:FRAME_INSTANCE {id: row.p}) "
                "MATCH (c:FRAME_INSTANCE {id: row.c}) "
                "MERGE (p)-[:FRAME_NESTS {role: row.role}]->(c)",
                rows=nest_rows,
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
    def _chunk_filters(session_id: str | None, state: str | None, alias: str = "c.") -> str:
        """Optional AND-filter tail for CHUNK queries (interpolated via _sql).

        ``alias`` is the Cypher pattern variable the tail qualifies; the SQL
        candidate query selects straight from the type and passes ``""``.
        """
        parts = []
        if session_id:
            parts.append(f"{alias}session_id = {quote(session_id)}")
        if state:
            parts.append(f"{alias}state = {quote(state)}")
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

        Served by the ``FULL_TEXT`` indexes on ``CHUNK.text`` and
        ``CHUNK.search_text``: ``CONTAINSTEXT`` is the operator ArcadeDB answers
        from one (``EXPLAIN`` reports ``FETCH FROM INDEX CHUNK[text]``), where
        the ``toLower(...) CONTAINS`` predicate this replaces got a bucket scan —
        the ``toLower()`` wrapper alone defeated any index. SQL rather than
        Cypher because ArcadeDB's openCypher dialect has no ``CONTAINSTEXT``; the
        full-text analyser folds case on both sides, so dropping ``toLower()``
        costs nothing. ``require_all=True`` uses AND (all terms must appear);
        False uses OR.

        Searches ``search_text`` — ``text`` prefixed with the speaker — so a query
        naming a person reaches utterances that never name them ("I'm opening a
        dance studio" is about Gina and says so nowhere). ``text`` stays in the
        predicate as a fallback: chunks written before ``search_text`` existed
        have it unset, and matching a null would silently drop them.
        """
        if not terms:
            return []
        safe = [sanitize(t) for t in terms if t]
        if not safe:
            return []
        op = " AND " if require_all else " OR "
        text_where = op.join(
            f"(search_text CONTAINSTEXT '{t}' OR text CONTAINSTEXT '{t}')" for t in safe
        )
        filters = self._chunk_filters(session_id, state, alias="")
        rows = await self._c.query(
            self._db,
            f"SELECT chunk_id, text FROM CHUNK WHERE ({text_where}){filters} LIMIT {int(top_k)}",  # nosec B608
            language="sql",
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
        :func:`~graphknows.temporal.query_date_candidates` cannot: such
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

    async def failed_decode_chunks(self, session_id: str) -> list[dict[str, Any]]:
        """CHUNK rows in this session whose last decode failed, for the re-drive."""
        if not session_id:
            return []
        rows = await self.query(
            "MATCH (c:CHUNK) "
            "WHERE c.session_id = $sid AND c.decode_state = 'failed' "
            "RETURN c.chunk_id AS chunk_id, c.session_id AS session_id, "
            "c.text AS text, c.speaker AS speaker, c.ts AS ts, c.seq AS seq "
            "ORDER BY c.seq",
            sid=session_id,
        )
        return [
            {
                "chunk_id": r["chunk_id"],
                "session_id": r.get("session_id") or "",
                "text": r.get("text", ""),
                "speaker": r.get("speaker") or "",
                "ts": r.get("ts") or "",
                "seq": r.get("seq"),
            }
            for r in rows
            if r.get("chunk_id")
        ]

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
        """ANN over FRAME.embedding, widen to narrower frames, then join to chunks.

        Vector-searches the frame definitions evoked in this namespace, expands
        one hop DOWN the BROADER taxonomy, and returns the evoked chunks scored
        by ``frame match x evocation strength``. Namespace-local — replaces the
        global numpy frame lookup on the query side.

        FRAME is canonical (one vertex per frame), so neither the session nor
        the chunk lives on it: both filters apply to the CHUNK on the far side
        of EVOKED_BY. The over-fetch remains because vectorNeighbors has no
        WHERE support.
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
        # Dedup by frame id; keep the best score per frame.
        score: dict[str, float] = {}
        for r in rows:
            fid = r.get("id")
            if not fid or fid in score:
                continue
            dist = float(r.get("distance") or 1.0)
            score[fid] = max(0.0, 1.0 - dist)
            if len(score) >= top_k:
                break
        if not score:
            return []
        score.update(
            {f: s for f, (s, _) in (await self._narrower_scores("FRAME", "id", score)).items()}
        )

        where = f"n.id IN {quoted_list(list(score))}"
        if session_id:
            where += f" AND c.session_id = {quote(session_id)}"
        if state:
            where += f" AND c.state = {quote(state)}"
        joined = await self.query(
            f"MATCH (n:FRAME)-[ev:EVOKED_BY]->(c:CHUNK) WHERE {where} "
            f"RETURN n.id AS id, n.frame_type AS frame_type, c.chunk_id AS chunk_id, "
            f"ev.confidence AS confidence"
        )
        # A row with no id cannot be scored: `score.get(None, 0.0)` would hand
        # back a silent 0.0 and rank a real frame chunk last. Drop it instead.
        out: list[dict[str, Any]] = []
        for r in joined:
            if not (r.get("chunk_id") and r.get("id")):
                continue
            conf = r.get("confidence")
            # Weight the query-to-frame match by how strongly THIS chunk evoked
            # the frame. Canonical frames fan out to every chunk that evoked
            # them, so without this every one of them would rank identically.
            # `or 1.0` is wrong here: a legitimate 0.0 must stay 0.0.
            weight = 1.0 if conf is None else float(conf)
            out.append(
                {
                    "chunk_id": r["chunk_id"],
                    "frame_type": r.get("frame_type"),
                    "score": score.get(r["id"], 0.0) * weight,
                }
            )
        return out

    async def _narrower_scores(
        self, vertex: str, key: str, matched: dict[str, float]
    ) -> dict[str, tuple[float, str]]:
        """One hop DOWN the BROADER taxonomy from the matched nodes, at a discount.

        A query matching ``Rewards_and_punishments`` should reach a chunk that
        only ever evoked ``Revenge``, and one matching ``Facility`` should reach
        a chunk tagged ``Dance Studio``: the chunk is about the general thing,
        it just named the specific one. That generalisation is the whole reason
        to persist a hierarchy — and it is the reader that keeps the BROADER
        edges from being another write-only layer.

        Shared by both taxonomies because both are read identically; only the
        vertex type and its key property differ (``FRAME.id`` /
        ``ONTOLOGY_CLASS.uri``). One level, because each extra hop costs a round
        trip for rapidly thinning evidence, and scored at ``_SUBCLASS_DISCOUNT``
        of the parent's match so a direct hit always outranks an inherited one.

        Returns ``{key: (score, label)}`` — the label rides along because the
        caller has no other way to name a node it never matched itself.
        """
        if not matched:
            return {}
        try:
            rows = await self.query(
                f"MATCH (child:{vertex})-[:BROADER]->(parent:{vertex}) "  # nosec B608
                f"WHERE parent.{key} IN {quoted_list(list(matched))} "
                f"RETURN child.{key} AS child, child.label AS label, parent.{key} AS parent"
            )
        except Exception as exc:
            logger.warning("%s narrower expansion failed: %s", vertex, exc)
            return {}
        out: dict[str, tuple[float, str]] = {}
        for r in rows:
            child = str(r.get("child") or "")
            # An already-matched node keeps its own, undiscounted score.
            if not child or child in matched:
                continue
            inherited = matched.get(str(r.get("parent") or ""), 0.0) * _SUBCLASS_DISCOUNT
            if inherited > out.get(child, (0.0, ""))[0]:
                out[child] = (inherited, str(r.get("label") or ""))
        if out:
            logger.debug("%s query expanded to %d narrower nodes", vertex, len(out))
        return out

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

    async def entity_coverage(self) -> dict[str, int]:
        """Whole-namespace audit: how many ENTITY nodes carry no semantic edge.

        Reproduces the issue #160 measurement — 70 of 334 conv-30 entities
        (21%) had no ``REL`` in either direction and no incoming
        ``PLAYS_ROLE`` from a ``FRAME_INSTANCE``: written, embedded, and
        connected to nothing but their chunk. That isolated count, not a
        vibe about span length, is the metric the entity-minting gate
        (``is_graph_entity_name``) is tuned against — this is how it gets
        re-measured after a change to that gate.

        Returns ``{"total", "connected", "isolated"}``.

        Collected as three directed queries (out-REL, in-REL, incoming
        PLAYS_ROLE) folded in Python — the ``verifier_coverage`` idiom — rather
        than one NOT-EXISTS pattern, for the same reason ``_entity_degree_map``
        splits REL by direction: ArcadeDB's Cypher subset here is not relied on
        for undirected patterns or negation, only for a plain grouped scan.
        """
        total_rows = await self.query(
            "MATCH (e:ENTITY) WHERE e.name_norm IS NOT NULL RETURN e.name_norm AS nn"
        )
        total = {r["nn"] for r in total_rows if r.get("nn")}
        out_rel = await self.query(
            "MATCH (a:ENTITY)-[:REL]->(b:ENTITY) RETURN DISTINCT a.name_norm AS nn"
        )
        in_rel = await self.query(
            "MATCH (a:ENTITY)<-[:REL]-(b:ENTITY) RETURN DISTINCT a.name_norm AS nn"
        )
        roles = await self.query(
            "MATCH (e:ENTITY)<-[:PLAYS_ROLE]-(fi:FRAME_INSTANCE) RETURN DISTINCT e.name_norm AS nn"
        )
        connected = {r["nn"] for r in out_rel + in_rel + roles if r.get("nn")} & total
        return {
            "total": len(total),
            "connected": len(connected),
            "isolated": len(total) - len(connected),
        }

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

        Matches are RANKED before the cap, because the cap binds: measured on
        conv-30, all 81 questions hit it. The cut used to be a bare ``LIMIT``
        with no ``ORDER BY``, so which six entities seeded the fact sheet was
        storage order — undefined, and unrelated to the question. Only 80 of
        452 entities were ever reachable, and 16 questions received a
        byte-identical seed set. Ranking is exact-token first, then shortest
        name, then alphabetical: an exact hit beats a span that merely contains
        the token, and the shorter of two matches is the more canonical one
        ("Gina" over "Gina's hard work and effort, Gina"). Alphabetical last so
        the result is deterministic rather than merely better.
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
            # No LIMIT here: the cap is applied after ranking, and the match set
            # is bounded by the namespace's entity count.
            rows = await self.query(
                f"MATCH (e:ENTITY) WHERE {conds} RETURN DISTINCT e.name AS name, e.name_norm AS nn"
            )
        except Exception as exc:
            logger.debug("resolve_query_entities failed (soft): %s", exc)
            return []
        tokset = set(toks)
        ranked = sorted(
            ({"name": r["name"], "nn": r.get("nn") or ""} for r in rows if r.get("name")),
            key=lambda r: (r["nn"] not in tokset, len(r["nn"]), r["nn"]),
        )
        return [r["name"] for r in ranked[: int(cap)]]

    async def query_chunks_by_frame_facts(
        self,
        frames: list[str],
        entity_names: list[str],
        top_k: int,
        session_id: str | None = None,
        state: str | None = None,
    ) -> list[dict[str, Any]]:
        """Chunks reached through the FRAME_INSTANCE layer, by frame and by role filler.

        The fact layer's one route back to a CHUNK. ``REL`` has none — which is
        why every improvement to the relation layer measured flat on retrieval —
        but a FRAME_INSTANCE carries ``FRAME_EVOKED_IN`` to the chunk it was read
        from, so a query that names a frame ("what did she give?") or an entity
        can land on the chunk stating that fact.

        Two reads rather than one OR: matching either the instance's frame or its
        role fillers in a single pattern makes ArcadeDB scan FRAME_INSTANCE whole.
        Scored by how many signals a chunk's instances matched, frames and
        fillers counting alike; the caller turns rank into an RRF contribution.
        """
        scope = ""
        if session_id:
            scope += f" AND c.session_id = {quote(session_id)}"
        if state:
            scope += f" AND c.state = {quote(state)}"

        score: dict[str, float] = defaultdict(float)
        if frames:
            rows = await self.query(
                f"MATCH (fi:FRAME_INSTANCE)-[:FRAME_EVOKED_IN]->(c:CHUNK) "
                f"WHERE fi.frame IN {quoted_list(sorted(set(frames)))}{scope} "
                f"RETURN c.chunk_id AS chunk_id, count(fi) AS n"
            )
            for r in rows:
                if cid := r.get("chunk_id"):
                    score[str(cid)] += float(r.get("n") or 0)
        # Names in, ``name_norm`` matched — the same contract as
        # entity_fact_context, whose resolved names this is called with.
        norms = sorted({_name_norm(n) for n in entity_names if n and n.strip()})
        if norms:
            rows = await self.query(
                f"MATCH (e:ENTITY)<-[:PLAYS_ROLE]-(fi:FRAME_INSTANCE)-[:FRAME_EVOKED_IN]->(c:CHUNK) "
                f"WHERE e.name_norm IN {quoted_list(norms)}{scope} "
                f"RETURN c.chunk_id AS chunk_id, count(fi) AS n"
            )
            for r in rows:
                if cid := r.get("chunk_id"):
                    score[str(cid)] += float(r.get("n") or 0)

        ranked = sorted(score.items(), key=lambda kv: (-kv[1], kv[0]))[: int(top_k)]
        return [{"chunk_id": cid, "score": s} for cid, s in ranked]

    async def frame_rank_signals(
        self,
        chunk_ids: list[str],
        frames: list[str],
        entity_names: list[str],
        session_id: str | None = None,
        state: str | None = None,
    ) -> tuple[list[dict[str, Any]], dict[str, int], int]:
        """Inputs for frame re-ranking: pool matches, frame document-frequency, corpus size.

        Returns ``(rows, df, total_chunks)`` where each row is
        ``{chunk_id, frame, anchor_hits}`` — ``anchor_hits`` counting how many of
        the query's own entities fill a role in that instance, which is the
        conjunction ("Jon fills a role in a Giving here") that separates the gold
        chunk from every chunk merely naming Jon.

        Scoped to the pool: re-ranking never needs a chunk the fusion did not
        already find, so this stays a bounded lookup rather than a layer scan.
        Soft-fails to empty — a re-rank signal must never take retrieval down.
        """
        if not chunk_ids or not frames:
            return [], {}, 0
        pool_lit = quoted_list(sorted({c for c in chunk_ids if c}))
        frame_lit = quoted_list(sorted(set(frames)))
        conds: list[str] = []
        if session_id:
            conds.append(f"c.session_id = {quote(session_id)}")
        if state:
            conds.append(f"c.state = {quote(state)}")
        scope = "".join(f" AND {c}" for c in conds)
        chunk_where = " AND ".join(conds) if conds else "1 = 1"
        try:
            matches = await self.query(
                f"MATCH (fi:FRAME_INSTANCE)-[:FRAME_EVOKED_IN]->(c:CHUNK) "
                f"WHERE c.chunk_id IN {pool_lit} AND fi.frame IN {frame_lit} "
                f"RETURN c.chunk_id AS chunk_id, fi.frame AS frame, fi.id AS fid"
            )
            # Nothing in the pool carries these frames — most importantly, a
            # namespace ingested without the frame layer at all. Stop before the
            # three follow-up reads rather than paying them to score nothing.
            if not matches:
                return [], {}, 0
            # Document frequency over the SESSION, not the pool: a frame's
            # specificity is a property of the corpus, and measuring it on the
            # pool would call every frame the query matched rare by construction.
            df_rows = await self.query(
                f"MATCH (fi:FRAME_INSTANCE)-[:FRAME_EVOKED_IN]->(c:CHUNK) "
                f"WHERE fi.frame IN {frame_lit}{scope} "
                f"RETURN fi.frame AS frame, count(DISTINCT c.chunk_id) AS df"
            )
            bound: set[str] = set()
            norms = sorted({_name_norm(n) for n in entity_names if n and n.strip()})
            if norms:
                rows = await self.query(
                    f"MATCH (fi:FRAME_INSTANCE)-[:PLAYS_ROLE]->(e:ENTITY) "
                    f"WHERE e.name_norm IN {quoted_list(norms)} "
                    f"RETURN DISTINCT fi.id AS fid"
                )
                bound = {str(r["fid"]) for r in rows if r.get("fid")}
            total_rows = await self.query(
                f"MATCH (c:CHUNK) WHERE {chunk_where} RETURN count(c) AS n"
            )
        except Exception as exc:
            logger.debug("frame rank signals failed (soft): %s", exc)
            return [], {}, 0

        out = [
            {
                "chunk_id": r.get("chunk_id"),
                "frame": r.get("frame"),
                "anchor_hits": 1 if str(r.get("fid")) in bound else 0,
            }
            for r in matches
            if r.get("chunk_id") and r.get("frame")
        ]
        df = {str(r["frame"]): int(r.get("df") or 0) for r in df_rows if r.get("frame")}
        total = int(total_rows[0]["n"]) if total_rows else 0
        return out, df, total

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
                # Frame-derived adjacency is excluded: the frame facts below
                # already state it, in a form that keeps the role names. Letting
                # both in spent the cap twice on one fact — the defect that got
                # the previous flat projection removed.
                f"AND (r.source IS NULL OR r.source <> 'frame') "
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

        # One hop of frame nesting off those same instances. This is what a "why"
        # question needs: the seed's frame states WHAT happened, the frame
        # embedded in its Explanation/Cause role states why, and a flat role list
        # can only hand the generator the raw span.
        #
        # Its OWN try: FRAME_NESTS is written by the SRL path only, so a namespace
        # ingested before it — or without it — has no such edge type, and inside
        # the read above that miss would take the whole fact-sheet down with it.
        nest_rows: list[dict[str, Any]] = []
        try:
            nest_rows = await self.query(
                f"MATCH (seed:ENTITY)<-[:PLAYS_ROLE]-(fi:FRAME_INSTANCE)"
                f"-[n:FRAME_NESTS]->(inner:FRAME_INSTANCE) "
                f"WHERE seed.name_norm IN {name_lit} "
                f"RETURN DISTINCT fi.frame AS outer, n.role AS role, inner.frame AS inner_frame"
            )
        except Exception as exc:
            logger.debug("frame-nesting fact read skipped (soft): %s", exc)

        # Frame instances first (highest structural signal); drop < 2 role fills.
        frame_facts: list[str] = []
        for r in frame_rows:
            frame = str(r.get("frame") or "").strip()
            roles = [str(x).strip() for x in (r.get("roles") or []) if x and "=" in str(x)]
            if not frame or len(roles) < 2:
                continue
            frame_facts.append(f"{frame}: {', '.join(roles)}")

        # Nesting reads as one line per embedded frame, e.g.
        # "Commerce_buy -Explanation-> Causation". Cheap (one short line) and it
        # is the only place the generator learns that two facts are linked rather
        # than merely co-occurring.
        nest_facts = [
            f"{o} -{role}-> {inner}"
            for r in nest_rows
            if (o := str(r.get("outer") or "").strip())
            and (role := str(r.get("role") or "").strip())
            and (inner := str(r.get("inner_frame") or "").strip())
        ]

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
        ranked = frame_facts[:half] + nest_facts + evidenced + frame_facts[half:] + plain
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

    async def _carry_rel_columns(
        self,
        src_name: str,
        predicate: str,
        dst_name: str,
        row: dict[str, Any],
        merge_key: dict[str, str] | None = None,
    ) -> None:
        """Re-apply the loser edge's structural columns to the repointed winner edge.

        ``write_relation`` cannot express these — it coalesces
        ``canonical_predicate``/``anchor``/``reified`` to its own tier-3/4
        defaults, which is right for a fresh relex edge and wrong for a replay of
        an edge that already had a tier-1 identity. Applied as a follow-up SET so
        the repair preserves what the loser knew instead of recomputing it from a
        predicate string.

        ``ontology_blocked`` is OR-ed rather than assigned: it is sticky by
        design, so a refusal recorded on either edge survives the merge.

        ``evidence`` is deliberately NOT carried here: it is a list that
        accumulates, and a blanket SET would overwrite the winner's provenance
        with the loser's. ``_repoint_relation`` replays it chunk by chunk through
        ``write_relation``'s dedup-append instead.

        ``merge_key`` is the SAME key the replay MERGEd on. It is added to the
        WHERE so this SET lands on exactly the one edge that was just written:
        predicate alone matches EVERY edge between the pair sharing it, and
        between two frame fillers that is every role and polarity variant of the
        frame — each one overwritten with the last loser's columns.
        """
        carried = {k: row.get(k) for k in _CARRIED_REL_COLUMNS if row.get(k) not in (None, "")}
        blocked = bool(row.get("ontology_blocked"))
        if not carried and not blocked:
            return
        assignments = ", ".join(f"r.{k} = ${k}" for k in carried)
        if blocked:
            assignments += (", " if assignments else "") + (
                "r.ontology_blocked = coalesce(r.ontology_blocked, false) OR true"
            )
        # `key_` prefixed so a key column that is also a carried column (roles,
        # polarity and modality all are) does not collide in the parameter map.
        key_where = "".join(f" AND r.{column} = $key_{column}" for column in (merge_key or {}))
        key_params = {f"key_{column}": value for column, value in (merge_key or {}).items()}
        await self.command(
            f"MATCH (a:ENTITY)-[r:REL]->(b:ENTITY) "
            f"WHERE a.name_norm = $src_nn AND b.name_norm = $dst_nn "
            f"AND r.predicate = $predicate{key_where} "
            f"SET {assignments}",
            src_nn=_name_norm(src_name),
            dst_nn=_name_norm(dst_name),
            predicate=predicate,
            **carried,
            **key_params,
        )

    async def _repoint_relation(self, src_name: str, dst_name: str, row: dict[str, Any]) -> bool:
        """Re-create one loser relation edge on the winner via write_relation.

        Reuses the audited write_relation (evidence-append dedup, max confidence,
        single-valued supersede). Each evidence chunk is replayed so the winner
        edge accumulates the loser's full provenance.

        The currency flag is carried, not defaulted, for the same reason
        ``verifier_score`` is: write_relation asserts "live as of now" by
        default, so replaying a tombstoned edge would resurrect it as current.
        single-valued supersede, sticky ontology_blocked, NULL-vs-0.0
        verifier_score). Each evidence chunk is replayed so the winner edge
        accumulates the loser's full provenance.

        A projected frame edge is replayed under the key ITS writer used
        (:func:`_frame_merge_key`), not under ``predicate`` alone — otherwise the
        repair is lossy in both directions: role and polarity variants of one
        frame between the same pair collapse onto a single edge here, and the
        survivor, lacking those properties, matches no future projection MERGE,
        so every later ingest adds a duplicate beside it.

        Returns True when the repointed edge has provenance by NEITHER route —
        no evidence chunk to replay and no ``frame_instance_id`` to follow — so
        the caller can count the residue instead of losing it silently.
        """
        predicate = row.get("predicate") or ""
        if not predicate or not src_name or not dst_name:
            return False
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
        key = _frame_merge_key(row)
        evidence = [e for e in (row.get("evidence") or []) if e]
        # An edge with no evidence is still written once, with no chunk_id: the
        # replay must not drop an edge for arriving unsourced.
        for cid in evidence or [""]:
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
                merge_key=key,
            )
        await self._carry_rel_columns(src_name, predicate, dst_name, row, merge_key=key)
        # frame_instance_id is carried above, so a frame edge with no evidence is
        # still reachable to its chunks via FRAME_EVOKED_IN. Without either, this
        # edge is unsourced by both routes.
        return not evidence and not row.get("frame_instance_id")

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
        # Both directions read the SAME columns (_REL_ROW_SELECT): the structural
        # ones are what write_relation cannot re-derive, and dropping them
        # silently downgrades a merged endpoint's edges — a frame projection
        # loses its roles and its `fn:` canonical_predicate and comes back as an
        # unreified `lemma:` edge, and `ontology_blocked` (sticky by design,
        # because "a refusal is a verdict, not an absence of one") is cleared,
        # re-publishing exactly the edges it exists to refuse.
        # Reading the columns is half of it: the replay also has to re-use the
        # MERGE KEY of whichever writer minted the edge (see _repoint_relation),
        # or two frame edges between the same pair that differ only in role or
        # polarity are merged into one on the way back.
        out_rows = await self.query(
            f"MATCH (:ENTITY {{name_norm: $loser}})-[r:REL]->(b:ENTITY) "
            f"RETURN b.name AS dst, b.name_norm AS dst_nn, {_REL_ROW_SELECT}",
            loser=loser_norm,
        )
        in_rows = await self.query(
            f"MATCH (a:ENTITY)-[r:REL]->(:ENTITY {{name_norm: $loser}}) "
            f"RETURN a.name AS src, a.name_norm AS src_nn, {_REL_ROW_SELECT}",
            loser=loser_norm,
        )
        # An endpoint with neither a name nor a name_norm cannot be repointed:
        # passing None through would recreate the edge against a null-named
        # node, i.e. manufacture the junk entity the merge exists to remove.
        unsourced = 0
        for row in out_rows:
            dst = row.get("dst") or row.get("dst_nn")
            if not dst or row.get("dst_nn") == winner_norm:
                continue
            unsourced += await self._repoint_relation(winner_name, dst, row)
        for row in in_rows:
            src = row.get("src") or row.get("src_nn")
            if not src or row.get("src_nn") == winner_norm:
                continue
            unsourced += await self._repoint_relation(src, winner_name, row)

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
        logger.info(
            "merge_entities: %s -> %s (%d repointed edge(s) with no recoverable "
            "provenance: no r.evidence and no frame_instance_id)",
            loser_norm,
            winner_norm,
            unsourced,
        )
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
        """Delete a session's FRAME_INSTANCE/CHUNK/FILE vertices and its SESSION node.

        Namespace-global vertices (ENTITY, TOPIC, TEMPORAL and now FRAME) are
        untouched — they may be evidenced by other sessions. FRAME joined that
        list when it became canonical: one vertex serves every session that
        evokes it, and it may additionally be a parent in the BROADER taxonomy
        with no evidence of its own. Its evidence for THIS session is the
        EVOKED_BY edge, which goes when the chunk it points at is detached.

        Raises:
            StoreError: If any label failed to delete for a reason other than the
                type not existing yet. ``purge_memory`` reports this to the
                caller instead of claiming a clean purge over orphaned nodes.
        """
        failures: list[str] = []
        for label in ("FRAME_INSTANCE", "CHUNK", "FILE", "SESSION"):
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
        # The FRAME rows go with them, so the write-once cache must forget them:
        # an embedding-less FRAME is invisible to the channel's vector search,
        # so a stale cache would silently kill the layer rather than fail.
        self._frames_materialised.clear()
        for label in _VERTEX_LABELS:
            try:
                await self.command(f"MATCH (n:{label}) DETACH DELETE n")
            except Exception:
                logger.debug("clear_all: %s delete ignored", label, exc_info=True)


__all__ = ["GraphStore"]

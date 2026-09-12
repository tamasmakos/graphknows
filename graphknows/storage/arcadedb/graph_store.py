"""ArcadeDB single-database graph store (golden layer).

One ``GraphStore`` per namespace database (``mem`` / ``mem_<ns>``) owns the
schema and the reads over it. Writes live in
:mod:`graphknows.storage.arcadedb.writers`, one writer per vertex family; what
stays here is the schema stamp and the top-down reads retrieval runs —
segments by embedding, by text, by id and by ``NEXT`` adjacency; facts by
entity and by predicate with their ``ASSERTED_IN`` evidence; entities by name
key. Every read declares the types it touches with :func:`reads`, so the
dead-weight check can diff what a run wrote against what it read back.

ArcadeDB constraints honoured here:
- Cypher has no DDL surface — schema uses SQL.
- Vector search is SQL ``vectorNeighbors`` over an inline embedding literal
  with no FROM clause (see ``_vec_search_sql``).
- openCypher has no ``IN $list`` — every interpolated string goes through
  :mod:`graphknows.storage.arcadedb._sql`.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from functools import wraps
from typing import Any, TypeVar, cast

from graphknows.exceptions import SchemaVersionMismatchError, StoreError
from graphknows.storage.arcadedb._base import ArcadeStoreBase
from graphknows.storage.arcadedb._schema import _CORE_DDL, schema_version
from graphknows.storage.arcadedb._sql import (
    is_missing_type,
    not_forgotten,
    quote,
    quoted_list,
    sanitize,
    vector_literal,
)
from graphknows.storage.arcadedb.client import ArcadeDBClient

logger = logging.getLogger(__name__)

_DB = "mem"

# Key of the SCHEMA_STAMP singleton; the UNIQUE index on it is the singleton.
_STAMP_ID = "singleton"

# Vertex types clear_all() wipes: every golden vertex type but the stamp, which
# records the schema the (now empty) graph was built with.
_VERTEX_TYPES = tuple(
    stmt.split()[3]
    for _, stmt in _CORE_DDL
    if stmt.startswith("CREATE VERTEX TYPE ") and "SCHEMA_STAMP" not in stmt
)

# What a fact needs to come back with the segment it was read from: the
# assertion, the names on either end, and the evidence. One shape for both
# fact reads, so the retriever renders them the same way.
_FACT_ROW = (
    "f.id AS id, f.subject AS subject, f.predicate AS predicate, f.object AS object, "
    "f.confidence AS confidence, s.name AS subject_name, o.name AS object_name, "
    "g.id AS segment_id, g.text AS text, g.role AS role, g.observed_at AS observed_at, "
    "g.source_id AS source_id"
)

_SEGMENT_ROW = (
    "g.id AS id, g.text AS text, g.role AS role, g.observed_at AS observed_at, "
    "g.source_id AS source_id, g.byte_start AS byte_start"
)


_ReadMethod = TypeVar("_ReadMethod", bound=Callable[..., Any])


def reads(*types: str) -> Callable[[_ReadMethod], _ReadMethod]:
    """Declare the vertex/edge types a store read method touches (FR-040).

    The decorated method records them in the store's per-connection
    ``read_types`` set as it runs, so the dead-weight check diffs what a run
    actually read against what it wrote. Deliberately a runtime observation:
    the SQL is built as strings, so a static scan would both miss types and
    invent them. The declaration is also readable as a registry — the
    ``read_types`` attribute on the method itself.
    """
    declared = frozenset(types)

    def decorate(method: _ReadMethod) -> _ReadMethod:
        @wraps(method)
        async def wrapper(self: GraphStore, *args: Any, **kwargs: Any) -> Any:
            # Lazy: test doubles subclass GraphStore and skip __init__.
            self.__dict__.setdefault("_read_types", set()).update(declared)
            return await method(self, *args, **kwargs)

        wrapper.read_types = declared  # type: ignore[attr-defined]
        return cast(_ReadMethod, wrapper)

    return decorate


def _vec_search_sql(index: str, emb: list[float], k: int) -> str:
    """Return SQL for a vectorNeighbors ANN search.

    No FROM clause — the index already knows its type. Adding FROM <type>
    makes ArcadeDB invoke vectorNeighbors once *per vertex*, returning N*k rows
    and making deduplication order-dependent.
    """
    return f'SELECT expand(vectorNeighbors("{index}", {vector_literal(emb)}, {k}))'


def _name_norm(name: str) -> str:
    """Canonical ENTITY merge key: casefolded, whitespace-collapsed name."""
    return " ".join(name.split()).casefold()


name_norm = _name_norm


def _name_key_tokens(words: list[str]) -> list[str]:
    """The query tokens worth matching against ``name_norm``: 3+ chars, casefolded."""
    return sorted({w.strip().casefold() for w in words if w and len(w.strip()) >= 3})


def _name_key_match(alias: str, tokens: list[str]) -> str:
    """Cypher predicate: *alias*.name_norm matches any token on a WORD boundary.

    Exact, or as a whole space-delimited word of a multi-word entity ("studio"
    -> "dance studio"). Intra-word substrings do NOT match, so "there" never
    resolves "theresa".
    """

    def _one(t: str) -> str:
        return (
            f"({alias}.name_norm = {quote(t)} "
            f"OR {alias}.name_norm STARTS WITH {quote(t + ' ')} "
            f"OR {alias}.name_norm ENDS WITH {quote(' ' + t)} "
            f"OR {alias}.name_norm CONTAINS {quote(' ' + t + ' ')})"
        )

    return "(" + " OR ".join(_one(t) for t in tokens) + ")"


def _source_filter(alias: str, source_ids: list[str] | None) -> str:
    """Optional ``AND`` tail restricting *alias* to the given sources.

    ``None`` is no filter; a list is the filter, and an empty list matches
    nothing — a caller that resolved a scope to no source gets no rows rather
    than the whole namespace.
    """
    if source_ids is None:
        return ""
    return f" AND {alias}.source_id IN {quoted_list(source_ids)}"


def _dedup_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Deduplicate vectorNeighbors results by @rid."""
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for r in rows:
        rid = str(r.get("@rid", r.get("id", "")))
        if rid not in seen:
            seen.add(rid)
            out.append(r)
    return out


class GraphStore(ArcadeStoreBase):
    """Single-database ArcadeDB store for one namespace (golden layer).

    Args:
        client: Shared ``ArcadeDBClient`` (connect() is idempotent).
        db: Database name, from :func:`graphknows.storage.namespace.db_name`.
    """

    def __init__(self, client: ArcadeDBClient, db: str = _DB) -> None:
        super().__init__(client, db)
        # Types read over this connection so far, accumulated by @reads as
        # queries execute; the panel harness snapshots it for the dead-weight
        # check (FR-040).
        self._read_types: set[str] = set()

    @property
    def read_types(self) -> frozenset[str]:
        """Vertex/edge types this connection has read back so far (FR-040)."""
        return frozenset(getattr(self, "_read_types", ()))

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

    # ------------------------------------------------------------------
    # Reads — sources
    # ------------------------------------------------------------------

    @reads("SOURCE")
    async def sources(self, uri: str = "") -> list[dict[str, Any]]:
        """Enumerate SOURCE rows, newest ``imported_at`` first.

        The reader that keeps SOURCE from being a write-only ledger: it is how
        a facade reports what a namespace holds, and how a scope named by its
        uri (a session, a file) resolves to the source ids segments carry.
        """
        where = f" WHERE s.uri = {quote(uri)}" if uri else ""
        return await self.query(
            f"MATCH (s:SOURCE){where} "
            "RETURN s.id AS id, s.uri AS uri, s.mime AS mime, s.content_hash AS content_hash, "
            "s.imported_at AS imported_at, s.namespace AS namespace "
            "ORDER BY imported_at DESC"
        )

    # ------------------------------------------------------------------
    # Reads — segments
    # ------------------------------------------------------------------

    @reads("SEGMENT")
    async def search_segments_ann(
        self, embedding: list[float], top_k: int, source_ids: list[str] | None = None
    ) -> list[dict[str, Any]]:
        """ANN over ``SEGMENT.embedding`` via vectorNeighbors.

        The source filter is applied client-side over an over-fetched candidate
        set: vectorNeighbors has no WHERE support.
        """
        if not embedding:
            return []
        k_pre = max(top_k * 20, 200) if source_ids is not None else top_k
        sql = _vec_search_sql("SEGMENT[embedding]", embedding, k_pre)
        rows = _dedup_rows(await self._c.query(self._db, sql, language="sql"))
        results: list[dict[str, Any]] = []
        for r in rows:
            if source_ids is not None and r.get("source_id") not in source_ids:
                continue
            results.append(
                {
                    "id": r.get("id", ""),
                    "text": r.get("text", ""),
                    "role": r.get("role") or "",
                    "observed_at": r.get("observed_at") or "",
                    "source_id": r.get("source_id") or "",
                    "cosine_score": max(0.0, 1.0 - float(r.get("distance") or 1.0)),
                }
            )
            if len(results) >= top_k:
                break
        return results

    @reads("SEGMENT")
    async def full_text_search_segments(
        self,
        terms: list[str],
        top_k: int,
        source_ids: list[str] | None = None,
        require_all: bool = False,
    ) -> list[dict[str, Any]]:
        """SEGMENTs whose text contains the given terms, case-insensitively.

        SQL rather than Cypher for ``ILIKE``, which folds case without a
        ``toLower()`` wrapper. Ordered by ``id`` (FR-032): this is the OR-pool
        the BM25 channel re-ranks in Python, and an unordered ``LIMIT`` would
        let storage order decide which segments reach the re-rank at all.

        ponytail: a bucket scan — SEGMENT.text carries no FULL_TEXT index in the
        golden DDL; add one (a schema-version change) when the scan shows up.
        """
        safe = [sanitize(t) for t in terms if t]
        if not safe:
            return []
        op = " AND " if require_all else " OR "
        text_where = op.join(f"text ILIKE '%{t}%'" for t in safe)
        source_where = ""
        if source_ids is not None:
            source_where = f" AND source_id IN {quoted_list(source_ids)}"
        rows = await self._c.query(
            self._db,
            f"SELECT id, text, role, observed_at, source_id FROM SEGMENT "  # nosec B608
            f"WHERE ({text_where}){source_where} ORDER BY id LIMIT {int(top_k)}",
            language="sql",
        )
        return [
            {
                "id": r["id"],
                "text": r.get("text", ""),
                "role": r.get("role") or "",
                "observed_at": r.get("observed_at") or "",
                "source_id": r.get("source_id") or "",
            }
            for r in rows
        ]

    @reads("SEGMENT")
    async def get_segments_by_ids(self, segment_ids: list[str]) -> dict[str, dict[str, Any]]:
        """Fetch segments by id. Returns ``{id: {text, role, observed_at, source_id, ...}}``."""
        ids = [s for s in segment_ids if s]
        if not ids:
            return {}
        rows = await self.query(
            f"MATCH (g:SEGMENT) WHERE g.id IN {quoted_list(ids)} RETURN {_SEGMENT_ROW}"
        )
        return {r["id"]: r for r in rows if r.get("id")}

    @reads("SEGMENT", "NEXT")
    async def neighbor_segments(
        self, segment_ids: list[str], radius: int = 1
    ) -> dict[str, list[dict[str, Any]]]:
        """``{seed id: [neighbour {id, text, byte_start}, ...]}`` within *radius* NEXT hops.

        Walked one hop per round trip, either direction, each newly reached
        segment credited to the seed it was reached from; a segment reachable
        from two seeds is credited to the first. Long ingested messages split
        into consecutive segments, so an answer often lives in a sibling of the
        segment retrieval matched.
        """
        ids = [s for s in segment_ids if s]
        if not ids or radius < 1:
            return {}
        owner: dict[str, str] = {s: s for s in ids}
        out: dict[str, list[dict[str, Any]]] = {}
        frontier = list(ids)
        for _ in range(radius):
            rows = await self.query(
                f"MATCH (g:SEGMENT)-[:NEXT]-(n:SEGMENT) WHERE g.id IN {quoted_list(frontier)} "
                "RETURN g.id AS seed, n.id AS id, n.text AS text, n.byte_start AS byte_start"
            )
            frontier = []
            for r in rows:
                nid = str(r.get("id") or "")
                if not nid or nid in owner:
                    continue
                seed = owner[str(r["seed"])]
                owner[nid] = seed
                frontier.append(nid)
                out.setdefault(seed, []).append(
                    {
                        "id": nid,
                        "text": r.get("text") or "",
                        "byte_start": int(r.get("byte_start") or 0),
                    }
                )
            if not frontier:
                break
        return out

    @reads("SEGMENT", "MENTIONS", "ENTITY")
    async def segments_mentioning(
        self, words: list[str], top_k: int, source_ids: list[str] | None = None
    ) -> list[dict[str, Any]]:
        """SEGMENTs linked (via MENTIONS) to an entity a query token names.

        Ordered by ``id`` (FR-032): this collector feeds RRF rank straight from
        row position, and an unordered ``LIMIT`` would let storage order decide
        which segments reach fusion, and in which order.
        """
        toks = _name_key_tokens(words)
        if not toks:
            return []
        return await self.query(
            "MATCH (g:SEGMENT)-[:MENTIONS]->(e:ENTITY) "
            f"WHERE {_name_key_match('e', toks)} AND {not_forgotten('e')}"
            f"{_source_filter('g', source_ids)} "
            f"RETURN DISTINCT {_SEGMENT_ROW} ORDER BY id LIMIT $k",
            k=top_k,
        )

    @reads("CONCEPT", "EVOKES", "SEGMENT")
    async def segments_evoking(
        self,
        embedding: list[float],
        concept_top_k: int,
        source_ids: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """ANN over ``CONCEPT.embedding``, then the segments that EVOKE the matches.

        The concept index is retrievable by the embedding of its definition —
        that is what makes it a symbol rather than a side-car store — and the
        EVOKES edge is what bottom-up labelling wrote. Each row scores the
        query-to-concept match weighted by how strongly the segment evoked it.
        Returns ``[{id, uri, label, score}]``.
        """
        if not embedding:
            return []
        sql = _vec_search_sql("CONCEPT[embedding]", embedding, concept_top_k)
        score: dict[str, float] = {}
        label: dict[str, str] = {}
        for r in _dedup_rows(await self._c.query(self._db, sql, language="sql")):
            uri = r.get("uri")
            if not uri or uri in score:
                continue
            score[uri] = max(0.0, 1.0 - float(r.get("distance") or 1.0))
            label[uri] = r.get("label", "")
        if not score:
            return []
        rows = await self.query(
            "MATCH (g:SEGMENT)-[v:EVOKES]->(c:CONCEPT) "
            f"WHERE c.uri IN {quoted_list(list(score))}{_source_filter('g', source_ids)} "
            "RETURN g.id AS id, c.uri AS uri, v.confidence AS confidence"
        )
        return [
            {
                "id": r["id"],
                "uri": r["uri"],
                "label": label.get(r["uri"], ""),
                "score": score.get(r["uri"], 0.0) * float(r.get("confidence") or 0.0),
            }
            for r in rows
            if r.get("id") and r.get("uri")
        ]

    # ------------------------------------------------------------------
    # Reads — entities and facts
    # ------------------------------------------------------------------

    @reads("ENTITY")
    async def resolve_query_entities(self, words: list[str], cap: int = 6) -> list[dict[str, Any]]:
        """The entities query tokens name, as ``[{id, name}]``, ranked before the cap.

        A token matches ``name_norm`` only on a WORD boundary (see
        :func:`_name_key_match`). Ranking is exact-token first, then shortest
        name, then alphabetical: an exact hit beats a span that merely contains
        the token, and the shorter of two matches is the more canonical one
        ("Gina" over "Gina's hard work and effort, Gina"). Alphabetical last so
        the result is deterministic rather than merely better. The cap binds —
        measured on conv-30, every question hit it — so it is applied after
        the ranking, never as a bare ``LIMIT``.
        """
        toks = _name_key_tokens(words)
        if not toks:
            return []
        rows = await self.query(
            f"MATCH (e:ENTITY) WHERE {_name_key_match('e', toks)} AND {not_forgotten('e')} "
            "RETURN DISTINCT e.id AS id, e.name AS name, e.name_norm AS nn"
        )
        tokset = set(toks)
        ranked = sorted(
            (r for r in rows if r.get("id") and r.get("name")),
            key=lambda r: (r.get("nn") not in tokset, len(r.get("nn") or ""), r.get("nn") or ""),
        )
        return [{"id": r["id"], "name": r["name"]} for r in ranked[: int(cap)]]

    @reads("ENTITY", "SUBJECT_OF", "FACT", "OBJECT_IS", "ASSERTED_IN", "SEGMENT")
    async def facts_by_entity(self, entity_ids: list[str], limit: int = 40) -> list[dict[str, Any]]:
        """Current facts with an entity at either end, each with one evidence row.

        Both directions, so object-side facts surface ("Caroline HAS Luna" when
        the query resolves Luna). Ordered by confidence, then id, so the cap
        cuts along a stated ordering.
        """
        ids = [e for e in entity_ids if e]
        if not ids:
            return []
        id_lit = quoted_list(ids)
        return await self.query(
            "MATCH (s:ENTITY)-[:SUBJECT_OF]->(f:FACT)-[:OBJECT_IS]->(o:ENTITY) "
            f"WHERE (s.id IN {id_lit} OR o.id IN {id_lit}) "
            f"AND {not_forgotten('s')} AND {not_forgotten('o')} AND {not_forgotten('f')} "
            "AND f.is_current = true "
            "MATCH (f)-[:ASSERTED_IN]->(g:SEGMENT) "
            f"RETURN {_FACT_ROW} ORDER BY confidence DESC, id LIMIT $k",
            k=limit,
        )

    @reads(
        "PREDICATE", "USES", "FACT", "SUBJECT_OF", "OBJECT_IS", "ENTITY", "ASSERTED_IN", "SEGMENT"
    )
    async def facts_by_predicate(
        self, keys: list[str], limit: int = 40, source_ids: list[str] | None = None
    ) -> list[dict[str, Any]]:
        """Current facts USING a predicate one of *keys* names, with their evidence.

        A key matches the predicate id or its canonical form, so a query word
        reaches the facts asserted under that relation without the entity ever
        being named.
        """
        safe = sorted({k.strip().casefold() for k in keys if k and k.strip()})
        if not safe:
            return []
        key_lit = quoted_list(safe)
        return await self.query(
            "MATCH (s:ENTITY)-[:SUBJECT_OF]->(f:FACT)-[:OBJECT_IS]->(o:ENTITY), "
            "(f)-[:USES]->(p:PREDICATE) "
            f"WHERE (p.id IN {key_lit} OR p.canonical IN {key_lit}) "
            f"AND {not_forgotten('s')} AND {not_forgotten('o')} AND {not_forgotten('f')} "
            "AND f.is_current = true "
            "MATCH (f)-[:ASSERTED_IN]->(g:SEGMENT) "
            f"WHERE true{_source_filter('g', source_ids)} "
            f"RETURN {_FACT_ROW} ORDER BY confidence DESC, id LIMIT $k",
            k=limit,
        )

    # ------------------------------------------------------------------
    # Deletion
    # ------------------------------------------------------------------

    async def delete_source(self, uri: str) -> None:
        """Delete every SOURCE at *uri*, its SEGMENTs, and the FACTs asserted in them.

        A purge, not a tombstone: what was learned from the source goes with
        it. Entities, concepts and predicates are namespace-global and stay —
        they may be evidenced elsewhere.

        Raises:
            StoreError: A delete failed for a reason other than the type not
                existing yet, so ``purge_memory`` reports it instead of
                claiming a clean purge over orphaned nodes.
        """
        failures: list[str] = []
        for label, stmt in (
            (
                "FACT",
                "MATCH (f:FACT)-[:ASSERTED_IN]->(:SEGMENT)-[:PART_OF]->(s:SOURCE {uri: $uri}) "
                "DETACH DELETE f",
            ),
            ("SEGMENT", "MATCH (g:SEGMENT)-[:PART_OF]->(s:SOURCE {uri: $uri}) DETACH DELETE g"),
            ("SOURCE", "MATCH (s:SOURCE {uri: $uri}) DETACH DELETE s"),
        ):
            try:
                await self.command(stmt, uri=uri)
            except Exception as exc:
                if is_missing_type(exc):
                    logger.debug("delete_source: %s type absent, nothing to delete", label)
                    continue
                logger.warning("delete_source: %s delete failed: %s", label, exc)
                failures.append(f"{label}: {exc}")
        if failures:
            raise StoreError(f"delete_source({uri!r}) failed for: {'; '.join(failures)}")

    async def clear_all(self) -> None:
        """Delete every golden vertex (and, via DETACH, every edge) in this database.

        The schema stamp stays: the emptied graph is still the schema it was
        built with.
        """
        for label in _VERTEX_TYPES:
            try:
                await self.command(f"MATCH (n:{label}) DETACH DELETE n")
            except Exception as exc:
                if not is_missing_type(exc):
                    raise
                logger.debug("clear_all: %s type absent, nothing to delete", label)


__all__ = ["GraphStore"]

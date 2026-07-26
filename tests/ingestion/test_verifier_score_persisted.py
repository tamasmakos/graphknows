"""The relation verifier's judgment must survive ingest onto the REL edge.

``RelationVerifierGate.filter`` scores every relation it gates and records the
result on the relation dict, but that score was read nowhere and written
nowhere — the edge kept only ``confidence``, which for SVO-mined triples is a
fixed structural prior. This test pins the score to the edge end to end: through
the ingest path, into the graph, readable by query.

Semantics under test:
  * persisted   — a scored relation carries ``verifier_score`` on its edge
  * keeps max   — a duplicate (src, predicate, dst) keeps the strongest score,
                  mirroring how ``confidence`` already merges
  * absent null — a relation the verifier never scored (passthrough, or the
                  gate disabled) leaves the property null, so "not scored" stays
                  distinguishable from "scored 0.0"
"""

from __future__ import annotations

import uuid
from typing import Any

import pytest

from graphknows.ingestion.extraction.entities.extractor import ExtractionResult


def _arcadedb_reachable() -> bool:
    """Return True when ArcadeDB is up at the configured URL."""
    import httpx

    from graphknows.settings import GraphKnowsSettings

    try:
        s = GraphKnowsSettings()
        r = httpx.get(
            f"{s.arcadedb_url}/api/v1/exists/ltm",
            auth=(s.arcadedb_user, s.arcadedb_password),
            timeout=3,
        )
    except Exception:
        return False
    else:
        return r.status_code == 200


class _StubExtractor:
    """Yields a fixed ExtractionResult per call, standing in for relex+verifier.

    The verifier itself is covered by tests/memory/test_relation_verifier.py;
    what is untested is whether the score it sets survives the write path.
    """

    def __init__(self, results: list[ExtractionResult]) -> None:
        self._results = list(results)

    def extract(self, text: str, *args: Any, **kwargs: Any) -> ExtractionResult:
        return self._results.pop(0) if self._results else ExtractionResult([], [])


def _rel(head: str, relation: str, tail: str, score: float, **extra: Any) -> dict[str, Any]:
    return {
        "head": head,
        "relation": relation,
        "tail": tail,
        "score": score,
        "source": "svo",
        "head_type": "PERSON",
        "tail_type": "ORG",
        **extra,
    }


@pytest.mark.integration
@pytest.mark.asyncio
async def test_verifier_score_persists_to_rel_edge() -> None:
    """A gated relation's verifier_score reaches the graph; max wins; absent is null."""
    if not _arcadedb_reachable():
        pytest.skip("ArcadeDB not reachable")

    from graphknows.ingestion.stm.ingest import _STMIngestHandler
    from graphknows.settings import GraphKnowsSettings
    from graphknows.storage import build_arcadedb_client
    from graphknows.storage.arcadedb.graph_store import GraphStore

    settings = GraphKnowsSettings()
    store = GraphStore(
        client=build_arcadedb_client(settings), db=f"mem_test_{uuid.uuid4().hex[:8]}"
    )
    await store.connect()
    try:
        await store.ensure_schema(dims=8)

        entities = [
            {"name": "Gina", "type": "PERSON", "score": 0.9},
            {"name": "clothing store", "type": "ORG", "score": 0.8},
        ]
        handler = _STMIngestHandler(
            store=store,
            extractor=_StubExtractor(
                [
                    # First chunk: one scored relation, one the gate never scored.
                    ExtractionResult(
                        entities=entities,
                        relations=[
                            _rel("Gina", "HAS", "clothing store", 0.6, verifier_score=0.6079),
                            _rel("Gina", "BE_BY", "clothing store", 0.6),
                        ],
                    ),
                    # Second chunk: same triple, weaker score — max must win.
                    ExtractionResult(
                        entities=entities,
                        relations=[
                            _rel("Gina", "HAS", "clothing store", 0.6, verifier_score=0.5123),
                        ],
                    ),
                ]
            ),
            embed_fn=lambda _t: [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        )

        await handler.ingest_text("Gina has a clothing store.", session_id="sess-vs")
        await handler.ingest_text("Gina runs the clothing store.", session_id="sess-vs")

        rows = await store.query(
            "MATCH (a:ENTITY {name_norm: 'gina'})-[r:REL]->(b:ENTITY) "
            "RETURN r.predicate AS predicate, r.confidence AS confidence, "
            "r.verifier_score AS verifier_score"
        )
        by_pred = {r["predicate"]: r for r in rows}

        assert "HAS" in by_pred, f"HAS edge not written; got {sorted(by_pred)}"
        # Strongest verifier judgment survives the duplicate merge.
        assert by_pred["HAS"]["verifier_score"] == pytest.approx(0.6079)
        # confidence is untouched — this change is additive.
        assert by_pred["HAS"]["confidence"] == pytest.approx(0.6)

        assert "BE_BY" in by_pred, f"BE_BY edge not written; got {sorted(by_pred)}"
        # Never scored by the gate — must stay null, not collapse to 0.0.
        assert by_pred["BE_BY"]["verifier_score"] is None
    finally:
        await store.drop_database()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_verifier_score_survives_entity_merge() -> None:
    """Entity-hygiene repair re-points relation edges; the score must survive it.

    ``merge_entities`` replays the loser's edges onto the winner through
    ``_repoint_relation`` and then DETACH DELETEs the loser. The repair runs on
    every flush and historically touched a large share of REL edges, so a score
    dropped here is a score wiped at the moment the graph is finalised — and
    wiped to NULL, which this module defines to mean "never scored".
    """
    if not _arcadedb_reachable():
        pytest.skip("ArcadeDB not reachable")

    from graphknows.settings import GraphKnowsSettings
    from graphknows.storage import build_arcadedb_client
    from graphknows.storage.arcadedb.graph_store import GraphStore

    settings = GraphKnowsSettings()
    store = GraphStore(
        client=build_arcadedb_client(settings), db=f"mem_test_{uuid.uuid4().hex[:8]}"
    )
    await store.connect()
    try:
        await store.ensure_schema(dims=8)

        # The loser carries a scored outgoing edge and a scored incoming one.
        await store.write_relation(
            "Gina Hey",
            "HAS",
            "clothing store",
            chunk_id="chunk-a",
            confidence=0.6,
            source="svo",
            verifier_score=0.6079,
        )
        await store.write_relation(
            "Jon",
            "SUPPORT",
            "Gina Hey",
            chunk_id="chunk-a",
            confidence=0.92,
            source="relex",
            verifier_score=0.7690,
        )
        # Winner must exist before the merge.
        await store.write_relation(
            "Gina",
            "LIVE",
            "downtown",
            chunk_id="chunk-a",
            confidence=0.5,
            source="svo",
            verifier_score=0.5500,
        )

        assert await store.merge_entities("gina", "gina hey") is True

        rows = await store.query(
            "MATCH (a:ENTITY)-[r:REL]->(b:ENTITY) "
            "RETURN a.name AS src, r.predicate AS predicate, b.name AS dst, "
            "r.verifier_score AS verifier_score"
        )
        by_pred = {r["predicate"]: r for r in rows}

        assert "HAS" in by_pred, f"re-pointed outgoing edge missing; got {sorted(by_pred)}"
        assert by_pred["HAS"]["src"] == "Gina"
        assert by_pred["HAS"]["verifier_score"] == pytest.approx(0.6079)

        assert "SUPPORT" in by_pred, f"re-pointed incoming edge missing; got {sorted(by_pred)}"
        assert by_pred["SUPPORT"]["dst"] == "Gina"
        assert by_pred["SUPPORT"]["verifier_score"] == pytest.approx(0.7690)
    finally:
        await store.drop_database()

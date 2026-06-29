"""Shared helpers for writing entity nodes + edges + relation edges to the STM store.

Both _STMIngestHandler and DialogCapture follow the same write pattern but use
different edge types (chunk vs. turn).  The edge_writer callable is the only
difference — callers supply a lambda that calls the appropriate edge method.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any, Coroutine

# spaCy NER labels that are numeric/temporal noise as graph entities. Dates and
# times are captured separately as canonical TEMPORAL nodes; numbers/percents/
# money saturate the entity graph and pollute community/topic membership without
# ever serving as useful retrieval discriminators.
_NON_GRAPH_ENTITY_TYPES = frozenset(
    {"DATE", "TIME", "CARDINAL", "ORDINAL", "PERCENT", "MONEY", "QUANTITY"}
)

# GLiNER often leaves date/time/number spans UNTYPED (type=None), so the type
# allowlist above misses them and they leak in as ENTITY nodes (e.g. "23 July
# 2023", "10:31 am", "2023") — corpus-polluting noise that also corrupts
# community detection. This pattern catches them by SHAPE. It requires a digit
# alongside any month word, so real person/place names ("May", "June") survive.
_TEMPORAL_NAME_RE = re.compile(
    r"^("
    r"[\W\d]+"  # bare numbers / punctuation / "10:31" / "100" / "4"
    r"|\d{1,2}:\d{2}\s*[ap]\.?m\.?"  # 10:31 am
    r"|\d{1,2}\s*[ap]\.?m\.?"  # 10 am / 10am
    r"|\d{1,4}\s+(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*(?:\s+\d{1,4})?"  # 23 July (2023)
    r"|(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\s+\d{1,4}"  # July 2023
    r"|\d{4}"  # bare year
    r")$",
    re.IGNORECASE,
)


def _is_temporal_or_numeric_name(name: str) -> bool:
    return bool(_TEMPORAL_NAME_RE.match((name or "").strip()))


def graph_entities(entities: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop numeric/temporal spans that should not become ENTITY nodes.

    Filters by spaCy/GLiNER type AND by name shape — the latter catches the
    common case where the extractor leaves a date/time/number untyped.
    """
    return [
        e
        for e in entities
        if (e.get("type") or "").upper() not in _NON_GRAPH_ENTITY_TYPES
        and not _is_temporal_or_numeric_name(e.get("name", ""))
    ]


def write_entity_nodes_and_edges(
    mg: Any,
    session_id: str,
    entities: list[dict[str, Any]],
    edge_writer: Callable[[dict[str, Any]], None],
    _run: Callable[..., Any],
) -> int:
    """Write entity nodes and their source edge. Returns the entity count.

    Args:
        mg: ArcadeDBSTMStore (or ISessionDb) for graph writes.
        session_id: Session namespace.
        entities: List of entity dicts with "name", optional "type" and "score".
        edge_writer: Callable(ent_dict) that writes the edge linking the entity
            to its source node (chunk or turn).  Callers supply this to avoid
            coupling the helper to a specific edge method.
        _run: Synchronous bridge for async store calls.
    """
    entity_count = 0
    for ent in graph_entities(entities):
        _run(
            mg.write_entity_node(
                session_id=session_id,
                entity_name=ent["name"],
                entity_type=ent.get("type", "ENTITY"),
                confidence=ent.get("score", 1.0),
            )
        )
        edge_writer(ent)
        entity_count += 1
    return entity_count


async def write_chunk_entities_and_edges_bulk(
    mg: Any,
    session_id: str,
    chunk_id: str,
    entities: list[dict[str, Any]],
) -> int:
    """Write all entity nodes + CHUNK→ENTITY edges for one chunk via bulk store method.

    This is the async (await-native) counterpart to write_entity_nodes_and_edges,
    designed for the concurrent chunk path in _ingest_parsed where coroutines
    await store methods directly (no _run nesting).  It applies the same
    graph_entities filter and returns the same entity count.

    The existing write_entity_nodes_and_edges function is left intact for the
    DialogCapture (turn) path which uses the _run sync bridge.

    Args:
        mg: ArcadeDBSTMStore instance.
        session_id: Session namespace.
        chunk_id: The CHUNK node id to link entities to.
        entities: Raw extraction entities (will be filtered by graph_entities).

    Returns:
        Number of graph entities written.
    """
    filtered = graph_entities(entities)
    if filtered:
        await mg.write_chunk_entities_bulk(
            session_id=session_id,
            chunk_id=chunk_id,
            entities=filtered,
        )
    return len(filtered)


def write_relation_edges(
    mg: Any,
    session_id: str,
    relations: list[dict[str, Any]],
    _run: Callable[..., Any],
) -> None:
    """Write relation edges from extraction results.

    Args:
        mg: ArcadeDBSTMStore (or ISessionDb) for graph writes.
        session_id: Session namespace.
        relations: List of relation dicts with "head", "relation", "tail", optional "score".
        _run: Synchronous bridge for async store calls.
    """
    for rel in relations:
        _run(
            mg.write_relation_edge(
                session_id=session_id,
                subject_name=rel["head"],
                relation_name=rel["relation"],
                object_name=rel["tail"],
                confidence=rel.get("score", 0.5),
            )
        )

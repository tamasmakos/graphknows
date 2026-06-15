"""Module-level Cypher writers for evidence, interaction, and relation graphs."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from ._helpers import _safe_rel_label


def _write_relation_edges(
    s: Any,
    triplets: list[tuple[str, str, str] | tuple[str, str, str, dict[str, Any]]],
    entity_by_name: dict[str, dict[str, Any]],
    session_id: str = "",
) -> int:
    """Write typed directed edges between ENTITY nodes using dynamic Cypher labels.

    Each triplet is (source_name, relation_label, target_name). The relation_label
    becomes an actual Cypher relationship type (e.g. REGULATES, FUNDS). Triplets
    whose endpoints cannot be matched to a known entity are silently skipped.

    Name matching is case-insensitive: the extractor may return uppercased names
    (e.g. "GRAPHRAG") while entity_by_name stores them as-extracted ("GraphRAG").

    Returns:
        Number of edges written.
    """
    # Build a case-insensitive index so we can resolve canonical names.
    # Normalize underscores to spaces so DSPy labels like "NEXUS_FOUNDATION"
    # match GLiNER2-extracted names like "Nexus Foundation".
    canonical: dict[str, str] = {k.upper().replace("_", " "): k for k in entity_by_name}

    edges = 0
    for item in triplets:
        src, rel, tgt = item[0], item[1], item[2]
        metadata = item[3] if len(item) > 3 and isinstance(item[3], dict) else {}
        src_canon = canonical.get(src.upper().replace("_", " "), src)
        tgt_canon = canonical.get(tgt.upper().replace("_", " "), tgt)
        safe = _safe_rel_label(rel)
        if not safe:
            continue
        src_type = metadata.get("source_type") or entity_by_name.get(src_canon, {}).get(
            "type", "ENTITY"
        )
        tgt_type = metadata.get("target_type") or entity_by_name.get(tgt_canon, {}).get(
            "type", "ENTITY"
        )
        s.run(
            "MERGE (e1:ENTITY {name: $src}) "
            "SET e1.type = coalesce(e1.type, $src_type) "
            "MERGE (e2:ENTITY {name: $tgt}) "
            "SET e2.type = coalesce(e2.type, $tgt_type) "
            f"MERGE (e1)-[r:{safe}]->(e2) "
            "SET r.weight = coalesce(r.weight, 0) + 1, "
            "r.session_id = $session_id, "
            "r.confidence = coalesce($confidence, r.confidence), "
            "r.evidence = coalesce($evidence, r.evidence), "
            "r.reasoning = coalesce($reasoning, r.reasoning)",
            src=src_canon,
            tgt=tgt_canon,
            src_type=src_type,
            tgt_type=tgt_type,
            session_id=session_id,
            confidence=metadata.get("confidence"),
            evidence=metadata.get("evidence"),
            reasoning=metadata.get("reasoning"),
        )
        edges += 1
    return edges


def _write_evidence_graph(
    s: Any,
    session_id: str,
    stm_data: dict[str, Any],
    entity_by_name: dict[str, dict[str, Any]],
    now: str,
) -> tuple[int, int, dict[str, list[str]]]:
    """MERGE SESSION→FILE→CHUNK→ENTITY into *s*. Returns (nodes, edges, chunk_to_ents).

    CO_OCCURS edges are NOT written here. Relation extraction runs separately in
    flush.py using the returned chunk_to_ents map.
    """
    nodes, edges = 1, 0
    s.run("MERGE (sess:SESSION {id: $sid}) SET sess.flushed_at = $now", sid=session_id, now=now)
    for f in stm_data.get("files", []):
        s.run(
            "MERGE (f:FILE {id: $fid}) "
            "SET f.title=$title, f.parser_name=$parser_name, f.source_hint=$source_hint "
            "WITH f MATCH (sess:SESSION {id: $sid}) MERGE (sess)-[:IN_SESSION]->(f)",
            fid=f["id"],
            title=f.get("title", ""),
            parser_name=f.get("parser_name", ""),
            source_hint=f.get("source_hint", ""),
            sid=session_id,
        )
        nodes += 1
        edges += 1
    for c in stm_data.get("chunks", []):
        s.run(
            "MERGE (c:CHUNK {id: $cid}) "
            "SET c.text=$text, c.heading_path=$hp, c.chunk_index=$ci "
            "WITH c MATCH (f:FILE {id: $fid}) MERGE (f)-[:HAS_CHUNK]->(c)",
            cid=c["id"],
            text=c.get("text", ""),
            hp=c.get("heading_path", ""),
            ci=c.get("chunk_index", 0),
            fid=c["file_id"],
        )
        nodes += 1
        edges += 1
    written: set[str] = set()
    chunk_to_ents: dict[str, list[str]] = defaultdict(list)
    for ce in stm_data.get("chunk_entity_edges", []):
        ename = ce["entity_name"]
        chunk_to_ents[ce["chunk_id"]].append(ename)
        em = entity_by_name.get(ename, {"type": "UNKNOWN", "confidence": 1.0})
        if ename not in written:
            s.run(
                "MERGE (e:ENTITY {name: $n}) SET e.type=$t, e.confidence=$c",
                n=ename,
                t=em.get("type", "UNKNOWN"),
                c=em.get("confidence", 1.0),
            )
            nodes += 1
            written.add(ename)
        s.run(
            "MATCH (c:CHUNK {id: $cid}), (e:ENTITY {name: $n}) MERGE (c)-[:MENTIONS]->(e)",
            cid=ce["chunk_id"],
            n=ename,
        )
        edges += 1
    return nodes, edges, dict(chunk_to_ents)


def _write_interaction_graph(
    s: Any,
    session_id: str,
    stm_data: dict[str, Any],
    entity_by_name: dict[str, dict[str, Any]],
    base_written: set[str],
) -> tuple[int, int]:
    """MERGE SESSION→DIALOG→TURN + NEXT + TURN→ENTITY into *s*. Returns (nodes, edges)."""
    nodes, edges = 0, 0
    for d in stm_data.get("dialogs", []):
        s.run(
            "MERGE (d:DIALOG {id: $did}) SET d.session_id=$sid, d.started_at=$sa "
            "WITH d MATCH (sess:SESSION {id: $sid}) MERGE (sess)-[:IN_SESSION]->(d)",
            did=d["id"],
            sid=session_id,
            sa=d.get("started_at", ""),
        )
        nodes += 1
        edges += 1
    for t in stm_data.get("turns", []):
        s.run(
            "MERGE (t:TURN {id: $tid}) "
            "SET t.session_id=$sid, t.turn_index=$ti, t.role=$r, t.text=$txt "
            "WITH t MATCH (d:DIALOG {id: $did}) MERGE (d)-[:HAS_TURN]->(t)",
            tid=t["id"],
            sid=session_id,
            ti=t.get("turn_index", 0),
            r=t.get("role", "unknown"),
            txt=t.get("text", ""),
            did=t["dialog_id"],
        )
        nodes += 1
        edges += 1
    for ne in stm_data.get("next_edges", []):
        s.run(
            "MATCH (t1:TURN {id: $a}), (t2:TURN {id: $b}) MERGE (t1)-[:NEXT]->(t2)",
            a=ne["from_turn_id"],
            b=ne["to_turn_id"],
        )
        edges += 1
    written = set(base_written)
    for te in stm_data.get("turn_entity_edges", []):
        ename = te["entity_name"]
        em = entity_by_name.get(ename, {"type": "UNKNOWN", "confidence": 1.0})
        if ename not in written:
            s.run(
                "MERGE (e:ENTITY {name: $n}) SET e.type=$t, e.confidence=$c",
                n=ename,
                t=em.get("type", "UNKNOWN"),
                c=em.get("confidence", 1.0),
            )
            nodes += 1
            written.add(ename)
        s.run(
            "MATCH (t:TURN {id: $tid}), (e:ENTITY {name: $n}) MERGE (t)-[:HAS_ENTITY]->(e)",
            tid=te["turn_id"],
            n=ename,
        )
        edges += 1
    return nodes, edges

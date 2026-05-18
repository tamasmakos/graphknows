#!/usr/bin/env python3
"""demo_edges.py — Exercise E3, E4, E5, E6 without a running stack.

Run with:
    cd /root/graphknows
    python research/demo_edges.py

Each section degrades gracefully when the relevant service is not reachable.
Required env vars (all optional — sensible defaults shown):
    NEO4J_URI          bolt://localhost:7687
    NEO4J_USER         neo4j
    NEO4J_PASSWORD     testpassword123
    MEMGRAPH_URI       bolt://localhost:7688
    GRAPHKNOWS_URL     http://localhost:8025
"""
from __future__ import annotations

import asyncio
import os
import sys
import textwrap

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SEP = "─" * 72


def header(title: str) -> None:
    print(f"\n{SEP}")
    print(f"  {title}")
    print(SEP)


def ok(msg: str) -> None:
    print(f"  ✓ {msg}")


def warn(msg: str) -> None:
    print(f"  ⚠ {msg}")


def section(label: str, body: str) -> None:
    indented = textwrap.indent(body.strip(), "    ")
    print(f"\n[{label}]\n{indented}")


# ---------------------------------------------------------------------------
# E3 — GLiNER2 + DSPy pipeline is the explicit default
# ---------------------------------------------------------------------------

def demo_e3() -> None:
    header("E3 · GLiNER2+DSPy is the explicit default pipeline")
    try:
        # Import settings without loading any heavy models
        sys.path.insert(0, "/root/graphknows/services/graphknows")
        from graphgen.config.settings import ExtractionSettings
        s = ExtractionSettings()
        backend = s.ner_backend
        ok(f"ExtractionSettings.ner_backend default = {backend!r}")
        assert backend == "gliner2", f"Expected 'gliner2', got {backend!r}"
        ok("Default confirmed as 'gliner2' — GLiNER2+DSPy is the active pipeline.")
    except Exception as exc:
        warn(f"E3 settings check failed: {exc}")

    # Show extraction fallback logic (dry-run, no model loaded)
    try:
        from graphgen.pipeline.entity_relation.extraction import (
            _build_relation_eligible_entities,
        )
        sample = [
            {"text": "URSULA_VON_DER_LEYEN", "label": "PERSON", "ontology_label": "PERSON", "confidence": 0.9},
            {"text": "EUROPEAN_COMMISSION", "label": "ORGANIZATION", "ontology_label": "ORGANIZATION", "confidence": 0.8},
            {"text": "EU", "label": "ORGANIZATION", "ontology_label": "ORGANIZATION", "confidence": 0.85},
        ]
        eligible = _build_relation_eligible_entities(sample)
        ok(f"_build_relation_eligible_entities() → {eligible}")
        assert "URSULA_VON_DER_LEYEN" in eligible
        assert "EU" not in eligible, "'EU' is too short and should be filtered"
        ok("Short entity 'EU' correctly filtered by the heuristic.")
    except Exception as exc:
        warn(f"E3 extraction filter check failed: {exc}")


# ---------------------------------------------------------------------------
# E4 — Schemaless taxonomy: all entity type names live in YAML
# ---------------------------------------------------------------------------

def demo_e4() -> None:
    header("E4 · Schemaless taxonomy — entity types from YAML, zero hardcodes in Python")
    try:
        from graphgen.config.taxonomy import get_default_taxonomy
        tax = get_default_taxonomy()
        ok(f"Loaded taxonomy: name={tax.name!r}  version={tax.version!r}")
        ok(f"Entity groups: {list(tax.entity_groups.keys())}")
        ok(f"Relation types: {tax.relation_types[:5]}…")
        ok(f"Overrides: {dict(list(tax.overrides.items())[:3])}…")

        # Test classify_name with taxonomy (PERSON needs a title token prefix, e.g. PRESIDENT_)
        from graphgen.patch.ontology_classifier import classify_name
        cases = [
            ("EUROPEAN_COMMISSION", "ORGANIZATION"),    # COMMISSION token → ORGANIZATION
            ("PRESIDENT_MACRON", "PERSON"),             # PRESIDENT token → PERSON
            ("CLIMATE_POLICY", "POLICY"),               # POLICY token → POLICY
            ("URSULA_VON_DER_LEYEN", "CONCEPT"),        # no title token → falls through to CONCEPT
        ]
        for name, expected in cases:
            result = classify_name(name, taxonomy=tax)
            status = "✓" if result == expected else "⚠"
            print(f"  {status} classify_name({name!r}, taxonomy) → {result!r}  (expected {expected!r})")
    except Exception as exc:
        warn(f"E4 taxonomy demo failed: {exc}")

    # Show patch preview (dry-run, no Neo4j connection needed for import)
    try:
        from graphgen.patch.runner import DEFAULT_PHASE_ORDER, PHASE_RUNNERS
        ok(f"Patch phase order: {DEFAULT_PHASE_ORDER}")
        ok(f"Registered runners: {list(PHASE_RUNNERS.keys())}")
        assert "gliner2_fallback" in PHASE_RUNNERS
        ok("'gliner2_fallback' phase is registered.")
    except Exception as exc:
        warn(f"E4 patch runner check failed: {exc}")


# ---------------------------------------------------------------------------
# E5 — Session temporal merge preview
# ---------------------------------------------------------------------------

async def demo_e5() -> None:
    header("E5 · Session temporal merge — ingested_at / contributing_session_id / TTL")

    graphknows_url = os.environ.get("GRAPHKNOWS_URL", "http://localhost:8025")

    try:
        import httpx
        async with httpx.AsyncClient(timeout=5.0) as client:
            # Create a throwaway session
            r_sess = await client.post(
                f"{graphknows_url}/session",
                json={"seed_entities": []},
            )
            r_sess.raise_for_status()
            session_id = r_sess.json()["session_id"]
            ok(f"Created session: {session_id}")

            # Preview merge (nothing in Memgraph yet — should return empty preview)
            r_prev = await client.get(f"{graphknows_url}/session/{session_id}/preview-merge")
            r_prev.raise_for_status()
            preview = r_prev.json()
            section("merge preview", str(preview))
            ok("GET /session/{id}/preview-merge is live.")

            # Clean up
            await client.delete(f"{graphknows_url}/session/{session_id}")
            ok(f"Session {session_id} deleted.")
    except Exception as exc:
        warn(f"E5 live demo skipped (graphknows not reachable at {graphknows_url}): {exc}")

    # Verify temporal fields exist in the merge Cypher (code inspection)
    try:
        import inspect
        from graphknows.stores.neo4j import Neo4jStore
        source = inspect.getsource(Neo4jStore.merge_session_entities_to_corpus)
        for field in ("ingested_at", "contributing_session_id", "merged_session_count"):
            assert field in source, f"Missing field {field!r} in merge Cypher"
            ok(f"Field {field!r} present in merge_session_entities_to_corpus Cypher.")
    except Exception as exc:
        warn(f"E5 code inspection failed: {exc}")


# ---------------------------------------------------------------------------
# E6 — SEGMENT temporal anchor above DOC
# ---------------------------------------------------------------------------

async def demo_e6() -> None:
    header("E6 · SEGMENT as temporal anchor above DOC")

    # Schema check
    try:
        from graphgen.config.schema import get_default_schema
        schema = get_default_schema()
        seg = schema.nodes.get("Segment")
        pas = schema.nodes.get("Passage")
        ok(f"Schema 'Segment' node → label={seg.label!r}  source_type={seg.source_type!r}")
        ok(f"Schema 'Passage' node → label={pas.label!r}  source_type={pas.source_type!r}")
        edge_types = {e.relation_type for e in schema.edges}
        assert "CONTAINS_DOC" in edge_types, "CONTAINS_DOC edge missing"
        assert "HAS_PASSAGE" in edge_types, "HAS_PASSAGE edge missing"
        ok("CONTAINS_DOC and HAS_PASSAGE edges confirmed in schema.")
    except Exception as exc:
        warn(f"E6 schema check failed: {exc}")

    # Builder: create_temporal_segment_node
    try:
        import networkx as nx
        from graphgen.pipeline.lexical_graph_building.builder import create_temporal_segment_node
        g = nx.DiGraph()
        nid = create_temporal_segment_node(g, run_id="test-run-001", ingested_at="2025-01-01T00:00:00+00:00")
        node = g.nodes[nid]
        ok(f"create_temporal_segment_node() → node_id={nid!r}  node_type={node['node_type']!r}")
        assert node["node_type"] == "SEGMENT"
        ok("SEGMENT node correctly typed.")
    except Exception as exc:
        warn(f"E6 builder check failed: {exc}")

    # Live segment list
    graphknows_url = os.environ.get("GRAPHKNOWS_URL", "http://localhost:8025")
    try:
        import httpx
        async with httpx.AsyncClient(timeout=5.0) as client:
            r = await client.get(f"{graphknows_url}/segments")
            r.raise_for_status()
            segments = r.json()
            ok(f"GET /segments → {len(segments)} SEGMENT node(s) found.")
            for seg in segments[:3]:
                print(f"    {seg.get('segment_id')}  {seg.get('ingested_at')}")
    except Exception as exc:
        warn(f"E6 live segments skipped ({graphknows_url} not reachable): {exc}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main() -> None:
    print()
    print("graphknows · Edge Demos: E3, E4, E5, E6")
    print("=" * 72)

    demo_e3()
    demo_e4()
    await demo_e5()
    await demo_e6()

    print(f"\n{SEP}")
    print("  Demo complete.")
    print(SEP)


if __name__ == "__main__":
    asyncio.run(main())

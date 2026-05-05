#!/usr/bin/env python3
"""LLM-based relation extraction over existing Chunks → RELATION edges in Neo4j.

Reads all chunks from pgvector, calls the Groq LLM (llama-3.3-70b-versatile) to
extract (subject, predicate, object) triplets, then stores them as:

    (Entity {name: subject})-[:RELATION {predicate, chunk_id, source}]->(Entity {name: object})

Usage::

    PYTHONPATH=/root/graphknows python3 scripts/extract_relations.py [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path

import psycopg2
from dotenv import dotenv_values
from neo4j import GraphDatabase
from openai import OpenAI

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s – %(message)s")
log = logging.getLogger(__name__)

_REPO = Path(__file__).parent.parent
_ENV = dotenv_values(str(_REPO / ".env"))

# ── Config ─────────────────────────────────────────────────────────────────────
NEO4J_URI = "bolt://172.18.0.1:7687"
NEO4J_AUTH = ("neo4j", _ENV.get("NEO4J_PASSWORD", "testpassword123"))

PG_DSN = dict(
    host=_ENV.get("PGVECTOR_HOST", "172.18.0.1"),
    port=int(_ENV.get("PGVECTOR_PORT", 5432)),
    user=_ENV.get("PGVECTOR_USER", "postgres"),
    password=_ENV.get("PGVECTOR_PASSWORD", "pgvectorpass123"),
    dbname=_ENV.get("PGVECTOR_DB", "vectordb"),
)

GROQ_API_KEY = _ENV.get("GROQ_API_KEY") or _ENV.get("LLM_GROQ_API_KEY", "")
LLM_MODEL = _ENV.get("LLM_MODEL", "llama-3.3-70b-versatile")

EXTRACT_PROMPT = """\
You are a political-text information-extraction engine.
Extract relation triplets from the political text below.

Rules:
- Return ONLY a valid JSON array — no prose, no markdown fences.
- Each element: {{"subject": str, "predicate": str, "object": str}}
- Predicates must be UPPER_SNAKE_CASE active verb phrases, e.g.
  SUPPORTS, OPPOSES, IMPOSES_SANCTIONS_ON, NEGOTIATES_WITH, CALLS_FOR, CONDEMNS.
- Both subject and object must be named entities (persons, countries, organisations,
  institutions, agreements, locations). Skip generic noun phrases.
- Max 20 triplets. If none found, return [].

Text:
{text}
"""


def _make_llm_client() -> OpenAI:
    """Instantiate an OpenAI-compatible client pointed at Groq."""
    return OpenAI(
        api_key=GROQ_API_KEY or os.environ.get("GROQ_API_KEY", "no-key"),
        base_url="https://api.groq.com/openai/v1",
    )


def extract_triplets(client: OpenAI, chunk_id: str, text: str) -> list[dict]:
    """Call the LLM and parse returned JSON triplets.

    Args:
        client: Configured OpenAI-compatible client.
        chunk_id: Identifier of the originating chunk (for logging).
        text: Raw chunk text to analyse.

    Returns:
        List of dicts with keys ``subject``, ``predicate``, ``object``.
    """
    prompt = EXTRACT_PROMPT.format(text=text[:2000])
    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=1024,
        )
    except Exception as exc:
        log.warning("LLM call failed for chunk %s: %s", chunk_id, exc)
        return []

    content = response.choices[0].message.content or ""
    content = content.strip()

    # Strip optional markdown fences
    content = re.sub(r"^```[a-z]*\n?", "", content)
    content = re.sub(r"\n?```$", "", content)

    m = re.search(r"\[.*\]", content, re.DOTALL)
    if not m:
        log.debug("No JSON array found in LLM response for chunk %s", chunk_id)
        return []
    try:
        triplets = json.loads(m.group(0))
    except json.JSONDecodeError as exc:
        log.warning("JSON parse error for chunk %s: %s", chunk_id, exc)
        return []

    valid = []
    for t in triplets:
        if (
            isinstance(t, dict)
            and isinstance(t.get("subject"), str)
            and isinstance(t.get("predicate"), str)
            and isinstance(t.get("object"), str)
            and t["subject"].strip()
            and t["predicate"].strip()
            and t["object"].strip()
        ):
            valid.append(
                {
                    "subject": t["subject"].strip(),
                    "predicate": t["predicate"].strip().upper(),
                    "object": t["object"].strip(),
                }
            )
    return valid


def store_triplets(
    session,
    triplets: list[dict],
    chunk_id: str,
    source_doc: str = "",
) -> int:
    """Merge triplets into Neo4j as RELATION edges.

    Args:
        session: Active Neo4j session.
        triplets: List of dicts with subject/predicate/object and optional
            confidence, evidence, source_doc keys.
        chunk_id: Source chunk id for edge metadata.
        source_doc: Optional source document identifier.

    Returns:
        Number of edges created or merged.
    """
    count = 0
    for t in triplets:
        session.run(
            """
            MERGE (a:Entity {name: $subject})
            MERGE (b:Entity {name: $object})
            MERGE (a)-[r:RELATION {
                predicate: $predicate,
                chunk_id: $chunk_id
            }]->(b)
            SET r.source = 'llm_extraction',
                r.confidence = $confidence,
                r.evidence = $evidence,
                r.source_doc = $source_doc
            """,
            subject=t["subject"],
            predicate=t["predicate"],
            object=t["object"],
            chunk_id=chunk_id,
            confidence=t.get("confidence"),
            evidence=t.get("evidence", ""),
            source_doc=t.get("source_doc") or source_doc,
        )
        count += 1
    return count


def run(dry_run: bool = False) -> None:
    """Main entry point — iterate chunks, extract relations, persist to Neo4j.

    Args:
        dry_run: If True, log triplets but do not write to Neo4j.
    """
    # ── Fetch all chunks from pgvector ─────────────────────────────────────────
    conn = psycopg2.connect(**PG_DSN)
    try:
        cur = conn.cursor()
        cur.execute("SELECT id, doc_id, text FROM chunks ORDER BY id")
        rows = cur.fetchall()
    finally:
        conn.close()

    log.info("Found %d chunks to process", len(rows))

    client = _make_llm_client()
    driver = GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH)

    total_triplets = 0
    total_edges = 0

    try:
        with driver.session() as session:
            for chunk_id, doc_id, text in rows:
                if not text or not text.strip():
                    log.debug("Skipping empty chunk %s", chunk_id)
                    continue

                log.info("Processing chunk %s (doc=%s, len=%d chars)", chunk_id, doc_id, len(text))
                triplets = extract_triplets(client, chunk_id, text)
                log.info("  → %d triplets extracted", len(triplets))
                total_triplets += len(triplets)

                if dry_run:
                    for t in triplets:
                        log.info("  [DRY] %s -[%s]-> %s", t["subject"], t["predicate"], t["object"])
                else:
                    edges = store_triplets(session, triplets, chunk_id)
                    total_edges += edges

            # Verification
            if not dry_run:
                rel_count = session.run(
                    "MATCH ()-[r:RELATION]->() RETURN count(r) AS n"
                ).single()["n"]
                log.info("Total RELATION edges in Neo4j: %d", rel_count)
    finally:
        driver.close()

    log.info(
        "Done. Triplets extracted: %d | Edges stored: %d%s",
        total_triplets,
        total_edges,
        " (dry-run, nothing stored)" if dry_run else "",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM relation extraction → Neo4j RELATION edges")
    parser.add_argument("--dry-run", action="store_true", help="Extract but do not write to Neo4j")
    args = parser.parse_args()
    run(dry_run=args.dry_run)

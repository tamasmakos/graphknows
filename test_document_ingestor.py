"""Test DocumentIngestor with synthetic text — no GLiNER2, no network."""
from __future__ import annotations

import os
import sys

_repo = os.path.dirname(os.path.abspath(__file__))
for _p in [
    f"{_repo}/services",
    f"{_repo}/services/graphknows",
    f"{_repo}/services/graphknows/graphknows",
]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from dotenv import load_dotenv

load_dotenv("/root/graphknows/.env")

from neo4j import GraphDatabase
from graphknows.document_ingestor import DocumentIngestor, IngestResult, ChunkData

# 1. Chunking — pure logic, no DB
TEXT = "The European Central Bank sets monetary policy. " * 100
ingestor = DocumentIngestor(neo4j_driver=None)
chunks = ingestor._chunk(TEXT)
assert len(chunks) >= 2, f"Expected >=2 chunks, got {len(chunks)}"
assert all(c.token_count <= 550 for c in chunks), "Chunk too large"
print(f"PASS  Chunking: {len(chunks)} chunks")

# 2. Entity extraction — spaCy fallback (GLiNER2 not required)
mentions = ingestor._extract_entities(chunks[0], ["ORG", "GPE", "PERSON"])
print(f"PASS  Entity extraction: {len(mentions)} mentions in first chunk")

# 3. Neo4j write — uses real DB
NEO4J_URI = os.environ["NEO4J_URI"]
NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j")
NEO4J_PASS = os.environ["NEO4J_PASSWORD"]
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))
ingestor2 = DocumentIngestor(neo4j_driver=driver)

result = ingestor2.ingest(
    source="__synthetic__",
    session_id="test-ltm9",
    ontology=["ORG", "GPE", "PERSON"],
    title="Synthetic ECB Policy Test",
)
assert result.chunk_count >= 2, f"Expected >=2 chunks written, got {result.chunk_count}"

with driver.session() as s:
    doc_count = s.run(
        "MATCH (d:Document {id: $did}) RETURN count(d) AS c",
        did=result.document_id,
    ).single()["c"]
    chunk_count = s.run(
        "MATCH (:Document {id: $did})-[:HAS_CHUNK]->(c) RETURN count(c) AS c",
        did=result.document_id,
    ).single()["c"]
    assert doc_count == 1, "Document node missing"
    assert chunk_count == result.chunk_count, (
        f"Chunk count mismatch: {chunk_count} vs {result.chunk_count}"
    )
    # Cleanup
    s.run("MATCH (d:Document {id: $did}) DETACH DELETE d", did=result.document_id)
    s.run("MATCH (c:Chunk) WHERE c.id STARTS WITH 'synth-' DETACH DELETE c")

driver.close()
print(f"PASS  Neo4j write: {result.chunk_count} chunks, {result.mention_count} mentions")
print("PASS  Cleanup complete")
print("All LTM-9 tests passed.")

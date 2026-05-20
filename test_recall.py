"""Test recall_by_document, recall_by_dialog, recall_by_topic with synthetic fixtures."""
from __future__ import annotations

import os
import sys

_repo = os.path.dirname(os.path.abspath(__file__))
_services = os.path.join(_repo, "services")
_graphknows = os.path.join(_repo, "services", "graphknows")
_graphknows_inner = os.path.join(_repo, "services", "graphknows", "graphknows")
for _p in [_services, _graphknows, _graphknows_inner]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from dotenv import load_dotenv

load_dotenv(os.path.join(_repo, ".env"))

from neo4j import GraphDatabase

from graphknows.recall import recall_by_document, recall_by_dialog, recall_by_topic, RecallResult

NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://172.18.0.1:7687")
NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "testpassword123")

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

SESSION = "test-recall-ltm11"

with driver.session() as s:
    s.run("MERGE (e:Entity {name: 'European Central Bank'}) SET e.type = 'INSTITUTION'")
    s.run("""
        MERGE (d:Document {id: 'test-doc-recall'})
        SET d.title = 'ECB Policy', d.url = 'http://test.ecb.eu'
        MERGE (c:Chunk {id: 'test-chunk-recall'})
        SET c.text = 'The European Central Bank sets monetary policy for the eurozone.',
            c.char_offset = 0, c.token_count = 14
        MERGE (d)-[:HAS_CHUNK]->(c)
        WITH c
        MATCH (e:Entity {name: 'European Central Bank'})
        MERGE (c)-[:MENTIONS]->(e)
    """)
    s.run("""
        MERGE (seg:Segment {session_id: $sid})
        MERGE (dia:Dialog {id: 'test-dia-recall'}) SET dia.session_id = $sid
        MERGE (seg)-[:HAS_DIALOG]->(dia)
        MERGE (p:Participant {id: 'test-p-recall'}) SET p.role = 'user', p.name = 'user'
        MERGE (dia)-[:PARTICIPANT]->(p)
        MERGE (u:Utterance {id: 'test-utt-recall'})
        SET u.text = 'What does the European Central Bank do?', u.turn_index = 0,
            u.timestamp = '2025-01-01T10:00:00Z'
        MERGE (p)-[:SAID]->(u)
        WITH u
        MATCH (e:Entity {name: 'European Central Bank'})
        MERGE (u)-[:MENTIONS]->(e)
    """, sid=SESSION)
    s.run("""
        MERGE (t:Topic {community_id: 99999})
        SET t.label = 'Monetary Policy', t.entity_count = 1,
            t.top_entities = ['European Central Bank']
        WITH t
        MATCH (e:Entity {name: 'European Central Bank'})
        MERGE (e)-[:BELONGS_TO]->(t)
    """)

doc_results = recall_by_document(driver, "European Central Bank", top_k=5)
assert len(doc_results) >= 1, f"Expected >=1 doc result, got {len(doc_results)}"
assert any("European Central Bank" in r.entity_names for r in doc_results)
print(f"PASS  recall_by_document: {len(doc_results)} results")

dial_results = recall_by_dialog(driver, "European Central Bank", top_k=5)
assert len(dial_results) >= 1, f"Expected >=1 dialog result, got {len(dial_results)}"
print(f"PASS  recall_by_dialog: {len(dial_results)} results")

topic_results = recall_by_topic(driver, "monetary policy", top_k=5)
assert len(topic_results) >= 1, f"Expected >=1 topic result, got {len(topic_results)}"
print(f"PASS  recall_by_topic: {len(topic_results)} results")

with driver.session() as s:
    s.run("MATCH (d:Document {id: 'test-doc-recall'}) DETACH DELETE d")
    s.run("MATCH (c:Chunk {id: 'test-chunk-recall'}) DETACH DELETE c")
    s.run("MATCH (dia:Dialog {id: 'test-dia-recall'}) DETACH DELETE dia")
    s.run("MATCH (u:Utterance {id: 'test-utt-recall'}) DETACH DELETE u")
    s.run("MATCH (p:Participant {id: 'test-p-recall'}) DETACH DELETE p")
    s.run("MATCH (seg:Segment {session_id: $sid}) DETACH DELETE seg", sid=SESSION)
    s.run("MATCH (t:Topic {community_id: 99999}) DETACH DELETE t")

driver.close()
print("PASS  Cleanup complete")
print("All LTM-11 recall tests passed.")

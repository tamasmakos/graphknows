"""Test DialogWriteBack with synthetic turns — no LangGraph, no GPU."""
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
from agent.session.dialog import DialogWriteBack, TurnData

NEO4J_URI = os.environ["NEO4J_URI"]
driver = GraphDatabase.driver(NEO4J_URI, auth=(os.environ.get("NEO4J_USER", "neo4j"), os.environ["NEO4J_PASSWORD"]))

SESSION_ID = "test-dialog-ltm10"

with driver.session() as s:
    s.run("MERGE (seg:Segment {session_id: $sid})", sid=SESSION_ID)

turns = [
    TurnData(0, "user", "user", "What is the EU AI Act?", "2025-01-01T10:00:00Z"),
    TurnData(1, "agent", "GraphKnows-Agent", "The EU AI Act is a regulation...", "2025-01-01T10:00:05Z"),
    TurnData(2, "user", "user", "Who proposed it?", "2025-01-01T10:00:10Z"),
    TurnData(3, "agent", "GraphKnows-Agent", "The European Commission proposed it in 2021.", "2025-01-01T10:00:15Z"),
]

wb = DialogWriteBack(neo4j_driver=driver)
result = wb.write(session_id=SESSION_ID, turns=turns)

assert result.utterance_count == 4, f"Expected 4 utterances, got {result.utterance_count}"
assert not result.errors, f"Errors: {result.errors}"

with driver.session() as s:
    dial_count = s.run("MATCH (d:Dialog {session_id: $sid}) RETURN count(d) AS c", sid=SESSION_ID).single()["c"]
    utt_count = s.run(
        "MATCH (:Segment {session_id: $sid})-[:HAS_DIALOG]->(:Dialog)-[:PARTICIPANT]->(:Participant)-[:SAID]->(u:Utterance) RETURN count(u) AS c",
        sid=SESSION_ID,
    ).single()["c"]
    assert dial_count == 1, f"Expected 1 Dialog, got {dial_count}"
    assert utt_count == 4, f"Expected 4 Utterances via path, got {utt_count}"
    s.run("MATCH (seg:Segment {session_id: $sid}) DETACH DELETE seg", sid=SESSION_ID)
    s.run("MATCH (d:Dialog {session_id: $sid}) DETACH DELETE d", sid=SESSION_ID)

driver.close()
print("PASS  Dialog node created")
print("PASS  4 Utterances written via full path")
print("PASS  Cleanup complete")
print("All LTM-10 dialog writeback tests passed.")

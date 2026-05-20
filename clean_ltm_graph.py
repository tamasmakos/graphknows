"""Clean stale LTM graph data from Neo4j — removes Topic/SubTopic/GDSRun nodes,
resets community_id on entities. Preserves Entity nodes, Segment nodes, edges, embeddings.
"""
import argparse
import os
from dotenv import load_dotenv

load_dotenv('/root/graphknows/.env')

from neo4j import GraphDatabase

parser = argparse.ArgumentParser(description="Clean LTM graph data from Neo4j")
parser.add_argument(
    "--wipe-documents",
    action="store_true",
    help="Also wipe Document, Chunk, Dialog, Utterance nodes (full reset)",
)
args = parser.parse_args()

uri = os.environ.get('NEO4J_URI', 'bolt://172.18.0.1:7687')
user = os.environ.get('NEO4J_USER', 'neo4j')
pw = os.environ['NEO4J_PASSWORD']

d = GraphDatabase.driver(uri, auth=(user, pw))

with d.session() as s:
    # Count before
    ec_before = s.run("MATCH (e:Entity) RETURN count(e) as c").single()['c']
    tc_before = s.run("MATCH (t:Topic) RETURN count(t) as c").single()['c']
    stc_before = s.run("MATCH (t:SubTopic) RETURN count(t) as c").single()['c']
    rc_before = s.run("MATCH (r:GDSRun) RETURN count(r) as c").single()['c']
    seg_before = s.run("MATCH (s:Segment) RETURN count(s) as c").single()['c']

    doc_before = s.run("MATCH (d:Document) RETURN count(d) as c").single()['c']
    chunk_before = s.run("MATCH (c:Chunk) RETURN count(c) as c").single()['c']
    dialog_before = s.run("MATCH (d:Dialog) RETURN count(d) as c").single()['c']
    utt_before = s.run("MATCH (u:Utterance) RETURN count(u) as c").single()['c']

    print(f"Before cleanup:")
    print(f"  Entities: {ec_before}")
    print(f"  Topics: {tc_before}")
    print(f"  SubTopics: {stc_before}")
    print(f"  GDSRuns: {rc_before}")
    print(f"  Segments: {seg_before}")
    print(f"  Documents: {doc_before}")
    print(f"  Chunks: {chunk_before}")
    print(f"  Dialogs: {dialog_before}")
    print(f"  Utterances: {utt_before}")

    # Delete Topic, SubTopic, GDSRun nodes
    s.run("MATCH (t:Topic) DETACH DELETE t")
    s.run("MATCH (t:SubTopic) DETACH DELETE t")
    s.run("MATCH (r:GDSRun) DELETE r")

    # Remove community IDs from entities (so Leiden writes fresh IDs)
    s.run("MATCH (e:Entity) REMOVE e.community_id")
    s.run("MATCH (e:Entity) REMOVE e.subcommunity_id")

    # Optionally wipe Document+Chunk+Dialog+Utterance (full reset)
    if args.wipe_documents:
        s.run("MATCH (u:Utterance) DETACH DELETE u")
        s.run("MATCH (p:Participant) DETACH DELETE p")
        s.run("MATCH (d:Dialog) DETACH DELETE d")
        s.run("MATCH (c:Chunk) DETACH DELETE c")
        s.run("MATCH (d:Document) DETACH DELETE d")

    # Count after
    ec = s.run("MATCH (e:Entity) RETURN count(e) as c").single()['c']
    tc = s.run("MATCH (t:Topic) RETURN count(t) as c").single()['c']
    stc = s.run("MATCH (t:SubTopic) RETURN count(t) as c").single()['c']
    rc = s.run("MATCH (r:GDSRun) RETURN count(r) as c").single()['c']
    seg = s.run("MATCH (s:Segment) RETURN count(s) as c").single()['c']

    doc = s.run("MATCH (d:Document) RETURN count(d) as c").single()['c']
    chunk = s.run("MATCH (c:Chunk) RETURN count(c) as c").single()['c']
    dialog = s.run("MATCH (d:Dialog) RETURN count(d) as c").single()['c']
    utt = s.run("MATCH (u:Utterance) RETURN count(u) as c").single()['c']

    print(f"\nAfter cleanup:")
    print(f"  Entities: {ec} (preserved)")
    print(f"  Topics: {tc} (should be 0)")
    print(f"  SubTopics: {stc} (should be 0)")
    print(f"  GDSRuns: {rc} (should be 0)")
    print(f"  Segments: {seg} (preserved)")
    print(f"  Documents: {doc} (preserved without --wipe-documents)")
    print(f"  Chunks: {chunk} (preserved without --wipe-documents)")
    print(f"  Dialogs: {dialog} (preserved without --wipe-documents)")
    print(f"  Utterances: {utt} (preserved without --wipe-documents)")

d.close()
print("\nCleanup complete.")

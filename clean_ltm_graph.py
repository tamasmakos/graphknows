"""Clean stale LTM graph data from Neo4j — removes Topic/SubTopic/GDSRun nodes,
resets community_id on entities. Preserves Entity nodes, Segment nodes, edges, embeddings.
"""
import os
from dotenv import load_dotenv

load_dotenv('/root/graphknows/.env')

from neo4j import GraphDatabase

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

    print(f"Before cleanup:")
    print(f"  Entities: {ec_before}")
    print(f"  Topics: {tc_before}")
    print(f"  SubTopics: {stc_before}")
    print(f"  GDSRuns: {rc_before}")
    print(f"  Segments: {seg_before}")

    # Delete Topic, SubTopic, GDSRun nodes
    s.run("MATCH (t:Topic) DETACH DELETE t")
    s.run("MATCH (t:SubTopic) DETACH DELETE t")
    s.run("MATCH (r:GDSRun) DELETE r")

    # Remove community IDs from entities (so Leiden writes fresh IDs)
    s.run("MATCH (e:Entity) REMOVE e.community_id")
    s.run("MATCH (e:Entity) REMOVE e.subcommunity_id")

    # Count after
    ec = s.run("MATCH (e:Entity) RETURN count(e) as c").single()['c']
    tc = s.run("MATCH (t:Topic) RETURN count(t) as c").single()['c']
    stc = s.run("MATCH (t:SubTopic) RETURN count(t) as c").single()['c']
    rc = s.run("MATCH (r:GDSRun) RETURN count(r) as c").single()['c']
    seg = s.run("MATCH (s:Segment) RETURN count(s) as c").single()['c']

    print(f"\nAfter cleanup:")
    print(f"  Entities: {ec} (preserved)")
    print(f"  Topics: {tc} (should be 0)")
    print(f"  SubTopics: {stc} (should be 0)")
    print(f"  GDSRuns: {rc} (should be 0)")
    print(f"  Segments: {seg} (preserved)")

d.close()
print("\nCleanup complete.")

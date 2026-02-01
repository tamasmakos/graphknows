import sys
import os
sys.path.append("/app")
from src.infrastructure.graph_db import get_database_client 
from src.common.config.settings import AppSettings

def debug_edge():
    settings = AppSettings()
    db = get_database_client(settings)
    
    # 1. Get a Topic
    res = db.query("MATCH (t:TOPIC) RETURN t.id LIMIT 1")
    if not res:
        print("No TOPIC found!")
        return
    topic_id = res[0]['t.id']
    print(f"Using Topic: {topic_id}")
    
    # 2. Get an Entity
    res = db.query("MATCH (e:ENTITY_CONCEPT) RETURN e.id LIMIT 1")
    if not res:
        print("No Entity found!")
        return
    entity_id = res[0]['e.id']
    print(f"Using Entity: {entity_id}")
    
    # 3. Try to create edge manually using the EXACT logic from uploader
    # Uploader uses UNWIND with map
    
    batch = [{
        'source_id': entity_id,
        'target_id': topic_id,
        'properties': {'test': True}
    }]
    
    escaped_rel_type = "TEST_IN_TOPIC"
    
    cypher = f"""
    UNWIND $batch AS rel
    MATCH (source) WHERE source.id = rel.source_id
    MATCH (target) WHERE target.id = rel.target_id
    MERGE (source)-[r:{escaped_rel_type}]->(target)
    SET r = rel.properties
    RETURN count(r) as created
    """
    
    print("Executing MERGE query...")
    res = db.query(cypher, {'batch': batch})
    print(f"Result: {res}")
    
    # 4. Check if it exists
    check = f"MATCH (s)-[:{escaped_rel_type}]->(t) WHERE s.id='{entity_id}' AND t.id='{topic_id}' RETURN count(*) as c"
    res = db.query(check)
    print(f"Verification count: {res[0]['c']}")

if __name__ == "__main__":
    debug_edge()

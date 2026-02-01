import sys
import os
# Add /app to path to allow imports if running in container
sys.path.append("/app")
from src.infrastructure.graph_db import get_database_client 
from src.common.config.settings import AppSettings

def check_ids():
    settings = AppSettings()
    db = get_database_client(settings)
    
    print("--- TOPIC IDs ---")
    query = "MATCH (n:TOPIC) RETURN n.id LIMIT 5"
    result = db.query(query)
    for row in result:
        print(row)
        
    print("\n--- ENTITY_CONCEPT IDs ---")
    query = "MATCH (n:ENTITY_CONCEPT) RETURN n.id LIMIT 5"
    result = db.query(query)
    for row in result:
        print(row)

    print("\n--- Edge Count Total ---")
    query = "MATCH ()-[r]->() RETURN count(r) as count"
    result = db.query(query)
    print(result)

    print("\n--- IN_TOPIC Edge Count ---")
    query = "MATCH ()-[r:IN_TOPIC]->() RETURN count(r) as count"
    result = db.query(query)
    print(result)

if __name__ == "__main__":
    check_ids()

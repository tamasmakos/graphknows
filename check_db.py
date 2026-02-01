import asyncio
import os
import sys
from src.infrastructure.graph_db import get_database_client 
from src.common.config.settings import AppSettings

# Hack to get imports working
sys.path.append(os.getcwd())

def check_edges():
    settings = AppSettings()
    db = get_database_client(settings)
    
    print("Checking for IN_TOPIC edges...")
    query = "MATCH ()-[r:IN_TOPIC]->() RETURN count(r) as count"
    result = db.query(query)
    print(f"IN_TOPIC count: {result[0]['count']}")
    
    print("Checking for PARENT_TOPIC edges...")
    query = "MATCH ()-[r:PARENT_TOPIC]->() RETURN count(r) as count"
    result = db.query(query)
    print(f"PARENT_TOPIC count: {result[0]['count']}")
    
    print("Checking for Topic nodes...")
    query = "MATCH (n:TOPIC) RETURN count(n) as count"
    result = db.query(query)
    print(f"TOPIC count: {result[0]['count']}")

    print("Checking for Topic embeddings (FalkorDB)...")
    query = "MATCH (n:TOPIC) WHERE n.embedding IS NOT NULL RETURN count(n) as count"
    result = db.query(query)
    print(f"TOPIC with vector (Falkor) count: {result[0]['count']}")

    print("Checking for Topic embeddings (Postgres ID)...")
    query = "MATCH (n:TOPIC) WHERE n.pg_embedding_id IS NOT NULL RETURN count(n) as count"
    result = db.query(query)
    print(f"TOPIC with pg_id count: {result[0]['count']}")

if __name__ == "__main__":
    check_edges()

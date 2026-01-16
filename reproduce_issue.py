
import asyncio
import os
import logging
from src.app.infrastructure.graph_db import GraphDB
from src.app.services.graph_retriever import expand_subgraph

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def main():
    # Setup DB connection
    # Assuming env vars are set or defaults work. 
    # The user logs show host.docker.internal, but from inside the container it might be specific.
    # However, since the app is running in the same workspace, maybe we can just instantiate GraphDB.
    
    # We need to know specific config. 
    # Let's try to load config or manual setup.
    
    db_config = {
        "postgres": {
            "host": os.getenv("POSTGRES_HOST", "host.docker.internal"),
            "port": int(os.getenv("POSTGRES_PORT", 5432)),
            "user": os.getenv("POSTGRES_USER", "postgres"),
            "password": os.getenv("POSTGRES_PASSWORD", "password"),
            "database": os.getenv("POSTGRES_DB", "graphknows"),
            "schema": "public"
        },
        "falkordb": {
            "host": os.getenv("FALKORDB_HOST", "host.docker.internal"),
            "port": int(os.getenv("FALKORDB_PORT", 6379)),
            "password": os.getenv("FALKORDB_PASSWORD", "")
        }
    }
    
    print("Connecting to DB...")
    # GraphDB is synchronous? The file I read imported ThreadPoolExecutor, so maybe methods are sync.
    # Yes, methods looked sync.
    
    try:
        # Use FalkorDBDB directly
        from src.app.infrastructure.graph_db import FalkorDBDB
        
        db = FalkorDBDB(
            host=db_config["falkordb"]["host"],
            port=db_config["falkordb"]["port"],
            password=db_config["falkordb"]["password"],
            postgres_config=db_config["postgres"]
        )
        
        # 1. Find a Topic to test with - specifically one that has connected entities via SUBTOPIC
        print("Finding a TOPIC with Subtopic connections...")
        # Note: Entities are labelled ENTITY_CONCEPT
        query = """
        MATCH (t:TOPIC)<-[:PARENT_TOPIC]-(s:SUBTOPIC)<-[:IN_TOPIC]-(e:ENTITY_CONCEPT)
        RETURN t.id as id, t.title as title, count(e) as entity_count
        ORDER BY entity_count DESC
        LIMIT 1
        """
        results = db.query(query, {})
        
        if not results:
            print("No TOPICS with Subtopic->Entity connections found in DB. Trying direct connections...")
            query = """
            MATCH (t:TOPIC)<-[:IN_TOPIC]-(e:ENTITY_CONCEPT)
            RETURN t.id as id, t.title as title, count(e) as entity_count
            ORDER BY entity_count DESC
            LIMIT 1
            """
            results = db.query(query, {})
        
        if not results:
            print("No TOPICS with Subtopic->Entity connections found in DB.")
            # Fallback to check labels
            print("Checking what labels exist connected to TOPICs...")
            res = db.query("MATCH (t:TOPIC)<-[r]-(n) RETURN labels(t) as tl, type(r) as r, labels(n) as nl LIMIT 5", {})
            print(res)
            return

        test_topic_id = results[0]['id']
        test_topic_title = results[0].get('title', 'No Title')
        count = results[0].get('entity_count', 0)
        print(f"Testing with Topic: {test_topic_id} ({test_topic_title}) - Has {count} entities via Subtopic")
        
        # 2. Test expand_subgraph with this topic as seed
        print(f"Expanding subgraph for seed: {test_topic_id}")
        nodes, edges, timings = expand_subgraph(db, [test_topic_id])
        
        print(f"Nodes found: {len(nodes)}")
        print(f"Edges found: {len(edges)}")
        print("Timings:", timings)
        
        # Check if we got any chunks or entities
        has_entities = any("Entity" in n.get("labels", []) or "ENTITY_CONCEPT" in n.get("labels", []) for n in nodes.values())
        has_chunks = any("Chunk" in n.get("labels", []) for n in nodes.values())
        
        print(f"Has Entities: {has_entities}")
        print(f"Has Chunks: {has_chunks}")
        
        if not has_entities and not has_chunks:
            print("FAILURE: Topic seed did not expand to entities or chunks.")
        else:
            print("SUCCESS: Topic seed expanded successfully.")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())

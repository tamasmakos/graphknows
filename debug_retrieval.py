
import asyncio
import os
import sys
# Add source path
sys.path.append(os.getcwd())

from src.app.infrastructure.graph_db import get_database_client
from src.app.infrastructure.config import get_app_config
from src.app.services.graph_retriever import get_seed_entities
from src.app.llama.embeddings import embed_query

async def main():
    config = get_app_config()
    db = get_database_client(config, "falkordb")
    
    query = "family discussions"
    print(f"Testing retrieval for: {query}")
    
    emb = embed_query(query)
    keywords = ["family", "discussions"]
    
    seeds, timings = get_seed_entities(db, emb, keywords)
    print(f"\nSeeds found ({len(seeds)}):")
    for s in seeds:
        print(f" - {s}")
        
    print("\nTimings:")
    print(timings)

if __name__ == "__main__":
    asyncio.run(main())

import sys
import os
import requests
import json
from dotenv import load_dotenv

load_dotenv()

# Service URL
# GraphRAG service in docker-compose is named 'graphrag' and internally uses port 8000.
# If running this script from dev-environment, we use http://graphrag:8000
GRAPHRAG_URL = os.getenv("GRAPHRAG_URL", "http://graphrag:8000")

def run_example():
    print(f"[GraphRAG Client] Connecting to {GRAPHRAG_URL}...")
    
    query = "How could I improve my morning routine?"
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        
    print(f"[GraphRAG Client] Sending query: {query}")
    print("-" * 50)
    
    try:
        response = requests.post(
            f"{GRAPHRAG_URL}/chat",
            json={"query": query},
            timeout=120
        )
        
        if response.status_code == 200:
            result = response.json()
            print("\n" + "="*50)
            print(result.get("answer"))
            
            print("\n" + "="*50)
            print("🧠 REASONING TRACE")
            print("="*50)
            for step_info in result.get("reasoning_chain", []):
                print(f"• {step_info}")
        else:
            print(f"[GraphRAG Client] Error: {response.status_code}")
            print(response.text)
            
    except Exception as e:
        print(f"[GraphRAG Client] ❌ Execution failed: {e}")

if __name__ == "__main__":
    run_example()
import sys
import os
import requests
import json
from dotenv import load_dotenv

load_dotenv()

# Service URL
GRAPHGEN_URL = os.getenv("GRAPHGEN_URL", "http://graphgen:8000")

def test_graphgen():
    print("[GraphGen Client] Triggering pipeline via API...")
    
    try:
        response = requests.post(
            f"{GRAPHGEN_URL}/run", 
            json={"clean_database": True},
            timeout=10
        )
        
        if response.status_code == 200:
            print("[GraphGen Client] Pipeline started successfully!")
            print(f"Response: {response.json()}")
        else:
            print(f"[GraphGen Client] Failed to start pipeline. Code: {response.status_code}")
            print(f"Error: {response.text}")
            
    except Exception as e:
        print(f"[GraphGen Client] Error connecting to service: {e}")
        print("Ensure 'graphgen' service is running and accessible.")

if __name__ == "__main__":
    test_graphgen()
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
            print("-" * 50)
            print("[GraphGen Client] 📡 Streaming server logs to track progress...")
            print("[GraphGen Client] (Press Ctrl+C to stop watching logs - pipeline will continue)")
            print("-" * 50)
            try:
                import subprocess
                # Stream logs from the graphgen container
                subprocess.run(["docker", "compose", "logs", "-f", "graphgen"], check=False)
            except KeyboardInterrupt:
                print("\n[GraphGen Client] Stopped watching logs.")
        else:
            print(f"[GraphGen Client] Failed to start pipeline. Code: {response.status_code}")
            print(f"Error: {response.text}")
            
    except Exception as e:
        print(f"[GraphGen Client] Error connecting to service: {e}")
        print("Ensure 'graphgen' service is running and accessible.")

if __name__ == "__main__":
    test_graphgen()
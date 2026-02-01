import sys
import os
import requests
import json
import argparse
from dotenv import load_dotenv

load_dotenv()

# Service URL
GRAPHGEN_URL = os.getenv("GRAPHGEN_URL", "http://localhost:8020")

def run_simulation(days: int, start_date: str):
    print(f"[GraphGen Client] Triggering Life Simulation for {days} days from {start_date}...")
    
    try:
        response = requests.post(
            f"{GRAPHGEN_URL}/simulate", 
            json={"days": days, "start_date": start_date},
            timeout=60
        )
        
        if response.status_code == 200:
            print("[GraphGen Client] Simulation started successfully!")
            print(f"Response: {response.json()}")
            print("-" * 50)
            print("[GraphGen Client] 📡 Streaming server logs to track progress...")
            print("[GraphGen Client] (Press Ctrl+C to stop watching logs - simulation will continue)")
            print("-" * 50)
            try:
                import subprocess
                # Stream logs from the graphgen container
                subprocess.run(["docker", "compose", "logs", "-f", "graphgen"], check=False)
            except KeyboardInterrupt:
                print("\n[GraphGen Client] Stopped watching logs.")
        else:
            print(f"[GraphGen Client] Failed to start simulation. Code: {response.status_code}")
            print(f"Error: {response.text}")
            
    except Exception as e:
        print(f"[GraphGen Client] Error connecting to service: {e}")
        print("Ensure 'graphgen' service is running and accessible.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run GraphGen Life Simulation")
    parser.add_argument("--days", type=int, default=3, help="Number of days to simulate")
    parser.add_argument("--start-date", type=str, default="2025-01-01", help="Start date YYYY-MM-DD")
    
    args = parser.parse_args()
    
    run_simulation(args.days, args.start_date)
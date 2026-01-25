import subprocess
import sys
import time
import os
import socket
import signal
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent

def kill_existing_processes():
    """Aggressively find and kill existing dashboard processes."""
    print("🧹 Cleaning up existing processes...")
    try:
        result = subprocess.run(["ps", "aux"], capture_output=True, text=True)
        lines = result.stdout.splitlines()
        
        pids_to_kill = []
        for line in lines:
            # Match uvicorn processes for dashboard only
            if "uvicorn" in line and "dashboard.backend.main:app" in line:
                parts = line.split()
                if len(parts) > 1:
                    pids_to_kill.append(parts[1])

        for pid in pids_to_kill:
            try:
                print(f"Killing PID {pid}...")
                os.kill(int(pid), signal.SIGKILL)
            except Exception:
                pass
    except Exception as e:
        print(f"Cleanup warning: {e}")

def wait_for_port(port, timeout=60):
    """Wait for a port to become active."""
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            with socket.create_connection(('127.0.0.1', port), timeout=1):
                return True
        except (ConnectionRefusedError, socket.timeout):
            time.sleep(1)
    return False

def run_process(command, cwd, env=None):
    """Run a process and return the Popen object."""
    print(f"🚀 Starting in {cwd}: {' '.join(command)}")
    return subprocess.Popen(
        command,
        cwd=cwd,
        env=env,
        stdout=sys.stdout,
        stderr=sys.stderr
    )

def main():
    # Clean slate
    kill_existing_processes()
    
    # Check for API Key (Just a warning, since services might have it in docker-compose)
    if not os.getenv("GROQ_API_KEY") and not os.getenv("OPENAI_API_KEY"):
        print("ℹ️  Note: GROQ_API_KEY/OPENAI_API_KEY not found in local env. Ensure services have them.")

    processes = []

    try:
        # 1. Check if External Services are reachable
        print("\n--- Checking Services ---")
        # GraphRAG (Agent) usually at 8000 or 8010 depending on docker-compose. 
        # In the new setup, graphrag is a service.
        # But we are running this launcher LOCALLY (or in dev-container).
        # We need to know where GraphRAG is.
        # Default to localhost:8010 (mapped in docker-compose) or 8000 (internal).
        # If running in dev-container, we might access services via hostname 'graphrag'.
        
        # We'll just start the Dashboard.
        
        # 2. Start Dashboard Backend (Port 8001)
        print("\n--- Starting Dashboard (Port 8001) ---")
        dash_env = os.environ.copy()
        dash_env["PYTHONPATH"] = str(PROJECT_ROOT)
        
        # Pass Service URLs to Dashboard if needed
        # Defaults in code might be http://graphgen:8000
        # If running in dev-container, this is correct.
        
        backend_proc = run_process(
            ["uvicorn", "dashboard.backend.main:app", "--host", "0.0.0.0", "--port", "8001"],
            cwd=PROJECT_ROOT,
            env=dash_env
        )
        processes.append(backend_proc)
        
        # 3. Wait for dashboard
        print("Waiting for Dashboard...")
        if wait_for_port(8001, timeout=20):
            print("✅ Dashboard ready at http://localhost:8001")
        else:
            print("❌ Dashboard failed to start on port 8001")
        
        print("\nPress Ctrl+C to stop.")
        
        # Keep alive
        while True:
            time.sleep(1)
            for p in processes:
                if p.poll() is not None:
                    print(f"⚠️ Process {p.args} exited with code {p.returncode}")
                    raise KeyboardInterrupt

    except KeyboardInterrupt:
        print("\n🛑 Stopping services...")
    finally:
        for p in processes:
            if p.poll() is None:
                p.terminate()
                try:
                    p.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    p.kill()
        print("Services stopped.")

if __name__ == "__main__":
    main()
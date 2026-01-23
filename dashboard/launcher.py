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
    """Aggressively find and kill existing service processes."""
    print("🧹 Cleaning up existing processes...")
    try:
        result = subprocess.run(["ps", "aux"], capture_output=True, text=True)
        lines = result.stdout.splitlines()
        
        pids_to_kill = []
        for line in lines:
            # Match uvicorn processes for our specific apps
            if "uvicorn" in line and ("src.main:app" in line or "dashboard.backend.main:app" in line):
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
    
    # Check for API Key
    if not os.getenv("GROQ_API_KEY") and not os.getenv("OPENAI_API_KEY"):
        print("⚠️  Warning: Neither GROQ_API_KEY nor OPENAI_API_KEY found in environment.")

    processes = []

    try:
        # 1. Start Agent Service (Port 8000)
        print("\n--- Starting Agent Service (Port 8000) ---")
        agent_env = os.environ.copy()
        agent_dir = PROJECT_ROOT / "services" / "graphrag"
        agent_env["PYTHONPATH"] = str(agent_dir)
        
        agent_proc = run_process(
            ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"],
            cwd=agent_dir,
            env=agent_env
        )
        processes.append(agent_proc)
        
        # 2. Start Dashboard Backend (Port 8001)
        print("\n--- Starting Dashboard (Port 8001) ---")
        dash_env = os.environ.copy()
        dash_env["PYTHONPATH"] = str(PROJECT_ROOT)
        
        backend_proc = run_process(
            ["uvicorn", "dashboard.backend.main:app", "--host", "0.0.0.0", "--port", "8001"],
            cwd=PROJECT_ROOT,
            env=dash_env
        )
        processes.append(backend_proc)
        
        # 3. Wait for backends to be ready
        print("Waiting for services to initialize (up to 60s)...")
        
        # Check Dashboard first as it's faster
        if wait_for_port(8001, timeout=20):
            print("✅ Dashboard ready at http://localhost:8001")
        else:
            print("❌ Dashboard failed to start on port 8001")

        # Check Agent (might take longer)
        if wait_for_port(8000, timeout=40):
            print("✅ Agent Service ready at http://localhost:8000")
        else:
            print("⚠️  Agent Service taking longer than expected or failed. Check logs above.")
        
        print("\nPress Ctrl+C to stop all services.")
        
        # Keep alive and monitor
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

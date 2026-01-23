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
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DASHBOARD_DIR = PROJECT_ROOT / "src" / "dashboard"
BACKEND_DIR = DASHBOARD_DIR / "backend"

def kill_existing_processes():
    """Aggressively find and kill existing service processes."""
    print("🧹 Cleaning up existing processes...")
    try:
        # List processes
        result = subprocess.run(["ps", "aux"], capture_output=True, text=True)
        lines = result.stdout.splitlines()
        
        pids_to_kill = []
        for line in lines:
            if "uvicorn" in line and ("src.app.main:app" in line or "src.dashboard.backend.main:app" in line):
                parts = line.split()
                if len(parts) > 1:
                    pids_to_kill.append(parts[1])

        for pid in pids_to_kill:
            try:
                print(f"Killing PID {pid}...")
                os.kill(int(pid), signal.SIGKILL)
            except ProcessLookupError:
                pass
    except Exception as e:
        print(f"Cleanup warning: {e}")

def wait_for_port(port, timeout=30):
    start_time = time.time()
    while time.time() - start_time < timeout:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(('localhost', port)) == 0:
                return True
        time.sleep(0.5)
    return False

def run_process(command, cwd, env=None):
    """Run a process and return the Popen object."""
    print(f"🚀 Starting: {' '.join(command)}")
    return subprocess.Popen(
        command,
        cwd=cwd,
        env=env,
        stdout=sys.stdout,
        stderr=sys.stderr
    )

def main():
    # Environment variables
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT)
    
    # Clean slate
    kill_existing_processes()
    
    # Check for API Key
    if not os.getenv("GROQ_API_KEY") and not os.getenv("OPENAI_API_KEY"):
        print("⚠️  Warning: Neither GROQ_API_KEY nor OPENAI_API_KEY found in environment.")

    processes = []

    try:
        # 1. Start Agent Service (Port 8000)
        print("\n--- Starting Agent Service (Port 8000) ---")
        agent_proc = run_process(
            ["uvicorn", "src.app.main:app", "--host", "0.0.0.0", "--port", "8000"],
            cwd=PROJECT_ROOT,
            env=env
        )
        processes.append(agent_proc)
        
        # 2. Start Dashboard Backend (Port 8001)
        # This now also serves the static frontend
        print("\n--- Starting Dashboard (Port 8001) ---")
        backend_proc = run_process(
            ["uvicorn", "src.dashboard.backend.main:app", "--host", "0.0.0.0", "--port", "8001"],
            cwd=PROJECT_ROOT,
            env=env
        )
        processes.append(backend_proc)
        
        # 3. Wait for backends to be ready
        print("Waiting for services to initialize...")
        if not wait_for_port(8000):
            print("❌ Agent Service failed to start on port 8000")
        if not wait_for_port(8001):
            print("❌ Dashboard failed to start on port 8001")
        
        print("\n✅ All services started!")
        print("   - Agent API: http://localhost:8000")
        print("   - Dashboard: http://localhost:8001")
        print("\nPress Ctrl+C to stop all services.")
        
        # Keep alive
        while True:
            time.sleep(1)
            # Check if any process died
            for p in processes:
                if p.poll() is not None:
                    print(f"⚠️ Process {p.args} exited with code {p.returncode}")
                    # If a critical backend dies, we should probably exit
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
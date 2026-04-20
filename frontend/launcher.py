import subprocess
import sys
import time
import os
import socket
import signal
from pathlib import Path
from dotenv import load_dotenv

try:
    import psutil
except ImportError:
    psutil = None

load_dotenv()

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent

def kill_existing_processes():
    """Aggressively find and kill existing dashboard processes."""
    print("🧹 Cleaning up existing processes...")
    
    if psutil:
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                cmdline = proc.info.get('cmdline')
                if cmdline:
                    cmd_str = " ".join(cmdline)
                    # Match uvicorn processes for dashboard only
                    if "uvicorn" in cmd_str and "dashboard.backend.main:app" in cmd_str:
                        print(f"Killing PID {proc.info['pid']}...")
                        proc.kill()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
    else:
        # Fallback for systems without psutil (though we saw it in pip list)
        if os.name == 'nt':
            # Windows fallback using taskkill if possible, or just skip
            print("⚠️ psutil not found, skipping detailed cleanup.")
        else:
            try:
                result = subprocess.run(["ps", "aux"], capture_output=True, text=True)
                lines = result.stdout.splitlines()
                pids_to_kill = []
                for line in lines:
                    if "uvicorn" in line and "dashboard.backend.main:app" in line:
                        parts = line.split()
                        if len(parts) > 1:
                            pids_to_kill.append(parts[1])
                for pid in pids_to_kill:
                    try:
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
        except (ConnectionRefusedError, socket.timeout, OSError):
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
        print("ℹ️  Note: GROQ_API_KEY/OPENAI_API_KEY not found in local env. Ensure services have them.")

    processes = []

    try:
        # 2. Start Dashboard Backend (Port 8001)
        print("\n--- Starting Dashboard (Port 8001) ---")
        dash_env = os.environ.copy()
        dash_env["PYTHONPATH"] = str(PROJECT_ROOT)
        dash_env["PYTHONUNBUFFERED"] = "1"
        
        # Explicitly set FalkorDB connection for local host access
        # The container exposes 6379 as 6380 on the host
        if "FALKORDB_PORT" not in dash_env:
            dash_env["FALKORDB_PORT"] = "6380"
        if "FALKORDB_HOST" not in dash_env:
            dash_env["FALKORDB_HOST"] = "localhost"
        
        # Default Service URLs to point to Docker-exposed ports if not set
        # This matches the behavior of example scripts connecting to running containers.
        if "GRAPHGEN_URL" not in dash_env:
            dash_env["GRAPHGEN_URL"] = "http://localhost:8020"
            print(f"ℹ️  GRAPHGEN_URL not set, defaulting to {dash_env['GRAPHGEN_URL']}")
        
        if "GRAPHRAG_URL" not in dash_env:
            dash_env["GRAPHRAG_URL"] = "http://localhost:8010"
            print(f"ℹ️  GRAPHRAG_URL not set, defaulting to {dash_env['GRAPHRAG_URL']}")
        
        # Use sys.executable -m uvicorn for better compatibility
        command = [
            sys.executable, "-m", "uvicorn", 
            "dashboard.backend.main:app", 
            "--host", "0.0.0.0", 
            "--port", "8001",
            "--log-level", "info",
            "--access-log"
        ]
        
        backend_proc = run_process(
            command,
            cwd=PROJECT_ROOT,
            env=dash_env
        )
        processes.append(backend_proc)
        
        # 3. Wait for dashboard
        print("Waiting for Dashboard...")
        if wait_for_port(8001, timeout=20):
            print("\n✅ Dashboard ready at http://localhost:8001")
        else:
            print("❌ Dashboard failed to start on port 8001")
            print("💡 Tip: Ensure 'fastapi' and 'uvicorn' are installed: pip install -r requirements.txt")
        
        print("\nPress Ctrl+C to stop.")
        
        # Keep alive
        while True:
            time.sleep(1)
            for p in processes:
                if p.poll() is not None:
                    print(f"⚠️ Process exited with code {p.returncode}")
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

import threading
import time
import signal
import sys


def start_server():
    import uvicorn

    # Need to run from the graphknows root so `apps.web.server` is importable
    uvicorn.run(
        "apps.web.server:app",
        host="0.0.0.0",
        port=8080,
        log_level="info",
    )


def main():
    t = threading.Thread(target=start_server, daemon=True)
    t.start()

    # Give uvicorn a moment to bind
    time.sleep(2)

    tunnel = None
    try:
        from pyngrok import ngrok, conf

        # anonymous tunnel — no auth token required
        tunnel = ngrok.connect(8080, bind_tls=True)
        public_url = tunnel.public_url
        print(f"\n🌐 GRAPHKNOWS PUBLIC URL: {public_url}\n")
    except Exception as e:
        print(f"⚠️  pyngrok failed ({e}), falling back to cloudflared …")
        import subprocess, os

        cf = "/tmp/cloudflared"
        if not os.path.exists(cf):
            subprocess.run(
                "curl -fsSL https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -o /tmp/cloudflared && chmod +x /tmp/cloudflared",
                shell=True,
                check=True,
            )
        proc = subprocess.Popen(
            [cf, "tunnel", "--url", "http://localhost:8080"],
            stderr=subprocess.PIPE,
            text=True,
        )
        # Parse URL from stderr
        import re

        for line in proc.stderr:
            m = re.search(r"https://[a-z0-9\-]+\.trycloudflare\.com", line)
            if m:
                print(f"\n🌐 GRAPHKNOWS PUBLIC URL: {m.group()}\n")
                break

    print("Press Ctrl+C to stop.\n")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down …")
        if tunnel:
            from pyngrok import ngrok

            ngrok.kill()
        sys.exit(0)


if __name__ == "__main__":
    import os, sys

    # Ensure repo root is on path
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    main()

"""
Smoke test runner — executed inside the compose smoke-tests container.

Probes every app service health endpoint and exits non-zero on any failure.
Designed to be used with `docker compose up --exit-code-from smoke-tests`.

Services expected (intra-compose hostnames):
  graphknows :8025  GET /health
  agent      :8000  GET /health
  search     :8000  GET /api/v1/health
  frontend   :8080  GET /
"""

import sys
import time
import urllib.error
import urllib.request

PROBES = [
    ("graphknows", "http://graphknows:8025/health", 200),
    ("agent", "http://agent:8000/health", 200),
    ("search", "http://search:8000/api/v1/health", 200),
    ("frontend", "http://frontend:8080/", 200),
]

MAX_WAIT = 300  # seconds; compose depends_on healthy should have done the work
INTERVAL = 5


def probe(name: str, url: str, expected_status: int) -> bool:
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=5) as resp:
            if resp.status == expected_status:
                print(f"  OK  {name} — {url} → {resp.status}")
                return True
            print(f"  FAIL {name} — {url} → {resp.status} (expected {expected_status})")
            return False
    except Exception as exc:
        print(f"  FAIL {name} — {url} → {exc}")
        return False


def main() -> None:
    print("=== Graphknows smoke tests ===")
    deadline = time.monotonic() + MAX_WAIT
    pending = list(PROBES)

    while pending and time.monotonic() < deadline:
        still_failing = []
        for name, url, status in pending:
            if not probe(name, url, status):
                still_failing.append((name, url, status))
        pending = still_failing
        if pending:
            print(f"  {len(pending)} probe(s) not yet passing, retrying in {INTERVAL}s...")
            time.sleep(INTERVAL)

    if pending:
        print("\nFAILED — the following probes never passed:")
        for name, url, _ in pending:
            print(f"  {name} {url}")
        sys.exit(1)

    print("\nAll probes passed.")
    sys.exit(0)


if __name__ == "__main__":
    main()

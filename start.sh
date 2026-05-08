#!/usr/bin/env bash
# Graphknows service launcher with auto-restart watchdog
# Usage: ./start.sh [--no-tunnel]
set -euo pipefail

cd "$(dirname "$0")"
set -a && source .env && set +a
export LANG=C LC_ALL=C

PYPATH="/root/graphknows/services/agent:/root/graphknows/services:/root/graphknows/services/graphgen"

echo "=== Graphknows startup $(date) ==="

# ── Agent watchdog ──────────────────────────────────────────────────────────
agent_watchdog() {
  while true; do
    echo "[watchdog] Starting agent on :8002..."
    PYTHONPATH="$PYPATH" uvicorn services.agent.main:app \
      --host 0.0.0.0 --port 8002 --log-level info \
      >> /tmp/agent.log 2>&1 || true
    echo "[watchdog] Agent exited — restarting in 5s..."
    sleep 5
  done
}

# ── GraphRAG watchdog ───────────────────────────────────────────────────────
graphrag_watchdog() {
  while true; do
    echo "[watchdog] Starting graphrag on :8001..."
    PYTHONPATH="/root/graphknows/services/graphgen:/root/graphknows/services" \
      uvicorn services.graphrag.api:app \
      --host 0.0.0.0 --port 8001 --log-level info \
      >> /tmp/graphrag.log 2>&1 || true
    echo "[watchdog] GraphRAG exited — restarting in 5s..."
    sleep 5
  done
}

# ── Frontend watchdog ───────────────────────────────────────────────────────
frontend_watchdog() {
  while true; do
    echo "[watchdog] Starting frontend on :8080..."
    AGENT_URL=http://localhost:8002 \
      uvicorn apps.web.server:app \
      --host 0.0.0.0 --port 8080 --log-level info \
      >> /tmp/frontend.log 2>&1 || true
    echo "[watchdog] Frontend exited — restarting in 5s..."
    sleep 5
  done
}

# Kill anything on our ports first
fuser -k 8001/tcp 2>/dev/null || true
fuser -k 8002/tcp 2>/dev/null || true
# Don't kill 8080 — old frontend may still be the right one

agent_watchdog &
AGENT_PID=$!

graphrag_watchdog &
GRAPHRAG_PID=$!

frontend_watchdog &
FRONTEND_PID=$!

echo "Watchdogs started: agent=$AGENT_PID graphrag=$GRAPHRAG_PID frontend=$FRONTEND_PID"

# ── Cloudflare tunnel ───────────────────────────────────────────────────────
if [[ "${1:-}" != "--no-tunnel" ]]; then
  sleep 8  # wait for frontend
  echo "[tunnel] Starting cloudflare tunnel..."
  /tmp/cloudflared tunnel --url http://localhost:8080 > /tmp/cf.log 2>&1 &
  TUNNEL_PID=$!
  sleep 5
  URL=$(grep -o 'https://[a-z0-9-]*\.trycloudflare\.com' /tmp/cf.log | tail -1)
  echo "=== PUBLIC URL: $URL ==="
fi

# Keep script alive
wait

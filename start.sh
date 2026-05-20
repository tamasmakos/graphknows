#!/usr/bin/env bash
# Graphknows service launcher with auto-restart watchdog
# Usage: ./start.sh [--no-tunnel]
set -euo pipefail

cd "$(dirname "$0")"
set -a && source .env && set +a
export LANG=C LC_ALL=C

# ── Single-instance guard ────────────────────────────────────────────────────
PIDFILE="/tmp/graphknows_start.pid"
if [ -f "$PIDFILE" ] && kill -0 "$(cat $PIDFILE)" 2>/dev/null; then
    echo "[start.sh] Already running (PID $(cat $PIDFILE)). Exiting."
    exit 0
fi
echo $$ > "$PIDFILE"
trap 'rm -f "$PIDFILE"' EXIT

PYPATH="/root/graphknows/services:/root/graphknows/services/graphknows:/root/graphknows/services/graphknows/graphknows"
AGENT_UVICORN="/root/graphknows/services/agent/.venv/bin/uvicorn"

echo "=== Graphknows startup $(date) ==="

# ── Agent watchdog ──────────────────────────────────────────────────────────
agent_watchdog() {
  while true; do
    if curl -sf http://localhost:8002/health > /dev/null 2>&1; then
      sleep 15
      continue
    fi
    echo "[watchdog] Starting agent on :8002..."
    PYTHONPATH="$PYPATH" $AGENT_UVICORN services.agent.main:app \
      --host 0.0.0.0 --port 8002 --log-level info \
      >> /tmp/agent.log 2>&1 || true
    echo "[watchdog] Agent exited — restarting in 5s..."
    sleep 5
  done
}

# ── GraphKnows (HybridRAG) watchdog ─────────────────────────────────────────
graphknows_watchdog() {
  while true; do
    echo "[watchdog] Starting graphknows on :8025..."
    python -m graphknows.run --transport http --port 8025 \
      >> /tmp/graphknows.log 2>&1 || true
    echo "[watchdog] GraphKnows exited — restarting in 5s..."
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

# ── Search service watchdog ─────────────────────────────────────────────────
search_watchdog() {
  while true; do
    echo "[watchdog] Starting search service on :8040..."
    PYTHONPATH="$PYPATH" uvicorn services.search.main:app \
      --host 0.0.0.0 --port 8040 --log-level info \
      >> /tmp/search.log 2>&1 || true
    echo "[watchdog] Search exited — restarting in 5s..."
    sleep 5
  done
}

# Kill anything on our ports first
fuser -k 8025/tcp 2>/dev/null || true
fuser -k 8002/tcp 2>/dev/null || true
fuser -k 8040/tcp 2>/dev/null || true
# Don't kill 8080 — old frontend may still be the right one

agent_watchdog &
AGENT_PID=$!

graphknows_watchdog &
GRAPHKNOWS_PID=$!

search_watchdog &
SEARCH_PID=$!

frontend_watchdog &
FRONTEND_PID=$!

echo "Watchdogs started: agent=$AGENT_PID graphknows=$GRAPHKNOWS_PID search=$SEARCH_PID frontend=$FRONTEND_PID"

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

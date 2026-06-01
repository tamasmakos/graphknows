#!/usr/bin/env bash
# scripts/dev-local.sh — Start all Graphknows services on the host.
#
# Order:
#   1. Verify infrastructure (Neo4j, pgvector, Memgraph) via Docker
#   2. Kill any zombie app processes
#   3. Start bare-metal services with watchdog loops (background)
#   4. Wait for each to become healthy
#   5. Print final status board
#
# Usage: bash scripts/dev-local.sh [--no-tunnel] [--skip-infra-check]
set -euo pipefail

cd "$(dirname "$0")/.."
set -a; source .env; set +a
export LANG=C LC_ALL=C

NO_TUNNEL=0
SKIP_INFRA=0
for arg in "$@"; do
  case "$arg" in
    --no-tunnel)      NO_TUNNEL=1 ;;
    --skip-infra-check) SKIP_INFRA=1 ;;
  esac
done

# Colours
if [ -t 1 ]; then
  GREEN="\033[0;32m"; RED="\033[0;31m"; YELLOW="\033[0;33m"; RESET="\033[0m"; BOLD="\033[1m"
else
  GREEN=""; RED=""; YELLOW=""; RESET=""; BOLD=""
fi

log()  { echo -e "${BOLD}[gk]${RESET} $*"; }
warn() { echo -e "${YELLOW}[gk] WARN${RESET} $*"; }
die()  { echo -e "${RED}[gk] ERROR${RESET} $*"; exit 1; }

# ── 1. Infrastructure check ─────────────────────────────────────────────────
if [ "$SKIP_INFRA" -eq 0 ]; then
  log "Checking infrastructure containers..."

  infra_ok=1
  neo4j_http_port="${NEO4J_HTTP_PORT:-7474}"
  neo4j_host="${NEO4J_HOST:-172.18.0.1}"
  pgvector_host="${PGVECTOR_HOST:-172.18.0.1}"
  pgvector_port="${PGVECTOR_PORT:-5432}"
  memgraph_host="${MEMGRAPH_HOST:-172.18.0.1}"
  memgraph_port="${MEMGRAPH_BOLT_PORT:-17687}"

  curl -sf --max-time 3 "http://${neo4j_host}:${neo4j_http_port}" > /dev/null 2>&1 \
    || { warn "Neo4j not reachable at ${neo4j_host}:${neo4j_http_port}"; infra_ok=0; }

  python3 -c "import socket; s=socket.create_connection(('$pgvector_host',$pgvector_port),timeout=3); s.close()" 2>/dev/null \
    || { warn "pgvector not reachable at ${pgvector_host}:${pgvector_port}"; infra_ok=0; }

  python3 -c "import socket; s=socket.create_connection(('$memgraph_host',$memgraph_port),timeout=3); s.close()" 2>/dev/null \
    || { warn "Memgraph not reachable at ${memgraph_host}:${memgraph_port}"; infra_ok=0; }

  if [ "$infra_ok" -eq 0 ]; then
    echo ""
    echo -e "${YELLOW}Infrastructure containers are not all running.${RESET}"
    echo -e "  Start them with: ${BOLD}make infra-up${RESET}"
    echo -e "  Or bypass this check with: ${BOLD}--skip-infra-check${RESET}"
    echo ""
    read -r -p "Start infra containers now? [Y/n] " reply
    reply="${reply:-Y}"
    if [[ "$reply" =~ ^[Yy]$ ]]; then
      log "Starting infrastructure containers..."
      docker compose up -d neo4j pgvector memgraph
      log "Waiting for Neo4j to become healthy (up to 60s)..."
      for i in $(seq 1 12); do
        curl -sf --max-time 3 "http://${neo4j_host}:${neo4j_http_port}" > /dev/null 2>&1 && break
        echo -n "."
        sleep 5
      done; echo ""
      log "Waiting for pgvector (up to 30s)..."
      for i in $(seq 1 6); do
        python3 -c "import socket; s=socket.create_connection(('$pgvector_host',$pgvector_port),timeout=3); s.close()" 2>/dev/null && break
        echo -n "."; sleep 5
      done; echo ""
      log "Waiting for Memgraph (up to 30s)..."
      for i in $(seq 1 6); do
        python3 -c "import socket; s=socket.create_connection(('$memgraph_host',$memgraph_port),timeout=3); s.close()" 2>/dev/null && break
        echo -n "."; sleep 5
      done; echo ""
    else
      die "Infra not ready. Cannot start application services safely."
    fi
  else
    echo -e "  ${GREEN}OK${RESET}  Neo4j, pgvector, Memgraph all reachable."
  fi
fi

# ── 2. Kill zombie app processes ────────────────────────────────────────────
log "Clearing ports 8025, 8002, 8040..."
fuser -k 8025/tcp 2>/dev/null || true
fuser -k 8002/tcp 2>/dev/null || true
fuser -k 8040/tcp 2>/dev/null || true
pkill -f graphgen_worker 2>/dev/null || true
sleep 1

# ── 3. Service definitions ──────────────────────────────────────────────────
PYPATH="/root/graphknows/services:/root/graphknows/services/graphknows:/root/graphknows/services/graphknows/graphknows"
AGENT_UVICORN="/root/graphknows/services/agent/.venv/bin/uvicorn"
GK_PYTHON="/root/graphknows/services/agent/.venv/bin/python"

start_watchdog() {
  local name="$1" port="$2" cmd="$3" logfile="$4"
  (
    while true; do
      if curl -sf --max-time 2 "http://localhost:${port}/health" > /dev/null 2>&1 \
         || { [ "$port" = "8080" ] && curl -sf --max-time 2 "http://localhost:8080/" > /dev/null 2>&1; }; then
        sleep 15; continue
      fi
      echo "[watchdog:${name}] Starting on :${port}..."
      eval "$cmd" >> "$logfile" 2>&1 || true
      echo "[watchdog:${name}] Exited — restarting in 5s..."
      sleep 5
    done
  ) &
}

log "Starting application service watchdogs..."

start_watchdog "graphknows" "8025" \
  "PYTHONPATH=\"$PYPATH\" $GK_PYTHON -m graphknows.run --transport http --port 8025" \
  "/tmp/graphknows.log"

start_watchdog "agent" "8002" \
  "PYTHONPATH=\"$PYPATH\" $AGENT_UVICORN services.agent.main:app --host 0.0.0.0 --port 8002 --log-level info" \
  "/tmp/agent.log"

start_watchdog "search" "8040" \
  "PYTHONPATH=\"$PYPATH\" uvicorn services.search.main:app --host 0.0.0.0 --port 8040 --log-level info" \
  "/tmp/search.log"

start_watchdog "frontend" "8080" \
  "AGENT_URL=http://localhost:8002 uvicorn apps.web.server:app --host 0.0.0.0 --port 8080 --log-level info" \
  "/tmp/frontend.log"

# ── 4. Wait for app services to become healthy ──────────────────────────────
log "Waiting for services to become healthy (up to 90s)..."

wait_for_http() {
  local name="$1" url="$2" attempts="${3:-18}"
  for i in $(seq 1 "$attempts"); do
    curl -sf --max-time 3 "$url" > /dev/null 2>&1 && return 0
    echo -n "."
    sleep 5
  done
  echo ""
  warn "$name did not become healthy in time. Check /tmp/${name}.log"
  return 1
}

echo -n "  graphknows"
wait_for_http "graphknows" "http://localhost:8025/health" && echo -e " ${GREEN}ready${RESET}"
echo -n "  agent"
wait_for_http "agent" "http://localhost:8002/health" && echo -e " ${GREEN}ready${RESET}"
echo -n "  search"
wait_for_http "search" "http://localhost:8040/api/v1/health" && echo -e " ${GREEN}ready${RESET}"
echo -n "  frontend"
wait_for_http "frontend" "http://localhost:8080/" && echo -e " ${GREEN}ready${RESET}"

echo ""

# ── 5. Cloudflare tunnel ────────────────────────────────────────────────────
if [ "$NO_TUNNEL" -eq 0 ] && command -v /tmp/cloudflared &>/dev/null; then
  log "Starting Cloudflare tunnel..."
  /tmp/cloudflared tunnel --url http://localhost:8080 > /tmp/cf.log 2>&1 &
  sleep 5
  URL=$(grep -o 'https://[a-z0-9-]*\.trycloudflare\.com' /tmp/cf.log | tail -1)
  [ -n "$URL" ] && echo -e "  ${GREEN}Public URL: $URL${RESET}"
fi

# ── 6. Final status board ───────────────────────────────────────────────────
echo ""
bash "$(dirname "$0")/health.sh"

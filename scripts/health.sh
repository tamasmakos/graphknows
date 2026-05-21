#!/usr/bin/env bash
# scripts/health.sh — Graphknows full-stack health check
# Checks infrastructure (Docker) + application services (bare-metal).
# Exit 0 only when all required services are healthy.
# Usage: bash scripts/health.sh [--quiet]
set -euo pipefail

QUIET="${1:-}"
FAIL=0

# Colours (disabled when not a TTY)
if [ -t 1 ]; then
  GREEN="\033[0;32m"; RED="\033[0;31m"; YELLOW="\033[0;33m"; RESET="\033[0m"; BOLD="\033[1m"
else
  GREEN=""; RED=""; YELLOW=""; RESET=""; BOLD=""
fi

ok()   { echo -e "  ${GREEN}OK${RESET}    $1"; }
fail() { echo -e "  ${RED}FAIL${RESET}  $1"; FAIL=1; }
warn() { echo -e "  ${YELLOW}WARN${RESET}  $1"; }

http_check() {
  local label="$1" url="$2"
  if curl -sf --max-time 3 "$url" > /dev/null 2>&1; then
    ok "$label  ($url)"
  else
    fail "$label  ($url)"
  fi
}

tcp_check() {
  local label="$1" host="$2" port="$3"
  if nc -z -w3 "$host" "$port" 2>/dev/null; then
    ok "$label  ($host:$port)"
  else
    fail "$label  ($host:$port)"
  fi
}

pg_check() {
  local label="$1" host="$2" port="$3" user="$4" db="$5"
  if pg_isready -h "$host" -p "$port" -U "$user" -d "$db" -q 2>/dev/null; then
    ok "$label  ($host:$port)"
  elif nc -z -w3 "$host" "$port" 2>/dev/null; then
    warn "$label  ($host:$port — TCP open, pg_isready not available)"
  else
    fail "$label  ($host:$port)"
  fi
}

# Load .env for host/port values (silently — health.sh may be called standalone)
if [ -f "$(dirname "$0")/../.env" ]; then
  set -a; source "$(dirname "$0")/../.env"; set +a
fi

NEO4J_HOST="${NEO4J_HOST:-172.18.0.1}"
NEO4J_HTTP_PORT="${NEO4J_HTTP_PORT:-7474}"
NEO4J_BOLT_PORT="${NEO4J_BOLT_PORT:-7687}"
PGVECTOR_HOST="${PGVECTOR_HOST:-172.18.0.1}"
PGVECTOR_PORT="${PGVECTOR_PORT:-5432}"
PGVECTOR_USER="${PGVECTOR_USER:-postgres}"
PGVECTOR_DB="${PGVECTOR_DB:-vectordb}"
MEMGRAPH_HOST="${MEMGRAPH_HOST:-172.18.0.1}"
MEMGRAPH_BOLT_PORT="${MEMGRAPH_BOLT_PORT:-7688}"

echo ""
echo -e "${BOLD}=== Graphknows Health Check ===${RESET}  $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

# ── Infrastructure (Docker containers) ─────────────────────────────────────
echo -e "${BOLD}Infrastructure${RESET}"
http_check "Neo4j HTTP   " "http://${NEO4J_HOST}:${NEO4J_HTTP_PORT}"
tcp_check  "Neo4j Bolt   " "${NEO4J_HOST}" "${NEO4J_BOLT_PORT}"
pg_check   "pgvector     " "${PGVECTOR_HOST}" "${PGVECTOR_PORT}" "${PGVECTOR_USER}" "${PGVECTOR_DB}"
tcp_check  "Memgraph Bolt" "${MEMGRAPH_HOST}" "${MEMGRAPH_BOLT_PORT}"
echo ""

# ── Application services (bare-metal) ──────────────────────────────────────
echo -e "${BOLD}Application Services${RESET}"
http_check "graphknows   " "http://localhost:8025/health"
http_check "agent        " "http://localhost:8002/health"
http_check "search       " "http://localhost:8040/health"
http_check "frontend     " "http://localhost:8080/"
echo ""

# ── Result ─────────────────────────────────────────────────────────────────
if [ "$FAIL" -eq 0 ]; then
  echo -e "${GREEN}${BOLD}All services healthy.${RESET}"
  echo ""
  exit 0
else
  echo -e "${RED}${BOLD}One or more services are down. See FAIL lines above.${RESET}"
  echo -e "  Run ${BOLD}make dev-local${RESET} to start bare-metal services."
  echo -e "  Run ${BOLD}make infra-up${RESET} to start infrastructure containers."
  echo ""
  exit 1
fi

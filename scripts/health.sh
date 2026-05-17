#!/usr/bin/env bash
# scripts/health.sh — check every Graphknows service endpoint
# Works with both Docker Compose (hostname = service name) and bare-metal (localhost).
# Exit code 0 = all healthy, 1 = at least one failure.

set -euo pipefail

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

ok()   { printf "  ${GREEN}✔${NC}  %-28s %s\n" "$1" "$2"; }
fail() { printf "  ${RED}✗${NC}  %-28s %s\n" "$1" "$2"; }
warn() { printf "  ${YELLOW}~${NC}  %-28s %s\n" "$1" "$2"; }

check() {
  local name=$1 url=$2
  local code
  code=$(curl -sf -o /dev/null -w "%{http_code}" --max-time 3 "$url" 2>/dev/null) || code="ERR"
  if [[ "$code" =~ ^2 ]]; then
    ok "$name" "${CYAN}$url${NC} → HTTP $code"
  elif [[ "$code" == "ERR" ]]; then
    fail "$name" "${CYAN}$url${NC} → no response"
    return 1
  else
    warn "$name" "${CYAN}$url${NC} → HTTP $code"
    return 1
  fi
}

# Detect whether we're talking to Docker Compose services or bare-metal
if docker compose ps --services 2>/dev/null | grep -q "agent"; then
  HOST_MAP=(
    "neo4j|http://localhost:7474"
    "pgvector|http://localhost:5432"          # TCP only — curl will get 'no response'; skip
    "memgraph|http://localhost:7444"
    "hybridrag|http://localhost:8020/health"
    "graphgen|http://localhost:8020/health"
    "agent|http://localhost:8002/health"
    "search|http://localhost:8040/api/v1/health"
    "frontend|http://localhost:8080/"
  )
else
  # bare-metal (start.sh) uses different ports for agent/hybridrag
  HOST_MAP=(
    "hybridrag|http://localhost:8020/health"
    "agent|http://localhost:8002/health"
    "search|http://localhost:8040/api/v1/health"
    "frontend|http://localhost:8080/"
  )
fi

printf "\n${BOLD}Graphknows Service Health${NC}  ($(date '+%H:%M:%S'))\n"
printf '%0.s─' {1..55}; echo

FAILURES=0
for entry in "${HOST_MAP[@]}"; do
  name="${entry%%|*}"
  url="${entry##*|}"
  # pgvector has no HTTP — just show a note
  if [[ "$name" == "pgvector" ]]; then
    warn "pgvector" "TCP only — use: psql \$PGVECTOR_DSN"
    continue
  fi
  check "$name" "$url" || FAILURES=$((FAILURES + 1))
done

printf '%0.s─' {1..55}; echo
if [[ $FAILURES -eq 0 ]]; then
  printf "${GREEN}${BOLD}All services healthy.${NC}\n\n"
else
  printf "${RED}${BOLD}$FAILURES service(s) not responding.${NC}\n\n"
fi

exit $FAILURES

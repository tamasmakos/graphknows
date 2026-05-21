#!/usr/bin/env bash
# scripts/preflight.sh — Validate env and infra before starting services.
# Exits 0 (GO) only if everything required is present.
# Safe to run repeatedly.
set -euo pipefail

cd "$(dirname "$0")/.."
export LANG=C LC_ALL=C

# Colours (only when connected to a terminal)
if [ -t 1 ]; then
  OK="\\033[0;32m OK \\033[0m"; FAIL="\\033[0;31mFAIL\\033[0m"
  WARN="\\033[0;33mWARN\\033[0m"; BOLD="\\033[1m"; RESET="\\033[0m"
else
  OK=" OK "; FAIL="FAIL"; WARN="WARN"; BOLD=""; RESET=""
fi

row() { printf "  [%s] %s\\n" "$1" "$2"; }
pass=1

echo ""
echo -e "${BOLD}Graphknows preflight check${RESET}"
echo "────────────────────────────────────────"

# ── 1. .env file ─────────────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}Environment${RESET}"
if [ -f ".env" ]; then
  row "$OK" ".env file exists"
  set -a; source .env; set +a
else
  row "$FAIL" ".env missing  →  cp .env.example .env  then fill in the required keys"
  pass=0
fi

required_vars=(
  "OPENROUTER_API_KEY"
  "NEO4J_PASSWORD"
  "PGVECTOR_PASSWORD"
)
for var in "${required_vars[@]}"; do
  if [ -n "${!var:-}" ]; then
    row "$OK" "$var is set"
  else
    row "$FAIL" "$var is NOT set in .env"
    pass=0
  fi
done

# ── 2. Python ─────────────────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}Python${RESET}"
py_ver=$(python3 --version 2>&1 || echo "missing")
if python3 -c "import sys; sys.exit(0 if sys.version_info >= (3,11) else 1)" 2>/dev/null; then
  row "$OK" "$py_ver"
else
  row "$FAIL" "Python >=3.11 required, found: $py_ver"
  pass=0
fi

if command -v uv &>/dev/null; then
  row "$OK" "uv $(uv --version 2>&1 | head -1)"
else
  row "$FAIL" "uv not found — install: curl -Ls https://astral.sh/uv/install.sh | sh"
  pass=0
fi

# ── 3. Docker ─────────────────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}Docker${RESET}"
if docker info &>/dev/null; then
  row "$OK" "Docker daemon reachable"
elif docker ps &>/dev/null 2>&1; then
  row "$OK" "Docker reachable (via docker ps)"
else
  # Inside some containers docker socket isn't available but infra may still be up via host.
  # Treat as a warning — infra checks below will catch actual connectivity failures.
  row "$WARN" "Docker socket not reachable from this environment (infra checks below will verify)"
fi

# ── 4. Infrastructure reachability ────────────────────────────────────────────
echo ""
echo -e "${BOLD}Infrastructure${RESET}"

neo4j_host="${NEO4J_HOST:-172.18.0.1}"
neo4j_http="${NEO4J_HTTP_PORT:-7474}"
pg_host="${PGVECTOR_HOST:-172.18.0.1}"
pg_port="${PGVECTOR_PORT:-5432}"
mg_host="${MEMGRAPH_HOST:-172.18.0.1}"
mg_port="${MEMGRAPH_BOLT_PORT:-17687}"

tcp_check() {
  python3 -c "import socket; s=socket.create_connection(('$1',$2),timeout=3); s.close()" 2>/dev/null
}

if curl -sf --max-time 3 "http://${neo4j_host}:${neo4j_http}" > /dev/null 2>&1; then
  row "$OK" "Neo4j HTTP  ${neo4j_host}:${neo4j_http}"
else
  row "$FAIL" "Neo4j HTTP  ${neo4j_host}:${neo4j_http}  not reachable — run: make infra-up"
  pass=0
fi

if tcp_check "$pg_host" "$pg_port"; then
  row "$OK" "pgvector    ${pg_host}:${pg_port}"
else
  row "$FAIL" "pgvector    ${pg_host}:${pg_port}  not reachable — run: make infra-up"
  pass=0
fi

if tcp_check "$mg_host" "$mg_port"; then
  row "$OK" "Memgraph    ${mg_host}:${mg_port}"
else
  row "$FAIL" "Memgraph    ${mg_host}:${mg_port}  not reachable — run: make infra-up"
  pass=0
fi

# ── 5. Port availability ──────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}App ports${RESET}"
app_ports=(8025 8002 8040 8080)
for port in "${app_ports[@]}"; do
  if python3 -c "import socket; s=socket.socket(); s.bind(('',${port})); s.close()" 2>/dev/null; then
    row "$OK" ":${port} free"
  else
    row "$WARN" ":${port} already in use (service may already be running)"
    # Not a hard failure — the watchdog handles this
  fi
done

# ── Result ────────────────────────────────────────────────────────────────────
echo ""
echo "────────────────────────────────────────"
if [ "$pass" -eq 1 ]; then
  echo -e "  ${BOLD}GO${RESET} — all checks passed."
  echo ""
  exit 0
else
  echo -e "  ${BOLD}NO-GO${RESET} — fix the issues above, then re-run."
  echo ""
  exit 1
fi

#!/usr/bin/env bash
set -euo pipefail

AGENT_URL="${AGENT_URL:-http://localhost:8002}"
PORT="${PORT:-8080}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

echo "=== Graphknows Frontend v2 ==="

# 1. Check agent health
echo -n "Checking agent at $AGENT_URL/health … "
if curl -sf "$AGENT_URL/health" > /dev/null 2>&1; then
  echo "OK ✅"
else
  echo "OFFLINE ⚠️  (agent not running — UI will show offline message)"
fi

# 2. Start FastAPI server in background
echo "Starting FastAPI server on port $PORT …"
cd "$REPO_ROOT"
uvicorn apps.web.server:app --host 0.0.0.0 --port "$PORT" &
SERVER_PID=$!
echo "Server PID: $SERVER_PID"
sleep 2

# 3. Open ngrok tunnel
echo "Opening ngrok tunnel …"
PUBLIC_URL=$(python3 -c "
from pyngrok import ngrok
t = ngrok.connect($PORT, bind_tls=True)
print(t.public_url)
" 2>/dev/null) || true

if [ -n "$PUBLIC_URL" ]; then
  echo ""
  echo "🌐 PUBLIC URL: $PUBLIC_URL"
  echo ""
else
  echo "pyngrok failed, falling back to cloudflared …"
  CF=/tmp/cloudflared
  if [ ! -f "$CF" ]; then
    curl -fsSL https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -o "$CF"
    chmod +x "$CF"
  fi
  "$CF" tunnel --url "http://localhost:$PORT" &
  sleep 5
  echo "(See cloudflared output above for URL)"
fi

# 4. Keep running
echo "Press Ctrl+C to stop."
tail -f /dev/null

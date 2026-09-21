#!/usr/bin/env bash
# Start the Streamlit app and open a public Cloudflare quick tunnel to it; print the shareable URL.
# The URL is random (https://<words>.trycloudflare.com), needs no account, and lives as long as this VM does.
set -euo pipefail

APP_DIR="${APP_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"
PORT="${PORT:-8501}"
LOG_DIR="${LOG_DIR:-/tmp/agent-logs}"
mkdir -p "$LOG_DIR"

# Ollama must be up (bootstrap starts it; a notebook restart may have killed it)
if ! curl -fs http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then
  setsid nohup ollama serve > "$LOG_DIR/ollama.log" 2>&1 < /dev/null &
  for i in $(seq 1 30); do curl -fs http://127.0.0.1:11434/api/tags >/dev/null 2>&1 && break; sleep 1; done
fi
mysqladmin ping --silent 2>/dev/null || service mysql start >/dev/null 2>&1 || true

if curl -fs http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then
  echo "Ollama: running - models: $(ollama list 2>/dev/null | awk 'NR>1{print $1}' | tr '
' ' ')"
else
  echo "Ollama: NOT running - last log lines:"; tail -10 "$LOG_DIR/ollama.log" 2>/dev/null || true
fi
mysqladmin ping --silent 2>/dev/null && echo "MySQL: running" || echo "MySQL: NOT running"

pkill -f "streamlit run app.py" 2>/dev/null || true
pkill -f "cloudflared tunnel" 2>/dev/null || true

cd "$APP_DIR"
setsid nohup python -m streamlit run app.py --server.headless true --server.port "$PORT" --server.address 0.0.0.0 \
  --browser.gatherUsageStats false > "$LOG_DIR/streamlit.log" 2>&1 &
for i in $(seq 1 60); do curl -fs "http://127.0.0.1:$PORT/_stcore/health" >/dev/null 2>&1 && break; sleep 1; done

if ! command -v cloudflared >/dev/null 2>&1; then
  echo "cloudflared missing - installing"
  curl -fsSL -o /tmp/cloudflared.deb https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb
  dpkg -i /tmp/cloudflared.deb > /dev/null || apt-get install -y -qq -f > /dev/null
fi
if ! curl -fs "http://127.0.0.1:$PORT/_stcore/health" >/dev/null 2>&1; then
  echo "Streamlit is not answering on port $PORT - last log lines:"; tail -20 "$LOG_DIR/streamlit.log" || true
fi

URL=""
for attempt in 1 2 3; do
  pkill -f "cloudflared tunnel" 2>/dev/null || true
  : > "$LOG_DIR/cloudflared.log"
  setsid nohup cloudflared tunnel --url "http://127.0.0.1:$PORT" --no-autoupdate --protocol http2     > "$LOG_DIR/cloudflared.log" 2>&1 < /dev/null &
  for i in $(seq 1 45); do
    URL=$(grep -oE 'https://[a-z0-9-]+\.trycloudflare\.com' "$LOG_DIR/cloudflared.log" | head -1 || true)
    [ -n "$URL" ] && break
    sleep 1
  done
  [ -n "$URL" ] && break
  echo "tunnel attempt $attempt did not produce a URL; retrying..."
done

echo
echo "======================================================================"
if [ -n "$URL" ]; then
  echo "  Share this link:  $URL"
  echo "  (first answer is slow while the models load into the GPU; logs in $LOG_DIR)"
else
  echo "  Tunnel URL not found. cloudflared log:"
  tail -25 "$LOG_DIR/cloudflared.log" || true
  echo "  Fallback (no account): run in a new cell:"
  echo "    !ssh -o StrictHostKeyChecking=no -p 443 -R0:localhost:$PORT a.pinggy.io"
fi
echo "======================================================================"
